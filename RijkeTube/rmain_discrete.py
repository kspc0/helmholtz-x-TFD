'''
objective: calculate shape derivative for a 2D rijke tube using discete 
adjoint approach with full border displacement of the outlet
'''

import datetime
import os
import sys

import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# parameters of the problem
import rparams
# HelmholtzX utilities
from helmholtz_x.io_utils import XDMFReader, dict_writer, xdmf_writer, write_xdmf_mesh # to write mesh data as files
from helmholtz_x.parameters_utils import sound_speed # to calculate sound speed from temperature
from helmholtz_x.acoustic_matrices import AcousticMatrices # to assemble the acoustic matrices for discrete Helm. EQU
from helmholtz_x.flame_transfer_function import nTau, stateSpace # to define the flame transfer function
from helmholtz_x.flame_matrices import DistributedFlameMatrix # to define the flame matrix for discrete Helm. EQU
from helmholtz_x.eigensolvers import fixed_point_iteration, eps_solver, newtonSolver # to solve the system
from helmholtz_x.dolfinx_utils import absolute # to get the absolute value of a function
from helmholtz_x.petsc4py_utils import conjugate_function, vector_matrix_vector

# mark the processing time
start_time = datetime.datetime.now()
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/Meshes" # folder of mesh file
mesh_name = "/RijkeMesh" # name of the mesh file
perturbed_mesh_name = "/RijkePerturbedMesh" # name of the perturbed mesh file
results_dir = "/Results" # folder for saving results


#--------------------------MAIN PARAMETERS-------------------------#
mesh_resolution = 0.008 # specify mesh resolution
tube_length = 1 # length of the tube
tube_height = 0.047 # height of the tube
degree = 2 # degree of FEM polynomials
frequ = 200 # target frequency - where to expect first mode in Hz
perturbation = 0.001 # perturbation distance of the mesh
homogeneous_case = False # set True for homogeneous case, False for inhomogeneous case
# set boundary conditions
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Neumann'}, # outlet
                        3:  {'Neumann'}, # upper wall
                        4:  {'Neumann'}} # lower wall


#--------------------------CREATE MESH----------------------------#
print("\n--- CREATING MESH ---")
gmsh.initialize() # start the gmsh session
gmsh.model.add("RijkeCase") # add the model name
# locate the points of the 2D geometry: [m]
p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_resolution)  
p2 = gmsh.model.geo.addPoint(0, tube_height, 0, mesh_resolution)
p3 = gmsh.model.geo.addPoint(tube_length, tube_height, 0, mesh_resolution)
p4 = gmsh.model.geo.addPoint(tube_length, 0, 0, mesh_resolution)
# create outlines by connecting points
l1 = gmsh.model.geo.addLine(p1, p2) # inlet boundary
l2 = gmsh.model.geo.addLine(p2, p3) # upper wall
l3 = gmsh.model.geo.addLine(p3, p4) # outlet boundary
l4 = gmsh.model.geo.addLine(p4, p1) # lower wall
# create curve loops for surface
loop1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4]) # entire geometry
# create surfaces from the curved loops
surface1 = gmsh.model.geo.addPlaneSurface([loop1]) # surface of entire geometry
# assign physical tags for 1D boundaries
gmsh.model.addPhysicalGroup(1, [l1], tag=1) # inlet boundary
gmsh.model.addPhysicalGroup(1, [l3], tag=2) # outlet boundary
gmsh.model.addPhysicalGroup(1, [l4], tag=3) # lower wall
gmsh.model.addPhysicalGroup(1, [l2], tag=4) # upper wall
# assign physical tag for 2D surface
gmsh.model.addPhysicalGroup(2, [surface1], tag=1)
# create 2D mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
# optionally launch GUI to see the results
#if '-nopopup' not in sys.argv:
#   gmsh.fltk.run() 
# save data in /Meshes directory
gmsh.write("{}.msh".format(path+mesh_dir+mesh_name)) # save as .msh file
write_xdmf_mesh(path+mesh_dir+mesh_name,dimension=2) # save as .xdmf file


#--------------------------LOAD MESH DATA-------------------------#
print("\n--- LOADING MESH ---")
Rijke = XDMFReader(path+mesh_dir+mesh_name)
mesh, subdomains, facet_tags = Rijke.getAll() # mesh, domains and tags
Rijke.getInfo()


#--------------------ASSEMBLE PASSIVE MATRICES--------------------#
print("\n--- ASSEMBLING PASSIVE MATRICES ---")
# initialize parameters for homogeneous or inhomogeneous case
if homogeneous_case: # homogeneous case
    T_output = rparams.T_in # no temperature gradient
    Rho_output = rparams.rho_u # no density gradient
else: # inhomogeneous case
    T_output = rparams.T_out
    Rho_output = rparams.rho_d
# distribute temperature gradient as function on the geometry
T = rparams.temperature_step_gauss_plane(mesh, rparams.x_f, rparams.T_in, T_output)
# calculate the sound speed function from temperature
c = sound_speed(T)
# calculate the passive acoustic matrices
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T , degree = degree) # very large, sparse matrices


#-------------------ASSEMBLE FLAME TRANSFER MATRIX----------------#
print("\n--- ASSEMBLING FLAME MATRIX ---")
# using nTau model to define the flame transfer function
FTF = nTau(rparams.n, rparams.tau)
# define input functions for the flame matrix
# density function:
rho = rparams.rhoFunctionPlane(mesh, rparams.x_f, rparams.a_f, Rho_output, rparams.rho_u)
# use differnet functions for heat release h(x) and measurement w(x)
if homogeneous_case:
    w = rparams.gaussianFunctionHplaneHomogenous(mesh, rparams.x_r, rparams.a_r)
    h = rparams.gaussianFunctionHplaneHomogenous(mesh, rparams.x_f, rparams.a_r)
else:
    w = rparams.gaussianFunctionHplane(mesh, rparams.x_r, rparams.a_r) 
    h = rparams.gaussianFunctionHplane(mesh, rparams.x_f, rparams.a_f) 
# calculate the flame matrix
D = DistributedFlameMatrix(mesh, w, h, rho, T, rparams.q_0, rparams.u_b, FTF, degree=degree, gamma=rparams.gamma)


#-------------------SOLVE THE DISCRETE SYSTEM---------------------#
print("\n--- STARTING NEWTON METHOD ---")
# unit of target: ([Hz])*2*pi = [rad/s] 
target = (frequ)*2*np.pi
print(f"---> \033[1mTarget\033[0m: {target.real:.2f}  {target.imag:.2f}j")
# solve using Newton's method and the parameters:
# i: index of the eigenvalue (i=0 is closest eigenvalue to target)
# nev: number of eigenvalues to find in close range to target
# tol: tolerance of the solution
# maxiter: maximum number of iterations
try:
    # direct problem
    print("\n- DIRECT PROBLEM -")
    D.assemble_submatrices('direct') # assemble direct flame matrix
    # calculate the eigenvalues and eigenvectors
    omega_dir, p_dir = newtonSolver(matrices, D, target, nev=3, i=0, tol=1e-2, degree=degree, maxiter=70, problem_type='direct', print_results= False)
    print("- omega_dir:", omega_dir)
    # adjoint problem
    print("\n- ADJOINT PROBLEM -")
    D.assemble_submatrices('adjoint') # assemble adjoint flame matrix
    # calculate the eigenvalues and eigenvectors
    omega_adj, p_adj = newtonSolver(matrices, D, target, nev=3, i=0, tol=1e-2, degree=degree, maxiter=70, problem_type='adjoint',print_results= False)
    print("- omega_adj:", omega_adj)
except IndexError:
    print("XXX--IndexError--XXX") # convergence of target failed in given range of iterations and tolerance
else:
    print("-> Iterations done - target converged successfully")


#-------------------POSTPROCESSING AND SAVING---------------------#
print("\n--- POSTPROCESSING ---")
# save solutions in a dictionary with labels and values
omega_dict = {'direct':omega_dir, 'adjoint': omega_adj}
dict_writer(path+results_dir+"/eigenvalues", omega_dict) # save as .txt
# save direct and adjoint eigenvector solution as .xdmf files
xdmf_writer(path+results_dir+"/p_dir", mesh, p_dir)
xdmf_writer(path+results_dir+"/p_dir_abs", mesh, absolute(p_dir))
xdmf_writer(path+results_dir+"/p_adj", mesh, p_adj)
xdmf_writer(path+results_dir+"/p_adj_abs", mesh, absolute(p_adj))


#-------------------PERTURBING THE MESH---------------------------#
print("\n--- PERTURBING THE MESH ---")
# for discrete shape derivatives the mesh needs to be perturbed
# read tags and coordinates of the mesh
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
# assign x,y,z coordinates to separate arrays
xcoords = node_coords[0::3] # get x-coordinates
ycoords = node_coords[1::3] # get y-coordinates
zcoords = node_coords[2::3] # get z-coordinates
# perturb the chosen mesh points slightly in x direction
for i, x in enumerate(xcoords):
    if x > 0.3: # only perturb parts of mesh that lie behind the flame located at 0.25m
        xcoords[i] += (x - 0.3)/(tube_length - 0.3) * perturbation
#xcoords += xcoords * perturbation #(case for perturbing the whole duct disregarding the flame)
# update node x coordinates in mesh to the perturbed points
node_coords[0::3] = xcoords
# update node positions
for tag, new_coords in zip(node_tags, node_coords.reshape(-1,3)):
    gmsh.model.mesh.setNode(tag, new_coords, [])
# update point positions
gmsh.model.setCoordinates(p3, perturbation + tube_length, tube_height, 0)
gmsh.model.setCoordinates(p4, perturbation + tube_length, 0, 0)
# optionally launch GUI to see the results
#if '-nopopup' not in sys.argv:
#   gmsh.fltk.run()
# save perturbed mesh data in /Meshes directory
gmsh.write("{}.msh".format(path+mesh_dir+perturbed_mesh_name)) # save as .msh file
write_xdmf_mesh(path+mesh_dir+perturbed_mesh_name,dimension=2) # save as .xdmf file


#--------------------------RELOAD MESH DATA-------------------------#
print("\n--- RELOADING MESH ---")
# reload to the perturbed mesh
perturbed_Rijke = XDMFReader(path+mesh_dir+perturbed_mesh_name)
perturbed_mesh, perturbed_subdomains, perturbed_facet_tags = perturbed_Rijke.getAll() # mesh, domains and tags
perturbed_Rijke.getInfo()


#--------------------REASSEMBLE PASSIVE MATRICES-----------------#
print("\n--- REASSEMBLING PASSIVE MATRICES ---")
# recalculate the acoustic matrices for the perturbed mesh
# define temperature gradient function in geometry
T_pert = rparams.temperature_step_gauss_plane(perturbed_mesh, rparams.x_f, rparams.T_in, T_output) # the central variable that affects is T_out! if changed to T_in we get the correct homogeneous starting case
# calculate the passive acoustic matrices
perturbed_matrices = AcousticMatrices(perturbed_mesh, perturbed_facet_tags, boundary_conditions, T_pert , degree = degree) # very large, sparse matrices


#-------------------CALCULATE SHAPE DERIVATIVES-------------------#
print("\n--- CALCULATING SHAPE DERIVATIVES ---")
print("- calculate difference perturbed matrices")
diff_A = perturbed_matrices.A - matrices.A
diff_C = perturbed_matrices.C - matrices.C
# using formula of numeric/discrete shape derivative
print("- assembling numerator matrix")
omega_square = omega_dir.real**2 - omega_dir.imag**2 + 2j*omega_dir.real*omega_dir.imag # square imaginary number
Mat_n = diff_A + omega_square * diff_C
# multiply numerator matrix with direct and
# y = PETSc.Vec().createSeq(Mat_n.getSize()[0]) # create empty vector to store the result of matrix-vector multiplication
# Mat_n.mult(p_dir.vector, y) # multiply the matrix with the direct eigenfunction
#p_adj_conj = conjugate_function(p_adj)
# # dot product
# numerator = p_adj_conj.vector.dot(y)
numerator = vector_matrix_vector(p_adj.vector, Mat_n, p_dir.vector)

# assemble flame matrix
D.assemble_submatrices('direct') # assemble direct flame matrix
print("- denominator...")
Mat_d = -2*(omega_dir)*matrices.C + D.get_derivative(omega_dir) #/2/np.pi#*338
# z = PETSc.Vec().createSeq(Mat_d.getSize()[0])
# Mat_d.mult(p_dir.vector, z)
#p_adj_conj = conjugate_function(p_adj)
# # dot product
# denominator2 = p_adj_conj.vector.dot(z)
denominator = vector_matrix_vector(p_adj.vector, Mat_d, p_dir.vector)

print("- total shape derivative...")
print("- numerator:", numerator)
print("- denominator:", denominator)
# calculate quotient of complex number
real_part = numerator.real*denominator.real + numerator.imag*denominator.imag
imag_part = (numerator.imag*denominator.real - numerator.real*denominator.imag)
derivative = numerator/denominator#(real_part + 1j*imag_part) / (denominator.real**2 + denominator.imag**2)

#derivative2 = numerator2 / denominator2
#print("- check: 1+0j==",derivative2/derivative)
print("- shape derivative1:", derivative)
#print("- shape derivative2:", derivative2)


#--------------------------FINALIZING-----------------------------#
# print most important parameters and results of calculation
print("\n")
if omega_dir.imag > 0: 
    stability = 'instable'
else:
    stability = 'stable'
print(f"---> \033[1mMesh Resolution =\033[0m {mesh_resolution}")
print(f"---> \033[1mDimensions =\033[0m {tube_length}m, {tube_height} m")
print(f"---> \033[1mPolynomial Degree of FEM =\033[0m {degree}")
print(f"---> \033[1mPerturbation Distance =\033[0m {perturbation} m")
print(f"---> \033[1mTarget =\033[0m {frequ} Hz ")
print(f"---> \033[1mEigenfrequency =\033[0m {round(omega_dir.real/2/np.pi,4)} + {round(omega_dir.imag/2/np.pi,4)}j Hz ({stability})")
print(f"---> \033[1mDiscrete Shape Derivative =\033[0m {round(derivative.real/2/np.pi,8)} + {round(derivative.imag/2/np.pi,8)}j")
# IDEA: Helmholtz number: He=2piL/lambda=(380/166*(2*np.pi*tube_length)) or maybe #/(2*np.pi)**(0.2) IDEA on imaginary part
print(f"---> \033[1mNormalized Discrete Shape Derivative =\033[0m {round(derivative.real/2/np.pi/perturbation,8)} + {round(derivative.imag/2/np.pi/perturbation,8)}j") 
# close the gmsh session which was required to run for calculating shape derivatives
gmsh.finalize()
# mark the processing time:
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
