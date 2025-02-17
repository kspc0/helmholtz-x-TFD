'''
objective: calculate shape derivatives for the Kornilov case using discrete adjoint approach
with full border displacement of the upper plenum wall
'''

import datetime
import os
import sys

import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Parameters of the problem
import kparams
# HelmholtzX utilities
from helmholtz_x.io_utils import XDMFReader, dict_writer, xdmf_writer, write_xdmf_mesh # to write mesh data as files
from helmholtz_x.parameters_utils import sound_speed # to calculate sound speed from temperature
from helmholtz_x.acoustic_matrices import AcousticMatrices # to assemble the acoustic matrices for discrete Helm. EQU
from helmholtz_x.flame_transfer_function import stateSpace # to define the flame transfer function
from helmholtz_x.flame_matrices import DistributedFlameMatrix # to define the flame matrix for discrete Helm. EQU
from helmholtz_x.eigensolvers import fixed_point_iteration, eps_solver, newtonSolver # to solve the system
from helmholtz_x.dolfinx_utils import absolute # to get the absolute value of a function
from helmholtz_x.petsc4py_utils import conjugate_function, vector_matrix_vector

# mark the processing time
start_time = datetime.datetime.now()
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/Meshes" # folder of mesh file
mesh_name = "/KornilovMesh" # name of the mesh file
perturbed_mesh_name = "/KornilovPerturbedMesh" # name of the perturbed mesh file
results_dir = "/Results" # folder for saving results


#--------------------------MAIN PARAMETERS-------------------------#
mesh_resolution = 0.2e-3 # specify mesh resolution
plenum_length = 10e-3 # length of the plenum
plenum_height = 2.5e-3 # height of the plenum
slit_height = 1e-3 # height of the slitdegree = 1 # the higher the degree, the longer the calulation takes but the more precise it is
degree = 1
frequ = 5000 # where to expect first mode in Hz
perturbation = 0.0001 # perturbation distance
homogeneous_case = False # True for homogeneous case, False for inhomogeneous case
# set boundary conditions case
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Dirichlet'}, # outlet
                        3:  {'Neumann'}, # upper combustion chamber and slit wall
                        4:  {'Neumann'}, # lower symmetry boundary
                        5:  {'Neumann'}} # upper plenum wall


#--------------------------CREATE MESH----------------------------#
print("\n--- CREATING MESH ---")
gmsh.initialize() # start the gmsh session
gmsh.model.add("KornilovCase") # add the model name
# locate the points of the 2D geometry: [m]
p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_resolution)  
p2 = gmsh.model.geo.addPoint(0, plenum_height, 0, mesh_resolution)
p3 = gmsh.model.geo.addPoint(plenum_length, plenum_height, 0, mesh_resolution)
p4 = gmsh.model.geo.addPoint(plenum_length, slit_height, 0, mesh_resolution/4) # refine the mesh at this point
p5 = gmsh.model.geo.addPoint(11e-3, slit_height, 0, mesh_resolution/4)
p6 = gmsh.model.geo.addPoint(11e-3, plenum_height, 0, mesh_resolution)
p7 = gmsh.model.geo.addPoint(37e-3, plenum_height, 0, mesh_resolution)
p8 = gmsh.model.geo.addPoint(37e-3, 0, 0, mesh_resolution)
# create outlines by connecting points
l1 = gmsh.model.geo.addLine(p1, p2) # inlet boundary
l2 = gmsh.model.geo.addLine(p2, p3) # upper plenum wall
l3 = gmsh.model.geo.addLine(p3, p4) # slit wall
l4 = gmsh.model.geo.addLine(p4, p5) # slit wall
l5 = gmsh.model.geo.addLine(p5, p6) # slit wall
l6 = gmsh.model.geo.addLine(p6, p7) # upper combustion chamber wall
l7 = gmsh.model.geo.addLine(p7, p8) # outlet boundary
l8 = gmsh.model.geo.addLine(p8, p1) # lower symmetry boundary
# create extra points to outline the plenum (needed for shape derivation of upper plenum wall)
# create curve loops for surface
loop1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4,l5,l6,l7,l8]) # entire geometry
# create surfaces from the curved loops
surface1 = gmsh.model.geo.addPlaneSurface([loop1]) # surface of entire geometry
# assign physical tags for 1D boundaries
gmsh.model.addPhysicalGroup(1, [l1], tag=1) # inlet boundary
gmsh.model.addPhysicalGroup(1, [l7], tag=2) # outlet boundary
gmsh.model.addPhysicalGroup(1, [l3,l4,l5,l6], tag=3) # upper combustion chamber and slit wall
gmsh.model.addPhysicalGroup(1, [l8], tag=4) # lower symmetry boundary
gmsh.model.addPhysicalGroup(1, [l2], tag=5) # upper wall of plenum
# assign physical tag for 2D surface
gmsh.model.addPhysicalGroup(2, [surface1], tag=1)
# create 2D mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
# optionally launch GUI to see the results
# if '-nopopup' not in sys.argv:
#    gmsh.fltk.run() 
# save data in /Meshes directory
gmsh.write("{}.msh".format(path+mesh_dir+mesh_name)) # save as .msh file
write_xdmf_mesh(path+mesh_dir+mesh_name,dimension=2) # save as .xdmf file


#--------------------------LOAD MESH DATA-------------------------#
print("\n--- LOADING MESH ---")
Kornilov = XDMFReader(path+mesh_dir+mesh_name)
mesh, subdomains, facet_tags = Kornilov.getAll() # mesh, domains and tags
Kornilov.getInfo()


#--------------------ASSEMBLE PASSIVE MATRICES--------------------#
print("\n--- ASSEMBLING PASSIVE MATRICES ---")

# initialize parameters for homogeneous or inhomogeneous case
if homogeneous_case: # homogeneous case
    T_output = kparams.T_in
    Rho_output = kparams.rho_u
else: # inhomogeneous case
    T_output = kparams.T_out
    Rho_output = kparams.rho_d

# define temperature gradient function in geometry
T = kparams.temperature_step_gauss_plane(mesh, kparams.x_f, kparams.T_in, T_output, kparams.amplitude, kparams.sig)
# calculate the sound speed function from temperature
c = sound_speed(T)
# calculate the passive acoustic matrices
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T , degree = degree) # very large, sparse matrices


#-------------------ASSEMBLE FLAME TRANSFER MATRIX----------------#
print("\n--- ASSEMBLING FLAME MATRIX ---")
# using statespace model to define the flame transfer function
FTF = stateSpace(kparams.S1, kparams.s2, kparams.s3, kparams.s4)
# define input functions for the flame matrix
# density function:
rho = kparams.rhoFunctionPlane(mesh, kparams.x_f, kparams.a_f, Rho_output, kparams.rho_u, kparams.amplitude, kparams.sig, kparams.limit)
# use differnet functions depending on the case for heat release h and measurement w
if homogeneous_case:
    w = kparams.gaussianFunctionHplaneHomogenous(mesh, kparams.x_r, kparams.a_r, kparams.amplitude, kparams.sig) 
    h = kparams.gaussianFunctionHplaneHomogenous(mesh, kparams.x_f, kparams.a_f, kparams.amplitude, kparams.sig) 
else:
    w = kparams.gaussianFunctionHplane(mesh, kparams.x_r, kparams.a_r, kparams.amplitude, kparams.sig) 
    h = kparams.gaussianFunctionHplane(mesh, kparams.x_f, kparams.a_f, kparams.amplitude, kparams.sig) 
# calculate the flame matrix
D = DistributedFlameMatrix(mesh, w, h, rho, T, kparams.q_0, kparams.u_b, FTF, degree=degree, gamma=kparams.gamma)


#-------------------SOLVE THE DISCRETE SYSTEM---------------------#
print("\n--- STARTING NEWTON METHOD ---")
# set the target (expected angular frequency of the system)
# unit of target: ([Hz])*2*pi = [rad/s] 
target = (frequ)*2*np.pi
print(f"---> \033[1mTarget\033[0m: {target.real:.2f}  {target.imag:.2f}j")
# solve using Newton's method and the parameters:
# i: index of the eigenvalue (i=0 is closest eigenvalue to target)
# nev: number of eigenvalues to calculate in close range to target
# tol: tolerance of the solution
# maxiter: maximum number of iterations
try:
    # direct problem
    print("\n- DIRECT PROBLEM -")
    D.assemble_submatrices('direct') # assemble direct flame matrix
    # calculate the eigenvalues and eigenvectors
    omega_dir, p_dir = newtonSolver(matrices, D, target, nev=3, i=0, tol=1e-2, degree=degree, maxiter=70, print_results= False)
    print("- omega_dir:", omega_dir)
    # adjoint problem
    print("\n- ADJOINT PROBLEM -")
    D.assemble_submatrices('adjoint') # assemble adjoint flame matrix
    # calculate the eigenvalues and eigenvectors
    omega_adj, p_adj = newtonSolver(matrices, D, target, nev=3, i=0, tol=1e-2, degree=degree, maxiter=70, print_results= False)
    print("- omega_adj:", omega_adj)
except IndexError:
    print("XXX--IndexError--XXX") # convergence of target failed in given range of iterations and tolerance
else:
    print("-> Iterations done - target converged successfully")


#-------------------POSTPROCESSING AND SAVING---------------------#
print("\n--- POSTPROCESSING ---")
# save solutions as a dictionary with labels and values:
omega_dict = {'direct':omega_dir, 'adjoint': omega_adj}
# save as textfile
dict_writer(path+results_dir+"/eigenvalues", omega_dict)
# save direct and adjoint eigenvector solution as xdmf files
xdmf_writer(path+results_dir+"/p_dir", mesh, p_dir) # as xdmf file for paraview analysis
xdmf_writer(path+results_dir+"/p_dir_abs", mesh, absolute(p_dir)) # also save the absolute pressure distribution
xdmf_writer(path+results_dir+"/p_adj", mesh, p_adj)
xdmf_writer(path+results_dir+"/p_adj_abs", mesh, absolute(p_adj))


#-------------------PERTURBING THE MESH---------------------------#
print("\n--- PERTURBING THE MESH ---")
# for discrete shape derivatives, the mesh needs to be perturbed
# read tags and coordinates of the mesh
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
# assign x,y,z coordinates to separate arrays
xcoords = node_coords[0::3] # get x-coordinates
ycoords = node_coords[1::3] # get y-coordinates
zcoords = node_coords[2::3] # get z-coordinates
# create list to store the indices of the plenum nodes
plenum_node_indices = []
# choose points which have smaller x coordinate then 10mm or have x coordinate of 10mm and y coordinate greater than 1mm
# these are all the points in the plenum without the slit entry
for i in range(len(xcoords)):
    if (xcoords[i] < plenum_length) or (xcoords[i] == plenum_length and ycoords[i] > slit_height):
        plenum_node_indices.append(i) # store the index of the plenum nodes in this array
# perturb the chosen mesh points slightly in y direction
# perturbation is percent based on the y-coordinate
ycoords[plenum_node_indices] += ycoords[plenum_node_indices] / plenum_height * perturbation
# update node y coordinates in mesh from the perturbed points and the unperturbed original points
node_coords[1::3] = ycoords

# update node positions
for tag, new_coords in zip(node_tags, node_coords.reshape(-1,3)):
    gmsh.model.mesh.setNode(tag, new_coords, [])
# update point positions 
gmsh.model.setCoordinates(p2, 0, perturbation + plenum_height, 0)
gmsh.model.setCoordinates(p3, plenum_length, perturbation + plenum_height, 0)
# optionally launch GUI to see the results
# if '-nopopup' not in sys.argv:
#    gmsh.fltk.run()
# save perturbed mesh data in /Meshes directory
gmsh.write("{}.msh".format(path+mesh_dir+perturbed_mesh_name)) # save as .msh file
write_xdmf_mesh(path+mesh_dir+perturbed_mesh_name,dimension=2) # save as .xdmf file


#--------------------REASSEMBLE PASSIVE MATRICES-----------------#
print("\n--- REASSEMBLING PASSIVE MATRICES ---")
# recalculate the acoustic matrices for the perturbed mesh
perturbed_Kornilov = XDMFReader(path+mesh_dir+perturbed_mesh_name)
perturbed_mesh, perturbed_subdomains, perturbed_facet_tags = perturbed_Kornilov.getAll() # mesh, domains and tags
perturbed_Kornilov.getInfo()

# define temperature gradient function in geometry
T = kparams.temperature_step_gauss_plane(perturbed_mesh, kparams.x_f, kparams.T_in, T_output, kparams.amplitude, kparams.sig) # the central variable that affects is T_out! if changed to T_in we get the correct homogeneous starting case
# calculate the sound speed function from temperature
c = sound_speed(T)
# calculate the passive acoustic matrices
perturbed_matrices = AcousticMatrices(perturbed_mesh, perturbed_facet_tags, boundary_conditions, T , degree = degree) # very large, sparse matrices


#-------------------REASSEMBLE FLAME TRANSFER MATRIX----------------#
print("\n--- REASSEMBLING FLAME MATRIX ---")
# define input functions for the flame matrix
rho = kparams.rhoFunctionPlane(perturbed_mesh, kparams.x_f, kparams.a_f, Rho_output, kparams.rho_u, kparams.amplitude, kparams.sig, kparams.limit)
if homogeneous_case:
    w = kparams.gaussianFunctionHplaneHomogenous(perturbed_mesh, kparams.x_r, kparams.a_r, kparams.amplitude, kparams.sig) 
    h = kparams.gaussianFunctionHplaneHomogenous(perturbed_mesh, kparams.x_f, kparams.a_f, kparams.amplitude, kparams.sig) 
else:
    w = kparams.gaussianFunctionHplane(perturbed_mesh, kparams.x_r, kparams.a_r, kparams.amplitude, kparams.sig) 
    h = kparams.gaussianFunctionHplane(perturbed_mesh, kparams.x_f, kparams.a_f, kparams.amplitude, kparams.sig) 
# calculate the flame matrix
D = DistributedFlameMatrix(perturbed_mesh, w, h, rho, T, kparams.q_0, kparams.u_b, FTF, degree=degree, gamma=kparams.gamma)


#-------------------CALCULATE SHAPE DERIVATIVES-------------------#
print("\n--- CALCULATING SHAPE DERIVATIVES ---")
print("- calculate perturbed matrices")
diff_A = perturbed_matrices.A - matrices.A
diff_C = perturbed_matrices.C - matrices.C

# using formula of numeric/discrete shape derivative
print("- numerator addition of matrices...")
omega_square = omega_dir.real**2 - omega_dir.imag**2 + 2j*omega_dir.real*omega_dir.imag
Mat_n = diff_A + omega_square * diff_C 

y = PETSc.Vec().createSeq(Mat_n.getSize()[0]) # create empty vector to store the result of matrix-vector multiplication
Mat_n.mult(p_dir.vector, y) # multiply the matrix with the direct eigenfunction
p_adj_conj = conjugate_function(p_adj)
# dot product
numerator = p_adj_conj.vector.dot(y)
numerator2 = vector_matrix_vector(p_adj.vector, Mat_n, p_dir.vector)

# assemble flame matrix
D.assemble_submatrices('direct') # assemble direct flame matrix
print("- denominator...")
Mat_d = -2*(omega_dir)*matrices.C + D.get_derivative(omega_dir)
z = PETSc.Vec().createSeq(Mat_d.getSize()[0])
Mat_d.mult(p_dir.vector, z)
p_adj_conj = conjugate_function(p_adj)
# dot product
denominator = p_adj_conj.vector.dot(z)
denominator2 = vector_matrix_vector(p_adj.vector, Mat_d, p_dir.vector)

print("- total shape derivative...")
print("- numerator:", numerator)
print("- denominator:", denominator)
# calculate quotient of complex number
real_part = numerator.real*denominator.real + numerator.imag*denominator.imag
imag_part = (numerator.imag*denominator.real - numerator.real*denominator.imag)
derivative = (real_part + 1j*imag_part) / (denominator.real**2 + denominator.imag**2)

derivative2 = numerator2 / denominator2
print("- check: 1==",derivative2/derivative)

#--------------------------FINALIZING-----------------------------#
# print most important parameters and results of calculation
print("\n")
if omega_dir.imag > 0:
    stability = 'instable'
else:
    stability = 'stable'
print(f"---> \033[1mMesh Resolution =\033[0m {mesh_resolution}")
print(f"---> \033[1mDimensions =\033[0m {None}m, {None} m")
print(f"---> \033[1mPolynomial Degree of FEM =\033[0m {degree}")
print(f"---> \033[1mPerturbation Distance =\033[0m {perturbation} m")
print(f"---> \033[1mTarget =\033[0m {frequ} Hz ")
print(f"---> \033[1mEigenfrequency =\033[0m {round(omega_dir.real/2/np.pi,4)} + {round(omega_dir.imag/2/np.pi,4)}j Hz ({stability})")
print(f"---> \033[1mDiscrete Shape Derivative =\033[0m {round(derivative.real/2/np.pi,8)} + {round(derivative.imag/2/np.pi,8)}j")
print(f"---> \033[1mNormalized Discrete Shape Derivative =\033[0m {round(derivative.real/2/np.pi/perturbation,8)} + {round(derivative.imag/2/np.pi/perturbation,8)}j")
# close the gmsh session which was required to run for calculating shape derivatives
gmsh.finalize()
# mark the processing time:
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
