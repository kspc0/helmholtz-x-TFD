'''
objective: calculate shape derivative for a simple 2D rijke tube using continuous adjoint approach
with full border displacement of the inlet
'''

import datetime
import os
import sys

import gmsh
import numpy as np
from mpi4py import MPI

# Parameters of the problem
import rparams
# HelmholtzX utilities
from helmholtz_x.io_utils import XDMFReader, dict_writer, xdmf_writer, write_xdmf_mesh # to write mesh data as files
from helmholtz_x.parameters_utils import sound_speed # to calculate sound speed from temperature
from helmholtz_x.acoustic_matrices import AcousticMatrices # to assemble the acoustic matrices for discrete Helm. EQU
from helmholtz_x.flame_transfer_function import nTau # to define the flame transfer function
from helmholtz_x.flame_matrices import DistributedFlameMatrix # to define the flame matrix for discrete Helm. EQU
from helmholtz_x.eigensolvers import fixed_point_iteration, eps_solver, newtonSolver # to solve the system
from helmholtz_x.dolfinx_utils import absolute # to get the absolute value of a function
from helmholtz_x.shape_derivatives import ShapeDerivativesFFDRectFullBorder, ffd_displacement_vector_rect_full_border # to calculate shape derivatives

# mark the processing time
start_time = datetime.datetime.now()
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/Meshes" # folder of mesh file
mesh_name = "/RijkeMesh" # name of the mesh file
results_dir = "/Results" # folder for saving results


#--------------------------MAIN PARAMETERS-------------------------#
mesh_resolution = 0.008 # specify mesh resolution
tube_length = 1 # length of the duct
tube_height = 0.1 #0.047 # height of the duct
degree = 2 # the higher the degree, the longer the calulation takes but the more precise it is
frequ = 100 # where to expect first mode in Hz
homogeneous_case = False # True for homogeneous case, False for inhomogeneous case


#--------------------------CREATE MESH----------------------------#
print("\n--- CREATING MESH ---")
gmsh.initialize() # start the gmsh session
gmsh.model.add("RijkeTube") # add the model name
# locate the points of the 2D geometry: [m]
p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_resolution)  
p2 = gmsh.model.geo.addPoint(0, tube_height, 0, mesh_resolution) # 0.1m high
p3 = gmsh.model.geo.addPoint(tube_length, tube_height, 0, mesh_resolution) # 1m long
p4 = gmsh.model.geo.addPoint(tube_length, 0, 0, mesh_resolution)
# create outlines by connecting points
l1 = gmsh.model.geo.addLine(p1, p2) # inlet boundary
l2 = gmsh.model.geo.addLine(p2, p3) # upper wall
l3 = gmsh.model.geo.addLine(p3, p4) # outlet boundary
l4 = gmsh.model.geo.addLine(p4, p1) # lower wall
# create extra points to outline the plenum (needed for shape derivation of upper plenum wall)
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
# if '-nopopup' not in sys.argv:
#    gmsh.fltk.run() 
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
# set boundary conditions case
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Dirichlet'}, # outlet
                        3:  {'Neumann'}, # upper wall
                        4:  {'Neumann'}} # lower wall
# initialize parameters for homogeneous or inhomogeneous case
if homogeneous_case: # homogeneous case
    T_output = rparams.T_in
    Rho_output = rparams.rho_u
else: # inhomogeneous case
    T_output = rparams.T_out
    Rho_output = rparams.rho_d

# define temperature gradient function in geometry
T = rparams.temperature_step_gauss_plane(mesh, rparams.x_f, rparams.T_in, T_output, rparams.amplitude, rparams.sig)
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
rho = rparams.rhoFunctionPlane(mesh, rparams.x_f, rparams.a_f, Rho_output, rparams.rho_u, rparams.amplitude, rparams.sig, rparams.limit)
# use differnet functions depending on the case for heat release h and measurement w
if homogeneous_case:
    w = rparams.gaussianFunctionHplaneHomogenous(mesh, rparams.x_r, rparams.a_r, rparams.amplitude, rparams.sig) 
    h = rparams.gaussianFunctionHplaneHomogenous(mesh, rparams.x_f, rparams.a_f, rparams.amplitude, rparams.sig) 
else:
    w = rparams.gaussianFunctionHplane(mesh, rparams.x_r, rparams.a_r, rparams.amplitude, rparams.sig) 
    h = rparams.gaussianFunctionHplane(mesh, rparams.x_f, rparams.a_f, rparams.amplitude, rparams.sig) 
# calculate the flame matrix
D = DistributedFlameMatrix(mesh, w, h, rho, T, rparams.q_0, rparams.u_b, FTF, degree=degree, gamma=rparams.gamma)


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


#-------------------CALCULATE SHAPE DERIVATIVES-------------------#
print("\n--- CALCULATING SHAPE DERIVATIVES ---")
# 1 for inlet
# 2 for outlet
physical_facet_tags = {1: 'inlet', 2: 'outlet'}
selected_facet_tag = 2 # tag of the wall to be displaced
# [-1,0] for inlet
# [1,0] for outlet
norm_vector_inlet = [1,0] # normal outside vector of the wall to be displaced

# visualize example displacement field for full displaced border
V_ffd = ffd_displacement_vector_rect_full_border(Rijke, selected_facet_tag, norm_vector_inlet, deg=1)
xdmf_writer(path+"/InputFunctions/V_ffd", mesh, V_ffd)

# calculate the shape derivatives for the border
print("- calculating shape derivative")
derivative = ShapeDerivativesFFDRectFullBorder(Rijke, selected_facet_tag, norm_vector_inlet, omega_dir, p_dir, p_adj, c, matrices, D)


#--------------------------FINALIZING-----------------------------#
# print most important parameters and results of calculation
print("\n")
if omega_dir.imag > 0: 
    stability = 'instable'
else:
    stability = 'stable'
print(f"---> \033[1mDisplaced Wall\033[0m: {physical_facet_tags[selected_facet_tag]}")
print(f"---> \033[1mMesh Resolution =\033[0m {mesh_resolution}")
print(f"---> \033[1mDimensions =\033[0m {tube_length}m, {tube_height} m")
print(f"---> \033[1mPolynomial Degree of FEM =\033[0m {degree}")
print(f"---> \033[1mTarget =\033[0m {frequ} Hz ")
print(f"---> \033[1mEigenfrequency =\033[0m {round(omega_dir.real/2/np.pi,4)} + {round(omega_dir.imag/2/np.pi,4)}j Hz ({stability})")
print(f"---> \033[1mContinuous Shape Derivative =\033[0m {round(derivative.real/2/np.pi,8)} + {round(derivative.imag/2/np.pi,8)}j")
# close the gmsh session which was required to run for calculating shape derivatives
gmsh.finalize()
# mark the processing time:
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
