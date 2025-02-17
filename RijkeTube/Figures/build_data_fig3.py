'''
compute data of figure3: shape optimization with discrete and continuous shape derivative for inhomogeneous Rijke tube
'''
import rparams
import os
import test_case
import datetime
import gmsh
# initialize gmsh
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
# mark the processing time
start_time = datetime.datetime.now()
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

#------------------------MAIN PARAMETERS-----------------------#
name = "/RijkeTube"
mesh_resolution = 0.008 # specify mesh resolution
tube_height = 0.047 # height of the tube
degree = 2 # degree of FEM polynomials
perturbation = 0.001 # perturbation distance of the mesh
# set boundary conditions
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Neumann'}, # outlet
                        3:  {'Neumann'}, # upper wall
                        4:  {'Neumann'}} # lower wall
# set True for homogeneous case, False for inhomogeneous case
homogeneous_case = False
type=None # type of the test case does not matter because no logging is done

# calculate shape derivatives for different duct lengths
discrete_shape_derivatives = []
continuous_shape_derivatives = []
eigenvalues = []
tube_length_list = np.linspace(1,2, num=11)
frequ_list = rparams.c_amb/2/tube_length_list # calculate expected frequencies for Dirichlet-Neumann boundary conditions


for tube_length, frequ in zip(tube_length_list, frequ_list):
    Rijke_Tube = test_case.TestCase(name, type, mesh_resolution, tube_length, tube_height, degree,
                                frequ, perturbation, boundary_conditions, homogeneous_case, parent_path)
    # set up and solve test case of 2D Rijke Tube
    Rijke_Tube.create_rijke_tube_mesh()
    Rijke_Tube.assemble_matrices()
    Rijke_Tube.solve_eigenvalue_problem()
    # save eigenvalue
    eigenvalues.append(Rijke_Tube.omega_dir)
    # calculate the continuous shape derivative
    Rijke_Tube.calculate_continuous_derivative()
    continuous_shape_derivatives.append(Rijke_Tube.derivative/2/np.pi)
    # calculate the discrete shape derivative
    Rijke_Tube.calculate_discrete_derivative()
    discrete_shape_derivatives.append(Rijke_Tube.derivative/perturbation/2/np.pi)
    # delete object to free memory and restart next run
    del Rijke_Tube


#------------------------SAVE DATA-----------------------#
# Save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig3.txt')
with open(output_file, 'w') as f:
    f.write("duct length, eigenvalues, continuous, discrete\n")
    for duc, eig, con, dis, in zip(tube_length_list, eigenvalues, continuous_shape_derivatives, discrete_shape_derivatives):
        f.write(f"{duc}, {eig}, {con}, {dis} \n")


#-----------------------------FINALIZE--------------------------#
# close the gmsh session which was required to run for calculating shape derivatives
gmsh.finalize()
# mark the processing time
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
