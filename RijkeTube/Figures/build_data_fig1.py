'''
compute data of figure1: domain of linearity of discrete shape derivative
'''

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
type = 'discrete'
mesh_resolution = 0.008 # specify mesh resolution
tube_length = 1 # length of the tube
tube_height = 0.047 # height of the tube
degree = 2 # degree of FEM polynomials
frequ = 200 # target frequency - where to expect first mode in Hz
perturbation = 0.001 # perturbation distance of the mesh
# set boundary conditions
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Neumann'}, # outlet
                        3:  {'Neumann'}, # upper wall
                        4:  {'Neumann'}} # lower wall
# set True for homogeneous case, False for inhomogeneous case
homogeneous_case = False

#------------------------TEST CASE-----------------------#
Rijke_Tube = test_case.TestCase(name, type, mesh_resolution, tube_length, tube_height, degree,
                                frequ, perturbation, boundary_conditions, homogeneous_case, parent_path)
# set up and solve test case of 2D Rijke Tube
Rijke_Tube.create_rijke_tube_mesh()
Rijke_Tube.assemble_matrices()
Rijke_Tube.solve_eigenvalue_problem()

# calculate shape derivatives for different perturbations
discrete_shape_derivatives = []
perturbations = np.linspace(0.001,0.3, num=20)

for perturbation in perturbations:
    # set new perturbation distance
    Rijke_Tube.perturbation = perturbation
    # calculate the shape derivative for this perturbation
    Rijke_Tube.calculate_discrete_derivative()
    # save the calculated shape derivative
    discrete_shape_derivatives.append(Rijke_Tube.derivative/perturbation/2/np.pi)
    # print log information
    Rijke_Tube.log()


#------------------------SAVE DATA-----------------------#
# Extract only the real part of the discrete shape derivatives
real_discrete_shape_derivatives = [derivative.real for derivative in discrete_shape_derivatives]
imag_discrete_shape_derivatives = [derivative.imag for derivative in discrete_shape_derivatives]
# Save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig1.txt')
with open(output_file, 'w') as f:
    f.write("Perturbation, Real Part, Imaginary Part\n")
    for p, real, imag in zip(perturbations, real_discrete_shape_derivatives, imag_discrete_shape_derivatives):
        f.write(f"{p}, {real}, {imag}\n")


#-----------------------------FINALIZE--------------------------#
# close the gmsh session which was required to run for calculating shape derivatives
gmsh.finalize()
# mark the processing time
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
