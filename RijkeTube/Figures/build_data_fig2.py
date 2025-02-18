'''
compute data of figure2: comparison of discrete, analytic and continuous shape derivative for homogeneous Rijke tube
'''

import os
import test_case
import gmsh
import numpy as np
import sys

# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)
# add the parent directory to the python path
sys.path.append(parent_path)
import rparams # then import the rparams module

# calculate shape derivatives for different duct lengths
discrete_shape_derivatives = []
coarse_discrete_shape_derivatives = []
continuous_shape_derivatives = []
analytic_shape_derivatives = []
tube_length_list = np.linspace(1, 1.1, num=11)
frequ_list = rparams.c_amb/4/tube_length_list # calculate expected frequencies for Neumann-Dirichlet boundary conditions

# set specific parameters for duct
specific_mesh_resolution = 0.01 # specify mesh resolution
# set specific boundary conditions
specific_boundary_conditions =  {1:  {'Neumann'}, # inlet
                                 2:  {'Dirichlet'}, # outlet
                                 3:  {'Neumann'}, # upper wall
                                 4:  {'Neumann'}} # lower wall
# set True for homogeneous case, False for inhomogeneous case
specific_homogeneous_case = True
type = None # type of the test case does not matter because no logging is done

# iterate over different duct lengths with different target frequencies
for tube_length, frequ in zip(tube_length_list, frequ_list):
    Rijke_Tube = test_case.TestCase("/RijkeTube", type, specific_homogeneous_case, parent_path)
    # set different parameters than the standard used in rparams.py
    Rijke_Tube.length = tube_length
    Rijke_Tube.frequ = frequ
    Rijke_Tube.mesh_resolution = specific_mesh_resolution
    Rijke_Tube.boundary_conditions = specific_boundary_conditions
    # set up and solve test case of 2D Rijke Tube
    Rijke_Tube.create_rijke_tube_mesh()
    Rijke_Tube.assemble_matrices()
    Rijke_Tube.solve_eigenvalue_problem()
    # calculate the continuous shape derivative
    Rijke_Tube.calculate_continuous_derivative()
    continuous_shape_derivatives.append(Rijke_Tube.derivative.real/2/np.pi)
    # calculate the discrete shape derivative
    Rijke_Tube.perturb_rijke_tube_mesh()
    Rijke_Tube.calculate_discrete_derivative()
    discrete_shape_derivatives.append(Rijke_Tube.derivative.real/2/np.pi)
    gmsh.finalize() # close the gmsh session
    # delete object to free memory and restart next run
    del Rijke_Tube

# set specific parameters for coarser duct
specific_mesh_resolution = 0.1
specific_perturbation = 0.01
# calculate coarse discrete shape derivative
for tube_length, frequ in zip(tube_length_list, frequ_list):
    Rijke_Tube = test_case.TestCase("/RijkeTube", type, specific_homogeneous_case, parent_path)
    # set different parameters than the standard used in rparams.py
    Rijke_Tube.length = tube_length
    Rijke_Tube.frequ = frequ
    Rijke_Tube.mesh_resolution = specific_mesh_resolution
    Rijke_Tube.boundary_conditions = specific_boundary_conditions
    Rijke_Tube.perturbation = specific_perturbation
    # set up and solve test case of 2D Rijke Tube
    Rijke_Tube.create_rijke_tube_mesh()
    Rijke_Tube.assemble_matrices()
    Rijke_Tube.solve_eigenvalue_problem()
    # calculate the discrete shape derivative
    Rijke_Tube.perturb_rijke_tube_mesh()
    Rijke_Tube.calculate_discrete_derivative()
    coarse_discrete_shape_derivatives.append(Rijke_Tube.derivative.real/2/np.pi)
    gmsh.finalize() # close the gmsh session
    # delete object to free memory and restart next run
    del Rijke_Tube

# calculate analytic shape derivatives
analytic_shape_derivatives = -rparams.c_amb/4/(tube_length_list)**2
# save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig2.txt')
with open(output_file, 'w') as f:
    f.write("duct length, frequency, analytic, continuous, discrete_fine, discrete_coarse\n")
    for duc, fre, ana, con, f_dis, c_dis in zip(tube_length_list, frequ_list, analytic_shape_derivatives, continuous_shape_derivatives, discrete_shape_derivatives, coarse_discrete_shape_derivatives):
        f.write(f"{duc}, {fre}, {ana}, {con}, {f_dis}, {c_dis}\n")
