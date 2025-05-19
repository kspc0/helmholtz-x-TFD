'''
Compute Data of Figure 2: Comparison of Discrete, Analytic and Continuous Shape Derivative for Acoustic Duct
'''

import os
import test_case
import gmsh
import numpy as np
import sys

# set path
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)
# add the parent directory to the python path
sys.path.append(parent_path)
import RijkeTube.rparams # then import the rparams module

# 3 iterations for fine, medium and coarse mesh
discrete_shape_derivatives = []

continuous_shape_derivatives = []

analytic_shape_derivatives = []

tube_length_list = np.linspace(0.9, 1.1, num=11) # 11 steps from 0.9m to 1.1m duct length
frequ_list = -RijkeTube.rparams.c_amb/4/tube_length_list # calculate expected frequencies for Neumann-Dirichlet boundary conditions
# calculate analytic shape derivatives (positive because targeting negative frequencies)
analytic_shape_derivatives = RijkeTube.rparams.c_amb/4/(tube_length_list)**2

# set specific parameters for acoustic duct
specific_mesh_resolution = np.linspace(0.2,0.01,10) # specify mesh resolution
type = None # type of the test case does not matter because no logging is done

print("mesh resolution: ", specific_mesh_resolution)
for i in range(10):
    print("-----------iteration: ", i)
    # iterate over different duct lengths with changing target frequencies
    for tube_length, frequ in zip(tube_length_list, frequ_list):
        print("- iterating on tube length: ", tube_length)
        Rijke_Tube = test_case.TestCase("/RijkeTube", type, True, parent_path + "/RijkeTube")
        # overwrite standard parameters used in rparams.py
        Rijke_Tube.length = tube_length
        Rijke_Tube.target = frequ
        Rijke_Tube.mesh_resolution = specific_mesh_resolution[i]
        Rijke_Tube.perturbation = specific_mesh_resolution[i] * 0.05 # set perturbation to 5% of mesh resolution
        # set up and solve test case of 2D Rijke Tube
        Rijke_Tube.create_rijke_tube_mesh()
        Rijke_Tube.assemble_matrices()
        Rijke_Tube.solve_eigenvalue_problem()
        # calculate the continuous shape derivative
        Rijke_Tube.calculate_continuous_derivative("outlet")
        continuous_shape_derivatives.append(Rijke_Tube.derivative.real/2/np.pi)
        # calculate the discrete shape derivative
        Rijke_Tube.perturb_rijke_tube_mesh("linear")
        Rijke_Tube.calculate_discrete_derivative()
        discrete_shape_derivatives.append(Rijke_Tube.derivative.real/2/np.pi)
        gmsh.finalize() # close the gmsh session
        # delete object to free memory and restart next run
        del Rijke_Tube


# save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig12.txt')
with open(output_file, 'w') as f:
    f.write("Duct Length [m], Frequency [Hz], Analytic [Hz/m], Continuous [Hz/m], Discrete [Hz/m] \n")
    n = max(len(discrete_shape_derivatives), len(continuous_shape_derivatives))
    # when there is no duct lengths left, start looping to save all data
    for i in range(n):
        duc = tube_length_list[i % len(tube_length_list)]
        fre = frequ_list[i % len(frequ_list)]
        ana = analytic_shape_derivatives[i % len(analytic_shape_derivatives)]
        con = continuous_shape_derivatives[i]
        dis = discrete_shape_derivatives[i]
        f.write(f"{duc}, {fre}, {ana}, {con}, {dis}\n")
