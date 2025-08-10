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

discrete_shape_derivatives = []
coarse_discrete_shape_derivatives = []
continuous_shape_derivatives = []
analytic_shape_derivatives = []

tube_length_list = np.linspace(1, 1.1, num=11) # 11 steps from 1m to 1.1m duct length
frequ_list = -RijkeTube.rparams.c_amb/4/tube_length_list # calculate expected frequencies for Neumann-Dirichlet boundary conditions
# calculate analytic shape derivatives (positive because targeting negative frequencies)
analytic_shape_derivatives = RijkeTube.rparams.c_amb/4/(tube_length_list)**2

# set specific parameters for acoustic duct
specific_mesh_resolution = 0.01 # specify mesh resolution
type = None # type of the test case does not matter because no logging is done

print("fine mesh resolution: ", specific_mesh_resolution)
# iterate over different duct lengths with changing target frequencies
for tube_length, frequ in zip(tube_length_list, frequ_list):
    print("- iterating on tube length: ", tube_length)
    Rijke_Tube = test_case.TestCase("/RijkeTube", type, True, parent_path + "/RijkeTube")
    # overwrite standard parameters used in rparams.py
    Rijke_Tube.length = tube_length
    Rijke_Tube.target = frequ
    Rijke_Tube.mesh_resolution = specific_mesh_resolution
    # set up and solve test case of 2D Rijke Tube
    Rijke_Tube.create_rijke_tube_mesh()
    Rijke_Tube.assemble_matrices()
    Rijke_Tube.solve_eigenvalue_problem()
    # calculate the continuous shape derivative
    Rijke_Tube.calculate_continuous_derivative("outlet")
    continuous_shape_derivatives.append(Rijke_Tube.derivative.real/2/np.pi)
    # calculate the discrete shape derivative
    Rijke_Tube.perturb_rijke_tube_mesh()
    Rijke_Tube.calculate_discrete_derivative()
    discrete_shape_derivatives.append(Rijke_Tube.derivative.real/2/np.pi)
    gmsh.finalize() # close the gmsh session
    # delete object to free memory and restart next run
    del Rijke_Tube

# set specific parameters for coarser duct
specific_mesh_resolution = 0.1 # 10 times coarser then fine mesh resolution
specific_perturbation = 0.01 # also 10 times larger perturbation
print("coarse mesh resolution: ", specific_mesh_resolution)
# calculate coarse discrete shape derivative
for tube_length, frequ in zip(tube_length_list, frequ_list):
    print("- iterating on tube length: ", tube_length)
    Rijke_Tube = test_case.TestCase("/RijkeTube", type, True, parent_path + "/RijkeTube")
    # overwrite standard parameters used in rparams.py
    Rijke_Tube.length = tube_length
    Rijke_Tube.frequ = frequ
    Rijke_Tube.mesh_resolution = specific_mesh_resolution
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


# save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig2.txt')
with open(output_file, 'w') as f:
    f.write("Duct Length [m], Frequency [Hz], Analytic [Hz/m], Continuous [Hz/m], Discrete_fine [Hz/m], Discrete_coarse [Hz/m] \n")
    for duc, fre, ana, con, f_dis, c_dis in zip(tube_length_list, frequ_list, analytic_shape_derivatives, continuous_shape_derivatives, discrete_shape_derivatives, coarse_discrete_shape_derivatives):
        f.write(f"{duc}, {fre}, {ana}, {con}, {f_dis}, {c_dis}\n")
