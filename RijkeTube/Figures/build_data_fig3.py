'''
compute data of figure3: shape optimization with discrete and continuous shape derivative for 2D Rijke tube
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
type=None # type of the test case does not matter because no logging is done

# calculate shape derivatives for different duct lengths
discrete_shape_derivatives = []
continuous_shape_derivatives = []
eigenvalues = []
tube_length_list = np.linspace(1,2, num=10)
frequ_list = -rparams.c_amb/4/tube_length_list # calculate a target frequency for Dirichlet-Neumann boundary conditions for homogeneous tube

for tube_length, frequ in zip(tube_length_list, frequ_list):
    Rijke_Tube = test_case.TestCase("/RijkeTube", type, False, parent_path)
    # set different parameters than the standard used in rparams.py
    Rijke_Tube.length = tube_length
    Rijke_Tube.frequ = frequ
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
    Rijke_Tube.perturb_rijke_tube_mesh()
    Rijke_Tube.calculate_discrete_derivative()
    discrete_shape_derivatives.append(Rijke_Tube.derivative/2/np.pi)
    gmsh.finalize() # close the gmsh session
    # delete object to free memory and restart next run
    del Rijke_Tube

# save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig3.txt')
with open(output_file, 'w') as f:
    f.write("duct length, eigenvalues, continuous, discrete\n")
    for duc, eig, con, dis, in zip(tube_length_list, eigenvalues, continuous_shape_derivatives, discrete_shape_derivatives):
        f.write(f"{duc}, {eig}, {con}, {dis} \n")
