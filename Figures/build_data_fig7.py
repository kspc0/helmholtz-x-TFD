'''
Compute Data of Figure 11: Shape Optimization with Discrete and Continuous Shape Derivative for instable Kornilov Case
'''

import os
import test_case
import gmsh
import numpy as np
import sys

# set path
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)
import KornilovCase.kparams as kparams # then import the kparams module
sys.path.append(parent_path)
type=None # type of the test case does not matter because no logging is done

# calculate shape derivatives for different duct lengths
discrete_shape_derivatives = []
continuous_shape_derivatives = []
eigenvalues = []

chamber_length = np.linspace(0.141, 0.15, num=11) # 11 steps from 14,1cm to 15cm chamber length
frequ = -800#-kparams.c_amb/4/kparams.length # calculate expected frequencies for Neumann-Dirichlet boundary conditions

for length in chamber_length:
    print("- running test case with chamber length: ", length)
    KornilovCase = test_case.TestCase("/KornilovCase", type, False, parent_path + "/KornilovCase")
    # overwrite standard parameters used in kparams.py
    KornilovCase.chamber_length = length
    KornilovCase.target = frequ
    # set up and solve test case of 2D Rijke Tube
    KornilovCase.create_kornilov_mesh()
    KornilovCase.assemble_matrices()
    KornilovCase.solve_eigenvalue_problem()
    # save eigenvalue
    eigenvalues.append(KornilovCase.omega_dir/2/np.pi)
    # calculate the continuous shape derivative
    KornilovCase.calculate_continuous_derivative("outlet")
    continuous_shape_derivatives.append(KornilovCase.derivative/2/np.pi)
    # calculate the discrete shape derivative
    KornilovCase.perturb_kornilov_mesh("x")
    KornilovCase.calculate_discrete_derivative()
    discrete_shape_derivatives.append(KornilovCase.derivative/2/np.pi)
    gmsh.finalize() # close the gmsh session
    # delete object to free memory and restart next run
    del KornilovCase

# save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig11.txt')
with open(output_file, 'w') as f:
    f.write("Chamber Length [m], Eigenvalues [Hz], Continuous [Hz], Discrete [Hz] \n")
    for plen, eig, con, dis, in zip(chamber_length, eigenvalues, continuous_shape_derivatives, discrete_shape_derivatives):
        f.write(f"{plen}, {eig}, {con}, {dis} \n")
