'''
Compute Data of Figure 7: Shape Optimization with Discrete Shape Derivative for Slit Duct
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

plenum_height = np.linspace(2.5e-3, 5e-3, num=4) # 4 steps from 2.5mm to 3mm plenum height
frequ = -kparams.c_amb/4/kparams.length # calculate expected frequencies for Neumann-Dirichlet boundary conditions

for height in plenum_height:
    print("- iterating for height: ", height)
    KornilovCase = test_case.TestCase("/KornilovCase", type, True, parent_path + "/KornilovCase")
    # overwrite standard parameters used in kparams.py
    KornilovCase.height = height
    KornilovCase.frequ = frequ
    # set up and solve test case of 2D Rijke Tube
    KornilovCase.create_kornilov_mesh()
    KornilovCase.assemble_matrices()
    KornilovCase.solve_eigenvalue_problem()
    # save eigenvalue
    eigenvalues.append(KornilovCase.omega_dir/2/np.pi)
    # calculate the discrete shape derivative
    KornilovCase.perturb_kornilov_mesh("y")
    KornilovCase.calculate_discrete_derivative()
    discrete_shape_derivatives.append(KornilovCase.derivative/2/np.pi)
    gmsh.finalize() # close the gmsh session
    # delete object to free memory and restart next run
    del KornilovCase

# extract the discrete shape derivatives as complex numbers
real_discrete_shape_derivatives = [derivative.real for derivative in discrete_shape_derivatives]
real_eigenvalues = [eig.real for eig in eigenvalues]
# save the real part of eigenvalue and shape derivative along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig10.txt')
with open(output_file, 'w') as f:
    f.write("Plenum Height [m], Eigenvalues [Hz], Discrete [Hz] \n")
    for plen, eig, dis, in zip(plenum_height, real_eigenvalues, real_discrete_shape_derivatives):
        f.write(f"{plen}, {eig}, {dis} \n")
