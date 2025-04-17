'''
compute data of figure6: domain of linearity of discrete shape derivative for Kornilov Case
'''

import os
import test_case
import numpy as np
import gmsh

# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

# create test case object
KornilovCase = test_case.TestCase("/KornilovCase", 'discrete', False, parent_path + "/KornilovCase")

# set up and solve test case of 2D Rijke Tube
KornilovCase.create_kornilov_mesh()
KornilovCase.assemble_matrices()
KornilovCase.solve_eigenvalue_problem()

# calculate shape derivatives for different perturbations
discrete_shape_derivatives = []
# total height of kornilov case l=2.5e-3m - perturbation should be less than 1/3 of l
perturbations = np.linspace(1e-6, 4e-4 , num=30)

for perturbation in perturbations:
    # set new perturbation distance
    KornilovCase.perturbation = perturbation
    # calculate the shape derivative for this perturbation
    KornilovCase.perturb_kornilov_mesh()
    KornilovCase.calculate_discrete_derivative()
    # save the calculated shape derivative
    discrete_shape_derivatives.append(KornilovCase.derivative/2/np.pi)
    # print log information
    KornilovCase.log()
gmsh.finalize() # close the gmsh session

# extract the discrete shape derivatives as complex numbers
real_discrete_shape_derivatives = [derivative.real for derivative in discrete_shape_derivatives]
imag_discrete_shape_derivatives = [derivative.imag for derivative in discrete_shape_derivatives]
# Save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig8.txt')
with open(output_file, 'w') as f:
    f.write("Perturbation, Real Part, Imaginary Part\n")
    for p, real, imag in zip(perturbations, real_discrete_shape_derivatives, imag_discrete_shape_derivatives):
        f.write(f"{p}, {real}, {imag}\n")
