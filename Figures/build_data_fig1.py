'''
compute data of figure1: domain of linearity of discrete shape derivative
'''

import os
import test_case
import numpy as np
import gmsh

# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

# create test case object
Rijke_Tube = test_case.TestCase("/RijkeTube", 'discrete', False, parent_path + "/RijkeTube")

# set up and solve test case of 2D Rijke Tube
Rijke_Tube.create_rijke_tube_mesh()
Rijke_Tube.assemble_matrices()
Rijke_Tube.solve_eigenvalue_problem()

# calculate shape derivatives for different perturbations
discrete_shape_derivatives = []
perturbations = np.linspace(0.001,0.3, num=20) # 20 steps from 0.1% to 30% perturbation

for perturbation in perturbations:
    # set new perturbation distance
    Rijke_Tube.perturbation = perturbation
    # calculate the shape derivative for this perturbation
    Rijke_Tube.perturb_rijke_tube_mesh()
    Rijke_Tube.calculate_discrete_derivative()
    # save the calculated shape derivative
    discrete_shape_derivatives.append(Rijke_Tube.derivative/2/np.pi)
    # print log information
    Rijke_Tube.log()
gmsh.finalize() # close the gmsh session

# extract the discrete shape derivatives as complex numbers
real_discrete_shape_derivatives = [derivative.real for derivative in discrete_shape_derivatives]
imag_discrete_shape_derivatives = [derivative.imag for derivative in discrete_shape_derivatives]
# Save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig1.txt')
with open(output_file, 'w') as f:
    f.write("Perturbation, Real Part, Imaginary Part\n")
    for p, real, imag in zip(perturbations, real_discrete_shape_derivatives, imag_discrete_shape_derivatives):
        f.write(f"{p}, {real}, {imag}\n")
