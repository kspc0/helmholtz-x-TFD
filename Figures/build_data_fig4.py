'''
Compute Data of Figure 4: Domain of Linearity of Discrete Shape Derivative for Acoustic Duct
'''

import os
import test_case
import numpy as np
import gmsh

# set path
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

# create test case object
Rijke_Tube = test_case.TestCase("/RijkeTube", 'discrete', True, parent_path + "/RijkeTube")

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
    Rijke_Tube.perturb_rijke_tube_mesh()
    Rijke_Tube.calculate_discrete_derivative()
    # save the calculated shape derivative
    discrete_shape_derivatives.append(Rijke_Tube.derivative/2/np.pi)
    # print log information
    Rijke_Tube.log()
gmsh.finalize() # close the gmsh session

# save derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig4.txt')
with open(output_file, 'w') as f:
    f.write("Perturbation [m], Discrete [Hz/m] \n")
    for p, deriv in zip(perturbations, discrete_shape_derivatives):
        f.write(f"{p}, {deriv.real}\n")
