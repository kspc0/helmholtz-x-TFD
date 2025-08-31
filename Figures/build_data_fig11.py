'''
Compute Data of Figure 8: Domain of Linearity of Discrete Shape Derivative for instable Kornilov Case
'''

import os
import test_case
import numpy as np
import gmsh

# set path
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

# create test case object
KornilovCase = test_case.TestCase("/KornilovCase", 'discrete', True, parent_path + "/KornilovCase")

# set up and solve test case of 2D Rijke Tube
KornilovCase.create_kornilov_mesh()
KornilovCase.assemble_matrices()
KornilovCase.solve_eigenvalue_problem()

# calculate shape derivatives for different perturbations
discrete_shape_derivatives = []
# total height of kornilov case l=2.5e-3m - perturbation should be less than 1/3 of l
perturbations = np.linspace(1e-6*0.22, 0.1*0.22, num=30)

for perturbation in perturbations:
    print("- iterating on perturbation: ", perturbation/0.22)
    # set new perturbation distance
    KornilovCase.perturbation = perturbation
    # calculate the shape derivative for this perturbation
    KornilovCase.perturb_kornilov_mesh("x")
    KornilovCase.calculate_discrete_derivative()
    # save the calculated shape derivative
    print("- discrete shape derivative: ", KornilovCase.derivative)
    discrete_shape_derivatives.append(KornilovCase.derivative/-2/np.pi*KornilovCase.perturbation)
gmsh.finalize() # close the gmsh session

# Save the real and imaginary derivatives along with the perturbations to a text file
output_file = os.path.join(path, 'data_fig11.txt')
with open(output_file, 'w') as f:
    f.write("Perturbation [%], delta omega [Hz/m]\n")
    for p, deriv in zip(perturbations, discrete_shape_derivatives):
        f.write(f"{p/0.22}, {deriv.real/0.22}\n")
