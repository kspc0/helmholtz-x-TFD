import os
import test_case
import gmsh

# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))

Rijke_Tube = test_case.TestCase("/RijkeTube", 'discrete', False, path)
# compute test case
Rijke_Tube.create_rijke_tube_mesh()
Rijke_Tube.assemble_matrices()
Rijke_Tube.solve_eigenvalue_problem()
#Kornilov.write_input_functions()
Rijke_Tube.perturb_rijke_tube_mesh()
Rijke_Tube.calculate_discrete_derivative()
Rijke_Tube.log()
gmsh.finalize() # close the gmsh session
