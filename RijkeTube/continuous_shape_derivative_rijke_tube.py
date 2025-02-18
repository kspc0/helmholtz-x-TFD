import os
import test_case
import gmsh

# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))

Rijke_Tube = test_case.TestCase("/RijkeTube", 'continuous', False ,path)
# compute test case
Rijke_Tube.create_rijke_tube_mesh()
Rijke_Tube.assemble_matrices()
Rijke_Tube.solve_eigenvalue_problem()
#Rijke_Tube.write_input_functions() # for testing
Rijke_Tube.calculate_continuous_derivative()
Rijke_Tube.log()
gmsh.finalize() # close the gmsh session
