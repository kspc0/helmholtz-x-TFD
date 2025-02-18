import os
import test_case
import gmsh

# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))

Kornilov = test_case.TestCase("/KornilovCase" ,'continuous', False ,path)
# compute test case
Kornilov.create_kornilov_mesh()
Kornilov.assemble_matrices()
Kornilov.solve_eigenvalue_problem()
#Kornilov.write_input_functions()
Kornilov.calculate_continuous_derivative()
Kornilov.log()
gmsh.finalize() # close the gmsh session
