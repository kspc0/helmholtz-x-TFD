import os
import test_case
import gmsh
import logging

# Global logger setup
logger = logging.getLogger()  # Default logger
logger.setLevel(logging.INFO)  # Set the logging level
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))

Kornilov = test_case.TestCase("/KornilovCase" ,'discrete', False, path)
# compute test case
Kornilov.create_kornilov_mesh()
Kornilov.assemble_matrices()
Kornilov.solve_eigenvalue_problem()
# optional for testing
#Kornilov.compute_residual()
#Kornilov.write_input_functions()
Kornilov.perturb_kornilov_mesh("x")
Kornilov.calculate_discrete_derivative() # so far only linear implemented
Kornilov.log()
gmsh.finalize() # close the gmsh session
