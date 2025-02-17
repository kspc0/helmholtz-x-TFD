import rparams
import os
import test_case
import datetime
import gmsh
from mpi4py import MPI

# mark the processing time
start_time = datetime.datetime.now()
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))

#------------------------MAIN PARAMETERS-----------------------#
name = "/RijkeTube"
type = 'continuous'
mesh_resolution = 0.008 # specify mesh resolution
tube_length = 1 # length of the tube
tube_height = 0.047 # height of the tube
degree = 2 # degree of FEM polynomials
frequ = 200 # target frequency - where to expect first mode in Hz
perturbation = None # perturbation distance of the mesh
# set boundary conditions
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Neumann'}, # outlet
                        3:  {'Neumann'}, # upper wall
                        4:  {'Neumann'}} # lower wall
# set True for homogeneous case, False for inhomogeneous case
homogeneous_case = False

#---------------------------TEST CASE---------------------------#
Rijke_Tube = test_case.TestCase(name, type, mesh_resolution, tube_length, tube_height, degree, frequ, perturbation, boundary_conditions, homogeneous_case, path)
# compute test case
Rijke_Tube.create_rijke_tube_mesh()
Rijke_Tube.assemble_matrices()
Rijke_Tube.solve_eigenvalue_problem()
#Rijke_Tube.write_input_functions() # for testing
Rijke_Tube.calculate_continuous_derivative()
Rijke_Tube.log()

#-----------------------------FINALIZE--------------------------#
# close the gmsh session which was required to run for calculating shape derivatives
gmsh.finalize()
# mark the processing time
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)