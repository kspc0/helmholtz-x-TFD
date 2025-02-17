import kparams
import os
import kornilov_case
import datetime
import gmsh
import numpy as np
from mpi4py import MPI

# mark the processing time
start_time = datetime.datetime.now()
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))

#------------------------MAIN PARAMETERS-----------------------#
name = "/Kornilov"
type = 'discrete'
mesh_resolution = 0.3e-3 # specify mesh resolution
plenum_length = 10e-3 # length of the plenum
height = 2.5e-3 # height of the plenum
slit_height = 1*1e-3 # height of the slitdegree = 1 # the higher the degree, the longer the calulation takes but the more precise it is
degree = 2
frequ = 4000 # where to expect first mode in Hz
perturbation = 0.00001 # perturbation distance
homogeneous_case = False # True for homogeneous case, False for inhomogeneous case
# set boundary conditions case
boundary_conditions =  {1:  {'Neumann'}, # inlet {'Robin': kparams.R_in}
                        2:  {'Dirichlet'}, # outlet {'Robin': kparams.R_out}
                        3:  {'Neumann'}, # upper combustion chamber and slit wall
                        4:  {'Neumann'}, # lower symmetry boundary
                        5:  {'Neumann'}} # upper plenum wall


#------------------------TEST CASE-----------------------#
Rijke_Tube = kornilov_case.TestCase(name, type, mesh_resolution, plenum_length, height, slit_height, degree, frequ, perturbation, boundary_conditions, homogeneous_case, path)
# compute test case
Rijke_Tube.create_kornilov_mesh()
Rijke_Tube.assemble_matrices()
Rijke_Tube.solve_eigenvalue_problem()
Rijke_Tube.write_input_functions() # for testing
Rijke_Tube.calculate_discrete_derivative()
Rijke_Tube.log()

#-----------------------------FINALIZE--------------------------#
# close the gmsh session which was required to run for calculating shape derivatives
gmsh.finalize()
# mark the processing time
if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)