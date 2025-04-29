'''
Compute Data of Figure 10: Stability Map of Acoustic Duct
'''

import os
import test_case
import numpy as np
import gmsh
import logging
import shutil
from helmholtz_x.eigensolvers import stability_map

size_of_map = 4 # number of eigenvalues to include in the stability map

# set logger
logger = logging.getLogger()  # Default logger
logger.setLevel(logging.INFO)  # Set the logging level
# set path
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

# set up and solve test case of 2D Rijke Tube
Rijke_Tube = test_case.TestCase("/RijkeTube", 'discrete', True, parent_path +"/RijkeTube")
Rijke_Tube.create_rijke_tube_mesh()
Rijke_Tube.assemble_matrices()
# center stability map around the origin
target = 0
eigenvalue_list = stability_map(Rijke_Tube.matrices, Rijke_Tube.D, target, nev=size_of_map, print_results=False)
eigenvalue_list = np.array(eigenvalue_list)/(2*np.pi) # convert to Hz

# finish computation
del Rijke_Tube
gmsh.finalize() # close the gmsh session

# save eigenvalues to a text file
output_file = os.path.join(path, 'data_fig10.txt')
with open(output_file, 'w') as f:
    f.write("Eigenvalues [Hz] \n")
    for eig in eigenvalue_list:
        f.write(f"{eig}\n")
