'''
Compute Data of Figure 6: Residual Analysis with Mesh Refinement
Note: this figure is plotted in the paraview state file
'''

import os
import test_case
import numpy as np
import gmsh
import logging
import shutil

# set path
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

# compute three times with increasing mesh resolution
for i, name in zip([1, 2, 3], ["coarse", "medium", "fine"]):
    print("- running test case with mesh resolution: ", name)
    # set up and solve test case of 2D Rijke Tube
    Rijke_Tube = test_case.TestCase("/RijkeTube", 'discrete', False, parent_path +"/RijkeTube")
    #Rijke_Tube.mesh_resolution = i
    Rijke_Tube.mesh_refinement_factor = i
    Rijke_Tube.create_rijke_tube_mesh()
    Rijke_Tube.assemble_matrices()
    Rijke_Tube.solve_eigenvalue_problem()
    Rijke_Tube.compute_residual()
    # Copy the residual.xdmf file to the /Figures folder with a new name
    source_file = os.path.join(parent_path + "/RijkeTube", "Results", "residual.xdmf")
    destination_file = os.path.join(path, "data_fig6_" + name + ".xdmf")
    shutil.copy(source_file, destination_file)
    # Modify the content in the .xdmf file so it fits the .h5 file names
    with open(destination_file, 'r') as file:
        xdmf_content = file.read()
    xdmf_content = xdmf_content.replace("residual.h5", f"data_fig6_{name}.h5")
    with open(destination_file, 'w') as file:
        file.write(xdmf_content)
    # Copy the residual.h5 file to the /Figures folder with a new name
    source_h5_file = os.path.join(parent_path + "/RijkeTube", "Results", "residual.h5")
    destination_h5_file = os.path.join(path, "data_fig6_" + name + ".h5")
    shutil.copy(source_h5_file, destination_h5_file)
    del Rijke_Tube

gmsh.finalize() # close the gmsh session