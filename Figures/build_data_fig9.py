'''
Compute Data of Figure 9: Mesh Perturbation Technique Showcase for Kornilov Case
'''

import os
import test_case
import gmsh
import logging
import sys
import shutil

# Global logger setup
logger = logging.getLogger()  # Default logger
logger.setLevel(logging.INFO)  # Set the logging level

# set path
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)
sys.path.append(parent_path)

Kornilov = test_case.TestCase("/KornilovCase" ,'discrete', False, parent_path + "/KornilovCase")
# compute test case
Kornilov.create_kornilov_mesh()
Kornilov.assemble_matrices()
Kornilov.solve_eigenvalue_problem()
# optional for testing
#Kornilov.compute_residual()
#Kornilov.write_input_functions()

# overextend the standard perturbation to make it more visible
Kornilov.perturbation = Kornilov.perturbation *10

Kornilov.perturb_kornilov_mesh()

# copy the mesh KornilovCase.xdmf file to the /Figures folder with a new name
source_file = os.path.join(parent_path + "/KornilovCase", "Meshes", "KornilovCase.xdmf")
destination_file = os.path.join(path, "data_fig9_original.xdmf")
shutil.copy(source_file, destination_file)
# Modify the content in the .xdmf file so it fits the .h5 file names
with open(destination_file, 'r') as file:
    xdmf_content = file.read()
xdmf_content = xdmf_content.replace("KornilovCase.h5", f"data_fig9_original.h5")
with open(destination_file, 'w') as file:
    file.write(xdmf_content)
# Copy the residual.h5 file to the /Figures folder with a new name
source_h5_file = os.path.join(parent_path + "/KornilovCase", "Meshes", "KornilovCase.h5")
destination_h5_file = os.path.join(path, "data_fig9_original.h5")
shutil.copy(source_h5_file, destination_h5_file)


# copy the mesh KornilovCase_perturbed.xdmf file to the /Figures folder with a new name
source_file = os.path.join(parent_path + "/KornilovCase", "Meshes", "KornilovCase_perturbed.xdmf")
destination_file = os.path.join(path, "data_fig9_perturbed.xdmf")
shutil.copy(source_file, destination_file)
# Modify the content in the .xdmf file so it fits the .h5 file names
with open(destination_file, 'r') as file:
    xdmf_content = file.read()
xdmf_content = xdmf_content.replace("KornilovCase_perturbed.h5", f"data_fig9_perturbed.h5")
with open(destination_file, 'w') as file:
    file.write(xdmf_content)
# Copy the residual.h5 file to the /Figures folder with a new name
source_h5_file = os.path.join(parent_path + "/KornilovCase", "Meshes", "KornilovCase_perturbed.h5")
destination_h5_file = os.path.join(path, "data_fig9_perturbed.h5")
shutil.copy(source_h5_file, destination_h5_file)

gmsh.finalize() # close the gmsh session
