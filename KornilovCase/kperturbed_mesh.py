'''
objective: create a perturbed plenum in the mesh for the Kornilov case
'''
import os
import sys

import gmsh
import numpy as np

from helmholtz_x.io_utils import XDMFReader, dict_writer, xdmf_writer, write_xdmf_mesh # to write mesh data as files


#-----------------------------SETUP--------------------------------#
# set variables to load and save files
path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/Meshes" # folder of mesh file
mesh_name = "/KornilovMesh" # name of the mesh file
perturbed_mesh_name = "/KornilovPerturbedMesh" # name of the perturbed mesh file
results_dir = "/Results" # folder for saving results
eigenvalues_dir = "/PlotEigenvalues" # folder for saving eigenvalues


#--------------------------CREATE MESH----------------------------#
print("\n--- CREATING MESH ---")
gmsh.initialize() # start the gmsh session
gmsh.model.add("KornilovCase") # add the model name
mesh_resolution = 0.0005 # specify mesh resolution
# locate the points of the 2D geometry: [m]
p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_resolution)  
p2 = gmsh.model.geo.addPoint(0, 2.5e-3, 0, mesh_resolution)
p3 = gmsh.model.geo.addPoint(10e-3, 2.5e-3, 0, mesh_resolution)
p4 = gmsh.model.geo.addPoint(10e-3, 1e-3, 0, mesh_resolution/4) # refine the mesh at this point
p5 = gmsh.model.geo.addPoint(11e-3, 1e-3, 0, mesh_resolution/4)
p6 = gmsh.model.geo.addPoint(11e-3, 2.5e-3, 0, mesh_resolution)
p7 = gmsh.model.geo.addPoint(37e-3, 2.5e-3, 0, mesh_resolution)
p8 = gmsh.model.geo.addPoint(37e-3, 0, 0, mesh_resolution)
# create outlines by connecting points
l1 = gmsh.model.geo.addLine(p1, p2) # inlet boundary
l2 = gmsh.model.geo.addLine(p2, p3) # upper plenum wall
l3 = gmsh.model.geo.addLine(p3, p4) # slit wall
l4 = gmsh.model.geo.addLine(p4, p5) # slit wall
l5 = gmsh.model.geo.addLine(p5, p6) # slit wall
l6 = gmsh.model.geo.addLine(p6, p7) # upper combustion chamber wall
l7 = gmsh.model.geo.addLine(p7, p8) # outlet boundary
l8 = gmsh.model.geo.addLine(p8, p1) # lower symmetry boundary
# create extra points to outline the plenum (needed for shape derivation of upper plenum wall)
# create curve loops for surface
loop1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4,l5,l6,l7,l8]) # entire geometry
# create surfaces from the curved loops
surface1 = gmsh.model.geo.addPlaneSurface([loop1]) # surface of entire geometry
# assign physical tags for 1D boundaries
gmsh.model.addPhysicalGroup(1, [l1], tag=1) # inlet boundary
gmsh.model.addPhysicalGroup(1, [l7], tag=2) # outlet boundary
gmsh.model.addPhysicalGroup(1, [l3,l4,l5,l6], tag=3) # upper combustion chamber and slit wall
gmsh.model.addPhysicalGroup(1, [l8], tag=4) # lower symmetry boundary
gmsh.model.addPhysicalGroup(1, [l2], tag=5) # upper wall of plenum
# assign physical tag for 2D surface
gmsh.model.addPhysicalGroup(2, [surface1], tag=1)
# create 2D mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
# optionally launch GUI to see the results
# if '-nopopup' not in sys.argv:
#    gmsh.fltk.run() 
# save data in /Meshes directory
gmsh.write("{}.msh".format(path+mesh_dir+mesh_name)) # save as .msh file
write_xdmf_mesh(path+mesh_dir+mesh_name,dimension=2) # save as .xdmf file


#-------------------PERTURBING THE MESH---------------------------#
# for discrete shape derivatives, the mesh needs to be perturbed
# read tags and coordinates of the mesh
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
# assign x,y,z coordinates to separate arrays
xcoords = node_coords[0::3] # get x-coordinates
ycoords = node_coords[1::3] # get y-coordinates
zcoords = node_coords[2::3] # get z-coordinates
# create list to store the indices of the plenum nodes
plenum_node_indices = []
# choose points which have smaller x coordinate then 10mm or have x coordinate of 10mm and y coordinate greater than 1mm
# these are all the points in the plenum without the slit entry
for i in range(len(xcoords)):
    if (xcoords[i] < 0.01) or (xcoords[i] == 0.01 and ycoords[i] > 0.001):
        plenum_node_indices.append(i) # store the index of the plenum nodes in this array

# perturb the chosen mesh points slightly in y direction
perturbation = 0.0005 # perturbation distance
# perturbation is percent based on the y-coordinate
ycoords[plenum_node_indices] += ycoords[plenum_node_indices] / 0.0025 * perturbation

# update node y coordinates in mesh from the perturbed points and the unperturbed original points
node_coords[1::3] = ycoords

# update node positions
for tag, new_coords in zip(node_tags, node_coords.reshape(-1,3)):
    gmsh.model.mesh.setNode(tag, new_coords, [])

# update point positions
gmsh.model.setCoordinates(p2, 0, 2.5e-3/ 0.0025 * perturbation + 2.5e-3, 0)
gmsh.model.setCoordinates(p3, 10e-3, 2.5e-3/ 0.0025 * perturbation + 2.5e-3, 0)


# recalculate the acoustic matrices for the perturbed mesh
'''
Todo
'''


# optionally launch GUI to see the results
if '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# save perturbed mesh data in /Meshes directory
gmsh.write("{}.msh".format(path+mesh_dir+perturbed_mesh_name)) # save as .msh file
write_xdmf_mesh(path+mesh_dir+perturbed_mesh_name,dimension=2) # save as .xdmf file


gmsh.finalize() # close the gmsh session