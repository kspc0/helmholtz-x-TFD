from .petsc4py_utils import conjugate_function
from .eigenvectors import normalize_adjoint
from .dolfinx_utils import unroll_dofmap
from dolfinx.fem import form, locate_dofs_topological, VectorFunctionSpace, Function, FunctionSpace
from dolfinx.fem.assemble import assemble_scalar
from ufl import  FacetNormal, grad, inner, Measure, div, diff, dot
from helmholtz_x.io_utils import XDMFReader,xdmf_writer
from math import comb
from dolfinx.io import XDMFFile
import numpy as np
import gmsh
import logging

# calculate the shape derivatives of a straight boundary that is moved in one direction
def ShapeDerivativeFullBorder(geometry, physical_facet_tag, selected_boundary_condition, norm_vector, omega_dir, p_dir, p_adj, c, acousticMatrices, FlameMatrix):
    # find the outward normal vector on the geometry outlines
    normal = FacetNormal(geometry.mesh)
    # define a measure used for integrating over the mesh boundary
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)
    logging.debug("- ds = %s", assemble_scalar(form(1*ds(physical_facet_tag))))
    # normalize adjoint eigenvector
    p_adj_norm = normalize_adjoint(omega_dir, p_dir, p_adj, acousticMatrices, FlameMatrix)
    # conjugate the adjoint after normalization
    p_adj_conj = conjugate_function(p_adj_norm)

    # calcualte the shape gradient G (scalar) for the geometry
    # use different formulas depending on which boundary is regarded
    if selected_boundary_condition == {'Neumann'}:
        logging.debug("- Neumann Shape Gradient")
        G_neu = div(p_adj_conj * c**2 * grad(p_dir))
        #G_neu = p_adj_conj *c**2 * div(grad(p_dir)) # Neumann alternative form
    elif selected_boundary_condition == {'Dirichlet'}:
        logging.debug("- Dirichlet Shape Gradient")
        G_neu = - c**2 * dot(grad(p_adj_conj), normal) * dot(grad(p_dir), normal)
    else:
        logging.error("Error - shape gradient needs definition of according boundary")

    logging.debug("- assembling shape derivatives of border")
    # calculate the local diplacement field V at the border
    V_ffd = ffd_displacement_vector_full_border(geometry, physical_facet_tag, norm_vector, deg=1)
    # integrate inner(V_ffd, normal) over the domain for logging
    V_ffd_normal_form = form(inner(V_ffd, normal) * ds(physical_facet_tag))
    V_ffd_normal_value = assemble_scalar(V_ffd_normal_form)
    logging.debug("- Integral of inner(V_ffd, normal) over the domain: %s", V_ffd_normal_value)

    # calculate the shape derivative of the border
    shape_derivative_form = form(inner(V_ffd, normal) *G_neu * ds(physical_facet_tag))
    deriv = assemble_scalar(shape_derivative_form)
    logging.debug("- shape derivative of border calculated: %s", deriv)
    return deriv


# compute displacement field for the full border of the geometry
def ffd_displacement_vector_full_border(geometry, surface_physical_tag, norm_vector,
                            includeBoundary=True, returnParametricCoord=True, tol=1e-6, deg=1):
    # extract the mesh data
    mesh, _, facet_tags = geometry.getAll()
    # create the functionspace
    Q = VectorFunctionSpace(mesh, ("CG", deg))
    # find tags of requested surface and retrieves coordinates of these surface nodes
    facets = facet_tags.find(surface_physical_tag)
    indices = locate_dofs_topological(Q, mesh.topology.dim-1 , facets)
    surface_coordinates = mesh.geometry.x[indices] # coordinates of the surface nodes of the wall
    # retrieve tag of surface boundary
    #surface_elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(1, surface_physical_tag)
    #logging.debug("- surface elementary tag %s", surface_elementary_tag)
    # get the coordinates of the surface nodes on the surface with the requested tag
    #node_tags, coords, t_coords = gmsh.model.mesh.getNodes(1, int(surface_elementary_tag), includeBoundary=includeBoundary, returnParametricCoord=returnParametricCoord)
    # set normal vector to be normal to chosen surface as defined earlier 
    coords = surface_coordinates[::-1].flatten()

    norm = np.tile(norm_vector, len(coords))


    # create function V depending on function space Q
    V_func = Function(Q)
    logging.debug("Indices: %s", indices)
    logging.debug("Block size (Q.dofmap.bs): %s", Q.dofmap.bs)
    # find the degrees of freedom and unrolls them into new array
    dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)
    logging.debug("Unrolled DOFs before reshaping: %s", dofs_Q)
    dofs_Q = dofs_Q.reshape(-1,2)
    logging.debug("Unrolled DOFs after reshaping: %s", dofs_Q)
    logging.debug("Surface coordinates shape: %s", surface_coordinates.shape)


    number_of_mesh_points_on_border = round(len(coords)/3) # for 3 dimensions
    logging.debug("- number of mesh points on border: %d", number_of_mesh_points_on_border)
    # set length of the displacement vector to 1 for each mesh point
    value = np.ones(number_of_mesh_points_on_border)

    # reshape norm vector to 2D array with elements [0,1]
    coords = coords.reshape(-1, 3)
    norm = norm.reshape(-1,2)

    # signs the displacement values to the degrees of freedom (DOF) of the finite element function
    for dofs_node, node in zip(dofs_Q, surface_coordinates):
        itemindex = np.where(np.isclose(coords, node, atol=tol).all(axis=1))[0]
        logging.debug("- Itemindex %s: for node %s", node, itemindex)
        if len(itemindex) == 1: 
            V_func.x.array[dofs_node] = value[itemindex]*norm[itemindex][0]
        elif len(itemindex) == 2 :
            V_func.x.array[dofs_node] = value[itemindex][0]*norm[itemindex][0]
        else:
            print(value[itemindex])
    
    V_func.x.scatter_forward()     

    return V_func