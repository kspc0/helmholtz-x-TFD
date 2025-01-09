from .petsc4py_utils import conjugate_function
from .eigenvectors import normalize_adjoint
from .dolfinx_utils import unroll_dofmap
from dolfinx.fem import form, locate_dofs_topological, VectorFunctionSpace, Function, FunctionSpace
from dolfinx.fem.assemble import assemble_scalar
from ufl import  FacetNormal, grad, inner, Measure, div
from helmholtz_x.io_utils import XDMFReader,xdmf_writer
from math import comb
from dolfinx.io import XDMFFile
import numpy as np
import gmsh


### CALCULATE THE SHAPE DERIVATIVES for every control point
def shapeDerivativesFFD(geometry, lattice, physical_facet_tag, omega_dir, p_dir, p_adj, c, acousticMatrices, FlameMatrix):
    # find the outward normal vector on the geometry surface
    normal = FacetNormal(geometry.mesh)
    # define a measure used for integrating over the mesh's boundary
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)
    # create the normalized adjoint conjugate solution
    p_adj_norm = normalize_adjoint(omega_dir, p_dir, p_adj, acousticMatrices, FlameMatrix)
    p_adj_conj = conjugate_function(p_adj_norm)

    # calcualte the shape gradient G (scalar) for the geometry
    G_neu = div(p_adj_conj * c**2 * grad(p_dir))

    derivatives = {}

    i = lattice.l-1
    # axial FFD control points
    for zeta in range(0,lattice.n):
        derivatives[zeta] = {}
        # azimuthal FFD control points:
        for phi in range(0,lattice.m):
            # calculate the local diplacement field V at the control point
            V_ffd = ffd_displacement_vector(geometry, lattice, physical_facet_tag, i, phi, zeta, deg=1)
            # calculate the shape derivative of the control point
            shape_derivative_form = form(inner(V_ffd, normal) * G_neu * ds(physical_facet_tag))
            eig = assemble_scalar(shape_derivative_form)
            # store the solution in the array
            derivatives[zeta][phi] = eig
    return derivatives


'''
computes the displacement vector field of a surface mesh based on Free-Form Deformation (FFD) lattice
then maps this deformation to the degrees of freedom (DOFs) of a finite element function in the function space Q
-> calculate the displacement vector field V for C=V*n
# Computes the displacement of surface nodes based on FFD lattice
# i,j(=zeta),k(=phi) are indices of the control points
'''
def ffd_displacement_vector(geometry, FFDLattice, surface_physical_tag, i, j, k,
                            includeBoundary=True, returnParametricCoord=True, tol=1e-6, deg=1):
    # extract the mesh data
    mesh, _, facet_tags = geometry.getAll()
    # create the functionspace with continous galerkian (CG)
    # finite element function
    Q = VectorFunctionSpace(mesh, ("CG", deg))

    # define some parameters... idk what this is used for then?
    facets = facet_tags.find(surface_physical_tag)
    indices = locate_dofs_topological(Q, mesh.topology.dim-1 , facets)
    surface_coordinates = mesh.geometry.x[indices]
    surface_elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(2,surface_physical_tag)
    node_tags, coords, t_coords = gmsh.model.mesh.getNodes(2, int(surface_elementary_tag), includeBoundary=includeBoundary, returnParametricCoord=returnParametricCoord)
    #print("before norm")
    norm = gmsh.model.getNormal(int(surface_elementary_tag),t_coords)
    #print("norm")
    # create function space V depending on Q
    V_func = Function(Q)

    dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)
    dofs_Q = dofs_Q.reshape(-1,3)

    # compute basis functions that describe how lattice deforms
    # uses bernstein polynomials with powers s,t,u
    s,t,u = FFDLattice.calcSTU(coords)
    print("s", s)
    print("t", t)
    print("u", u)
    value = comb(FFDLattice.l-1,i)*np.power(1-s, FFDLattice.l-1-i)*np.power(s,i) * \
            comb(FFDLattice.m-1,j)*np.power(1-t, FFDLattice.m-1-j)*np.power(t,j) * \
            comb(FFDLattice.n-1,k)*np.power(1-u, FFDLattice.n-1-k)*np.power(u,k)
    print("value", value)
    coords = coords.reshape(-1, 3) 
    norm = norm.reshape(-1,3)

    # DOF: degree of freedom
    # signs the displacement values to the degrees of freedom (DOFs) of the finite element function
    for dofs_node, node in zip(dofs_Q, surface_coordinates):
        itemindex = np.where(np.isclose(coords, node, atol=tol).all(axis=1))[0]
        if len(itemindex) == 1: 
            V_func.x.array[dofs_node] = value[itemindex]*norm[itemindex][0]
        elif len(itemindex) == 2 :
            V_func.x.array[dofs_node] = value[itemindex][0]*norm[itemindex][0]
        else:
            print(value[itemindex])
   
    V_func.x.scatter_forward()     

    return V_func










### CALCULATE THE SHAPE DERIVATIVES for every control point
def shapeDerivativesFFDRect(geometry, lattice, physical_facet_tag, omega_dir, p_dir, p_adj, c, acousticMatrices, FlameMatrix):
    # find the outward normal vector on the geometry outlines
    normal = FacetNormal(geometry.mesh)
    # define a measure used for integrating over the mesh's boundary
    print("geometry mesh subdomain data: ", geometry.facet_tags)
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)
    #print("geometry facet tags: ", geometry.facet_tags)
    # create the normalized adjoint conjugate solution
    p_adj_norm = normalize_adjoint(omega_dir, p_dir, p_adj, acousticMatrices, FlameMatrix)
    p_adj_conj = conjugate_function(p_adj_norm)

    # calcualte the shape gradient G (scalar) for the geometry
    G_neu = div(p_adj_conj * c**2 * grad(p_dir))

    derivatives = {}
    print("- iterating shape derivatives of points")
    # iterate for every control point in the lattice:
    j = lattice.m-1 # set constant in y direction
    for i in range(0,lattice.l): # counter in x direction
        # calculate the local diplacement field V at the control point
        V_ffd = ffd_displacement_vector_rect(geometry, lattice, physical_facet_tag, i, j, 0, deg=1)
        # calculate the shape derivative of the control point
        shape_derivative_form = form(inner(V_ffd, normal) * G_neu * ds(physical_facet_tag))
        eig = assemble_scalar(shape_derivative_form)
        # store the solution in the array
        derivatives[i] = eig
        print("- shape derivative of point", i, "calculated:", derivatives[i])
    return derivatives


'''
computes the displacement vector field of a surface mesh based on Free-Form Deformation (FFD) lattice
then maps this deformation to the degrees of freedom (DOFs) of a finite element function in the function space Q
-> calculate the displacement vector field V for C=V*n
# Computes the displacement of surface nodes based on FFD lattice
# i,j(=zeta),k(=phi) are indices of the control points
'''
def ffd_displacement_vector_rect(geometry, FFDLattice, surface_physical_tag, i, j, k,
                            includeBoundary=True, returnParametricCoord=True, tol=1e-6, deg=1):
    # extract the mesh data
    mesh, _, facet_tags = geometry.getAll()
    # create the functionspace with continous galerkian (CG)
    Q = VectorFunctionSpace(mesh, ("CG", deg))
    # find tags of surfaces and retrieves coordinates of these surface nodes
    facets = facet_tags.find(surface_physical_tag)
    indices = locate_dofs_topological(Q, mesh.topology.dim-1 , facets)
    surface_coordinates = mesh.geometry.x[indices] # coordinates of the surface nodes of the plenum wall
    # retrieve tag of surface boundary
    surface_elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(1, surface_physical_tag)
    # get the coordinates of the surface nodes on the upper plenum wall
    # t_coord: are the y-coordinates distributed and scaled between 0-1
    node_tags, coords, t_coords = gmsh.model.mesh.getNodes(1, int(surface_elementary_tag), includeBoundary=includeBoundary, returnParametricCoord=returnParametricCoord)
    norm = np.tile([0, 1], len(coords)) # normal vector is always [0,1,0] for the upper plenum wall
    # create function space V depending on Q
    V_func = Function(Q)

    dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)
    #print("dofs_Q", dofs_Q)
    dofs_Q = dofs_Q.reshape(-1,2)

    # compute basis functions that describe how lattice deforms
    # uses bernstein polynomials with powers s,t
    s,t = FFDLattice.calcSTU(coords)
    # list of values containing each node as element, calculated by the bernstein polynomials
    value = comb(FFDLattice.l-1,i)*np.power(1-s, FFDLattice.l-1-i)*np.power(s,i) * \
            comb(FFDLattice.m-1,j)*np.power(1-t, FFDLattice.m-1-j)*np.power(t,j)
    print("sum of value =", np.sum(value)) # should be =1??

    coords = coords.reshape(-1, 3)
    norm = norm.reshape(-1,2) # reshape norm vector to 2D array with elements [0,1]

    # DOF: degree of freedom
    # signs the displacement values to the degrees of freedom (DOFs) of the finite element function
    for dofs_node, node in zip(dofs_Q, surface_coordinates):
        itemindex = np.where(np.isclose(coords, node, atol=tol).all(axis=1))[0]
        if len(itemindex) == 1: 
            V_func.x.array[dofs_node] = value[itemindex]*norm[itemindex][0]
        elif len(itemindex) == 2 :
            V_func.x.array[dofs_node] = value[itemindex][0]*norm[itemindex][0]
        else:
            print(value[itemindex])
   
    V_func.x.scatter_forward()     

    return V_func



def ShapeDerivativesFFDRectFullBorder(geometry, physical_facet_tag, norm_vector, omega_dir, p_dir, p_adj, c, acousticMatrices, FlameMatrix):
    # find the outward normal vector on the geometry outlines
    normal = FacetNormal(geometry.mesh)
    # define a measure used for integrating over the mesh's boundary
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)
    print("geometry facet tags: ", geometry.facet_tags)
    # create the normalized adjoint conjugate solution
    p_adj_norm = normalize_adjoint(omega_dir, p_dir, p_adj, acousticMatrices, FlameMatrix)
    p_adj_conj = conjugate_function(p_adj_norm)

    # calcualte the shape gradient G (scalar) for the geometry
    G_neu = div(p_adj_conj * c**2 * grad(p_dir))

    derivatives = {}
    print("- shape derivatives of border")
    # calculate the local diplacement field V at the border
    V_ffd = ffd_displacement_vector_rect_full_border(geometry, physical_facet_tag, norm_vector, deg=1)
    # calculate the shape derivative of the control point
    print("normal :",normal.ufl_shape)
    print("integral of ds(physical_facet_tag) :",assemble_scalar(form(1*ds(physical_facet_tag))))
    print("C = V*n", inner(V_ffd, normal))
    shape_derivative_form = form(inner(V_ffd, normal) * G_neu * ds(physical_facet_tag)) #
    eig = assemble_scalar(shape_derivative_form)
    # store the solution in the array
    derivatives[1] = eig
    print("- shape derivative of border calculated:", derivatives[1])
    return derivatives



def ffd_displacement_vector_rect_full_border(geometry, surface_physical_tag, norm_vector,
                            includeBoundary=True, returnParametricCoord=True, tol=1e-6, deg=1):
    # extract the mesh data
    mesh, _, facet_tags = geometry.getAll()
    # create the functionspace with continous galerkian (CG)
    Q = VectorFunctionSpace(mesh, ("CG", deg))
    # find tags of surfaces and retrieves coordinates of these surface nodes
    facets = facet_tags.find(surface_physical_tag)
    #print("facets", facets)
    indices = locate_dofs_topological(Q, mesh.topology.dim-1 , facets)
    surface_coordinates = mesh.geometry.x[indices] # coordinates of the surface nodes of the plenum wall
    #print("surface coordinates", surface_coordinates)
    # retrieve tag of surface boundary
    surface_elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(1, surface_physical_tag)
    # get the coordinates of the surface nodes on the upper plenum wall
    # t_coord: are the y-coordinates distributed and scaled between 0-1
    node_tags, coords, t_coords = gmsh.model.mesh.getNodes(1, int(surface_elementary_tag), includeBoundary=includeBoundary, returnParametricCoord=returnParametricCoord)
    #print("coords", coords)
    # normal vector is always [0,1] for the upper plenum wall
    # and [1,0] for the left inlet wall
    norm = np.tile(norm_vector, len(coords))
    # create function space V depending on Q
    V_func = Function(Q)

    dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)
    #print("dofs_Q", dofs_Q)
    dofs_Q = dofs_Q.reshape(-1,2)

    number_of_mesh_points_on_border = round(len(coords)/3) # for 3 dimensions
     # list of values containing each node as element, calculated by the bernstein polynomials
    # value = comb(FFDLattice.l-1,i)*np.power(1-s, FFDLattice.l-1-i)*np.power(s,i) * \
    #         comb(FFDLattice.m-1,j)*np.power(1-t, FFDLattice.m-1-j)*np.power(t,j)
    # equally distribute displacement vector on all mesh points to total sum up to 1
    value = np.zeros(number_of_mesh_points_on_border) # number of mesh points on the border
    for element in range(0,number_of_mesh_points_on_border):
        value[element] = 1/number_of_mesh_points_on_border # value in total should sum up to 1
    #print("value", value)
    print("- integral of displacment field mesh vectors =", np.sum(value)) # should be =1 ?ß

    coords = coords.reshape(-1, 3)
    norm = norm.reshape(-1,2) # reshape norm vector to 2D array with elements [0,1]

    # DOF: degree of freedom
    # signs the displacement values to the degrees of freedom (DOFs) of the finite element function
    for dofs_node, node in zip(dofs_Q, surface_coordinates):
        itemindex = np.where(np.isclose(coords, node, atol=tol).all(axis=1))[0]
        if len(itemindex) == 1: 
            V_func.x.array[dofs_node] = value[itemindex]*norm[itemindex][0]
        elif len(itemindex) == 2 :
            V_func.x.array[dofs_node] = value[itemindex][0]*norm[itemindex][0]
        else:
            print(value[itemindex])
   
    V_func.x.scatter_forward()     

    return V_func