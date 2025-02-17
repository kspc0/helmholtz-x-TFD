'''
python class for test cases in the helmholtzX project
'''

import datetime
import os
import gmsh
import numpy as np
import sys
from mpi4py import MPI
# python script imports
import kparams
# HelmholtzX utilities
from helmholtz_x.io_utils import XDMFReader, dict_writer, xdmf_writer, write_xdmf_mesh # to write mesh data as files
from helmholtz_x.parameters_utils import sound_speed # to calculate sound speed from temperature
from helmholtz_x.acoustic_matrices import AcousticMatrices # to assemble the acoustic matrices for discrete Helm. EQU
from helmholtz_x.flame_transfer_function import nTau, stateSpace # to define the flame transfer function
from helmholtz_x.flame_matrices import DistributedFlameMatrix # to define the flame matrix for discrete Helm. EQU
from helmholtz_x.eigensolvers import newtonSolver # to solve the system
from helmholtz_x.dolfinx_utils import absolute # to get the absolute value of a function
from helmholtz_x.petsc4py_utils import vector_matrix_vector
from helmholtz_x.shape_derivatives import ShapeDerivativesFFDRectFullBorder, ffd_displacement_vector_rect_full_border # to calculate shape derivatives

# class for creating and computing different test cases
class TestCase:
    # constructor
    def __init__(self, name, type, mesh_resolution, plenum_length, height, slit_height, degree, frequ, perturbation, boundary_conditions, homogeneous_case, path):
        # initialize standard parameters of object
        self.type = type
        self.name = name
        self.mesh_resolution = mesh_resolution
        self.plenum_length = plenum_length
        self.height = height
        self.slit_height = slit_height
        self.degree = degree
        self.perturbation = perturbation
        self.frequ = frequ
        self.homogeneous_case = homogeneous_case
        self.boundary_conditions = boundary_conditions
        self.homogeneous_case = homogeneous_case
        self.path = path
        # initialize parameters for homogeneous or inhomogeneous case
        if self.homogeneous_case: # homogeneous case
            self.T_out = kparams.T_in # no temperature gradient
            self.Rho_out = kparams.rho_u # no density gradient
        else: # inhomogeneous case
            self.T_out = kparams.T_out
            self.Rho_out = kparams.rho_d


    def create_kornilov_mesh(self):
        print("\n--- CREATING MESH ---")
        gmsh.initialize() # start the gmsh session
        gmsh.model.add(self.name) # add the model name
        # locate the points of the 2D geometry: [m]
        p1 = gmsh.model.geo.addPoint(0, 0, 0, self.mesh_resolution)  
        p2 = gmsh.model.geo.addPoint(0, self.height, 0, self.mesh_resolution)
        p3 = gmsh.model.geo.addPoint(self.plenum_length, self.height, 0, self.mesh_resolution)
        p4 = gmsh.model.geo.addPoint(self.plenum_length, self.slit_height, 0, self.mesh_resolution/4) # refine the mesh at this point
        p5 = gmsh.model.geo.addPoint(self.plenum_length+1e-3, self.slit_height, 0, self.mesh_resolution/4)
        p6 = gmsh.model.geo.addPoint(self.plenum_length+1e-3, self.height, 0, self.mesh_resolution)
        p7 = gmsh.model.geo.addPoint(self.plenum_length*3, self.height, 0, self.mesh_resolution)
        p8 = gmsh.model.geo.addPoint(self.plenum_length*3, 0, 0, self.mesh_resolution)
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
        #if '-nopopup' not in sys.argv:
        #   gmsh.fltk.run() 
        # save data in /Meshes directory
        gmsh.write("{}.msh".format(self.path+"/Meshes"+self.name)) # save as .msh file
        write_xdmf_mesh(self.path+"/Meshes"+self.name, dimension=2) # save as .xdmf file
        print("\n--- LOADING MESH ---")
        self.MeshObject = XDMFReader(self.path+"/Meshes"+self.name)
        self.mesh, self.subdomains, self.facet_tags = self.MeshObject.getAll() # mesh, domains and tags
        self.MeshObject.getInfo()
        # save original positions of nodes and points for perturbation later
        self.original_node_tags, self.original_node_coords , _ = gmsh.model.mesh.getNodes()
        self.p2 = p2
        self.p3 = p3


    def assemble_matrices(self):
        print("\n--- ASSEMBLING PASSIVE MATRICES ---")
        # distribute temperature gradient as function on the geometry
        T = kparams.plane_step_function(self.mesh, kparams.x_f, kparams.T_in, self.T_out)
        # calculate the sound speed function from temperature
        self.c = sound_speed(T)
        # calculate the passive acoustic matrices
        self.matrices = AcousticMatrices(self.mesh, self.facet_tags, self.boundary_conditions, T , self.degree) # very large, sparse matrices
        print("\n--- ASSEMBLING FLAME MATRIX ---")
        # using statespace model to define the flame transfer function
        FTF = stateSpace(kparams.S1, kparams.s2, kparams.s3, kparams.s4)
        # define input functions for the flame matrix
        # density function:
        rho = kparams.plane_tanh_function(self.mesh, kparams.x_f, kparams.a_f, self.Rho_out, kparams.rho_u)
        # use differnet functions for heat release h(x) and measurement w(x)
        if self.homogeneous_case:
            w = kparams.homogeneous_flame_functions(self.mesh, kparams.x_r)
            h = kparams.homogeneous_flame_functions(self.mesh, kparams.x_f)
        else:
            w = kparams.plane_gauss_function(self.mesh, kparams.x_r, kparams.a_r)
            h = kparams.plane_gauss_function(self.mesh, kparams.x_f, kparams.a_f)
        # calculate the flame matrix
        self.D = DistributedFlameMatrix(self.mesh, w, h, rho, T, kparams.q_0, kparams.u_b, FTF, self.degree, gamma=kparams.gamma)


    def solve_eigenvalue_problem(self):
        print("\n--- STARTING NEWTON METHOD ---")
        # unit of target: ([Hz])*2*pi = [rad/s] 
        target = (self.frequ)*2*np.pi
        print(f"---> \033[1mTarget\033[0m: {target.real:.2f}  {target.imag:.2f}j")
        try:
            # direct problem
            print("\n- DIRECT PROBLEM -")
            self.D.assemble_submatrices('direct') # assemble direct flame matrix
            # calculate the eigenvalues and eigenvectors
            omega_dir, p_dir = newtonSolver(self.matrices, self.degree, self.D, target, nev=1, i=0, tol=1e-2, maxiter=70, problem_type='direct', print_results= False)
            print("- omega_dir:", omega_dir)
            # adjoint problem
            print("\n- ADJOINT PROBLEM -")
            self.D.assemble_submatrices('adjoint') # assemble adjoint flame matrix
            # calculate the eigenvalues and eigenvectors
            omega_adj, p_adj = newtonSolver(self.matrices, self.degree, self.D, target, nev=1, i=0, tol=1e-2, maxiter=70, problem_type='adjoint', print_results= False)
            print("- omega_adj:", omega_adj)
        except IndexError:
            print("XXX--IndexError--XXX") # convergence of target failed in given range of iterations and tolerance
        else:
            print("-> Iterations done - target converged successfully")
        print("\n--- SAVING EIGENSOLUTION ---")
        # save solutions in a dictionary with labels and values
        omega_dict = {'direct':omega_dir, 'adjoint': omega_adj}
        dict_writer(self.path+"/Results"+"/eigenvalues", omega_dict) # save as .txt
        # save direct and adjoint eigenvector solution as .xdmf files
        xdmf_writer(self.path+"/Results"+"/p_dir", self.mesh, p_dir)
        xdmf_writer(self.path+"/Results"+"/p_adj", self.mesh, p_adj)
        # save the eigenvalues and eigenvectors in the object for shape derivative calculation
        self.omega_dir = omega_dir
        self.omega_adj = omega_adj
        self.p_dir = p_dir
        self.p_adj = p_adj


    def calculate_discrete_derivative(self):
        print("\n--- PERTURBING THE MESH ---")
        # copy the original node coordinates in order to prevent overwriting
        node_coords = np.array(self.original_node_coords, dtype=np.float64)
        # for discrete shape derivatives the mesh needs to be perturbed
        # assign x,y,z coordinates to separate arrays
        xcoords = node_coords[0::3] # get x-coordinates
        ycoords = node_coords[1::3] # get y-coordinates
        zcoords = node_coords[2::3] # get z-coordinates
        # create list to store the indices of the plenum nodes
        plenum_node_indices = []
        perturbed_node_coordinates = node_coords
        # choose points which have smaller x coordinate then 10mm or have x coordinate of 10mm and y coordinate greater than 1mm
        # these are all the points in the plenum without the slit entry
        for i in range(len(xcoords)):
            if (xcoords[i] < self.plenum_length) or (xcoords[i] == self.plenum_length and ycoords[i] > self.slit_height):
                plenum_node_indices.append(i) # store the index of the plenum nodes in this array
        # perturb the chosen mesh points slightly in y direction
        # perturbation is percent based on the y-coordinate
        ycoords[plenum_node_indices] += ycoords[plenum_node_indices] / self.height * self.perturbation
        # update node y coordinates in mesh from the perturbed points and the unperturbed original points
        perturbed_node_coordinates[1::3] = ycoords
        # update node positions
        for tag, new_coords in zip(self.original_node_tags, perturbed_node_coordinates.reshape(-1,3)):
            gmsh.model.mesh.setNode(tag, new_coords, [])
        # update point positions
        gmsh.model.setCoordinates(self.p2, 0, self.perturbation + self.height, 0)
        gmsh.model.setCoordinates(self.p3, self.plenum_length, self.perturbation + self.height, 0)
        # optionally launch GUI to see the results
        # if '-nopopup' not in sys.argv:
        #   gmsh.fltk.run()
        # save perturbed mesh data in /Meshes directory
        gmsh.write("{}.msh".format(self.path+"/Meshes"+self.name+"_perturbed")) # save as .msh file
        write_xdmf_mesh(self.path+"/Meshes"+self.name+"_perturbed",dimension=2) # save as .xdmf file
        print("\n--- RELOADING MESH ---")
        # reload to the perturbed mesh
        MeshObject_perturbed = XDMFReader(self.path+"/Meshes"+self.name+"_perturbed")
        self.perturbed_mesh, perturbed_subdomains, self.perturbed_facet_tags = MeshObject_perturbed.getAll() # mesh, domains and tags
        MeshObject_perturbed.getInfo()
        print("\n--- REASSEMBLING PASSIVE MATRICES ---")
        # recalculate the acoustic matrices for the perturbed mesh
        # define temperature gradient function in geometry
        T_pert = kparams.plane_step_function(self.perturbed_mesh, kparams.x_f, kparams.T_in, self.T_out)
        # calculate the passive acoustic matrices
        perturbed_matrices = AcousticMatrices(self.perturbed_mesh, self.perturbed_facet_tags, self.boundary_conditions, T_pert , self.degree) # very large, sparse matrices
        print("\n--- CALCULATING DISCRETE SHAPE DERIVATIVES ---")
        print("- calculate difference perturbed matrices")
        diff_A = perturbed_matrices.A - self.matrices.A
        diff_C = perturbed_matrices.C - self.matrices.C
        # using formula of numeric/discrete shape derivative
        print("- assembling numerator matrix")
        Mat_n = diff_A + self.omega_dir**2 * diff_C
        # multiply numerator matrix with direct and adjoint conjugate eigenvector
        # vector_matrix_vector automatically conjugates transposes p_adj
        numerator = vector_matrix_vector(self.p_adj.vector, Mat_n, self.p_dir.vector)
        # assemble flame matrix
        self.D.assemble_submatrices('direct')
        print("- assembling denominator matrix")
        Mat_d = -2*(self.omega_dir)*self.matrices.C + self.D.get_derivative(self.omega_dir)
        # multiply denominator matrix with direct and adjoint conjugate eigenvector
        # vector_matrix_vector automatically conjugates transposes p_adj
        denominator = vector_matrix_vector(self.p_adj.vector, Mat_d, self.p_dir.vector)
        print("- total shape derivative...")
        print("- numerator:", numerator)
        print("- denominator:", denominator)
        # calculate quotient of complex number
        self.derivative = numerator/denominator
        # normalize with the perturbation
        self.derivative = self.derivative / self.perturbation


    def calculate_continuous_derivative(self):
        print("\n--- CALCULATING CONTINUOUS SHAPE DERIVATIVES ---")
        physical_facet_tags = {1: 'inlet', 5: 'upper plenum'}
        selected_facet_tag = 5 # tag of the wall to be displaced
        selected_boundary_condition = self.boundary_conditions[selected_facet_tag]
        print("- boundary:", selected_boundary_condition)
        # [1,0] for inlet
        # [0,1] for plenum
        norm_vector_inlet = [0,1] # normal outside vector of the wall to be displaced
        # calculate the shape derivatives for the border
        print("- calculating shape derivative")
        self.derivative = ShapeDerivativesFFDRectFullBorder(self.MeshObject, selected_facet_tag, selected_boundary_condition,
                                                             norm_vector_inlet, self.omega_dir, self.p_dir, self.p_adj, self.c, self.matrices, self.D)


    def log(self):
        # print most important parameters and results of the calculation
        print("\n")
        if self.omega_dir.imag > 0: 
            stability = 'instable'
        else:
            stability = 'stable'
        print(f"---> \033[1mMesh Resolution =\033[0m {self.mesh_resolution}")
        print(f"---> \033[1mDimensions =\033[0m {self.plenum_length}m, {self.height} m")
        print(f"---> \033[1mPolynomial Degree of FEM =\033[0m {self.degree}")
        print(f"---> \033[1mPerturbation Distance =\033[0m {self.perturbation} m")
        print(f"---> \033[1mTarget =\033[0m {self.frequ} Hz ")
        print(f"---> \033[1mEigenfrequency =\033[0m {round(self.omega_dir.real/2/np.pi,4)} + {round(self.omega_dir.imag/2/np.pi,4)}j Hz ({stability})")
        print(f"---> \033[1m{self.type.capitalize()} Shape Derivative =\033[0m {round(self.derivative.real/2/np.pi,8)} + {round(self.derivative.imag/2/np.pi,8)}j")


    def write_input_functions(self):
        # create and write the functions with plane flame shape for checking in paraview
        ### create CURVED flame functions
        # rho_func = kparams.curved_tanh_function(self.mesh, kparams.x_f, kparams.a_f, kparams.rho_d, kparams.rho_u, kparams.amplitude, kparams.sig, kparams.limit)
        # w_func = kparams.point_gauss_function(self.mesh, kparams.x_r, kparams.a_r)
        # h_func = kparams.curved_gauss_function(self.mesh, kparams.x_f, kparams.a_f/2, kparams.amplitude, kparams.sig) 
        # T_func = kparams.curved_step_function(self.mesh, kparams.x_f,kparams.T_in, kparams.T_out, kparams.amplitude, kparams.sig)
        # c_func = sound_speed(T_func)
        ### create PLANE flame functions
        h_func = kparams.plane_gauss_function(self.mesh, kparams.x_f, kparams.a_f)
        w_func = kparams.plane_gauss_function(self.mesh, kparams.x_r, kparams.a_r)
        rho_func = kparams.plane_tanh_function(self.mesh, kparams.x_f, kparams.a_f, kparams.rho_d, kparams.rho_u)
        T_func = kparams.plane_step_function(self.mesh, kparams.x_f, kparams.T_in, kparams.T_out)
        c_func = sound_speed(T_func)
        V_ffd = ffd_displacement_vector_rect_full_border(self.MeshObject, 5, [0,1], deg=1)
        # save the functions in the InputFunctions directory as .xdmf files used to examine with paraview  
        xdmf_writer("InputFunctions/rho", self.mesh, rho_func)
        xdmf_writer("InputFunctions/w", self.mesh, w_func)
        xdmf_writer("InputFunctions/h", self.mesh, h_func)
        xdmf_writer("InputFunctions/c", self.mesh, c_func)
        xdmf_writer("InputFunctions/T", self.mesh, T_func)
        xdmf_writer(self.path+"/InputFunctions/V_ffd", self.mesh, V_ffd)