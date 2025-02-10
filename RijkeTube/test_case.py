'''
python class for test cases in the helmholtzX project
'''

import datetime
import os
import gmsh
import numpy as np
from mpi4py import MPI
# python script imports
import rparams
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
    def __init__(self, name, type, mesh_resolution, tube_length, tube_height, degree, frequ, perturbation, boundary_conditions, homogeneous_case, path):
        # initialize standard parameters of object
        self.type = type
        self.name = name
        self.mesh_resolution = mesh_resolution
        self.tube_length = tube_length
        self.tube_height = tube_height
        self.degree = degree
        self.perturbation = perturbation
        self.frequ = frequ
        self.homogeneous_case = homogeneous_case
        self.boundary_conditions = boundary_conditions
        self.homogeneous_case = homogeneous_case
        self.path = path
        # initialize parameters for homogeneous or inhomogeneous case
        if self.homogeneous_case: # homogeneous case
            self.T_out = rparams.T_in # no temperature gradient
            self.Rho_out = rparams.rho_u # no density gradient
        else: # inhomogeneous case
            self.T_out = rparams.T_out
            self.Rho_out = rparams.rho_d


    def create_rijke_tube_mesh(self):
        print("\n--- CREATING MESH ---")
        gmsh.initialize() # start the gmsh session
        gmsh.model.add(self.name) # add the model name
        # locate the points of the 2D geometry: [m]
        p1 = gmsh.model.geo.addPoint(0, 0, 0, self.mesh_resolution)  
        p2 = gmsh.model.geo.addPoint(0, self.tube_height, 0, self.mesh_resolution)
        p3 = gmsh.model.geo.addPoint(self.tube_length, self.tube_height, 0, self.mesh_resolution)
        p4 = gmsh.model.geo.addPoint(self.tube_length, 0, 0, self.mesh_resolution)
        # create outlines by connecting points
        l1 = gmsh.model.geo.addLine(p1, p2) # inlet boundary
        l2 = gmsh.model.geo.addLine(p2, p3) # upper wall
        l3 = gmsh.model.geo.addLine(p3, p4) # outlet boundary
        l4 = gmsh.model.geo.addLine(p4, p1) # lower wall
        # create curve loops for surface
        loop1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4]) # entire geometry
        # create surfaces from the curved loops
        surface1 = gmsh.model.geo.addPlaneSurface([loop1]) # surface of entire geometry
        # assign physical tags for 1D boundaries
        gmsh.model.addPhysicalGroup(1, [l1], tag=1) # inlet boundary
        gmsh.model.addPhysicalGroup(1, [l3], tag=2) # outlet boundary
        gmsh.model.addPhysicalGroup(1, [l4], tag=3) # lower wall
        gmsh.model.addPhysicalGroup(1, [l2], tag=4) # upper wall
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
        self.p3 = p3
        self.p4 = p4


    def assemble_matrices(self):
        print("\n--- ASSEMBLING PASSIVE MATRICES ---")
        # distribute temperature gradient as function on the geometry
        T = rparams.temperature_step_function(self.mesh, rparams.x_f, rparams.T_in, self.T_out)
        # calculate the sound speed function from temperature
        self.c = sound_speed(T)
        # calculate the passive acoustic matrices
        self.matrices = AcousticMatrices(self.mesh, self.facet_tags, self.boundary_conditions, T , self.degree) # very large, sparse matrices
        print("\n--- ASSEMBLING FLAME MATRIX ---")
        # using nTau model to define the flame transfer function
        FTF = nTau(rparams.n, rparams.tau)
        # define input functions for the flame matrix
        # density function:
        rho = rparams.rho_function(self.mesh, rparams.x_f, rparams.a_f, self.Rho_out, rparams.rho_u)
        # use differnet functions for heat release h(x) and measurement w(x)
        if self.homogeneous_case:
            w = rparams.homogeneous_flame_functions(self.mesh, rparams.x_r, rparams.a_r)
            h = rparams.homogeneous_flame_functions(self.mesh, rparams.x_f, rparams.a_f)
        else:
            w = rparams.flame_functions(self.mesh, rparams.x_r, rparams.a_r)
            h = rparams.flame_functions(self.mesh, rparams.x_f, rparams.a_f)
        # calculate the flame matrix
        self.D = DistributedFlameMatrix(self.mesh, w, h, rho, T, rparams.q_0, rparams.u_b, FTF, self.degree, gamma=rparams.gamma)


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
        self.omega_dir = omega_dir
        self.omega_adj = omega_adj
        self.p_dir = p_dir
        self.p_adj = p_adj


    def calculate_discrete_derivative(self):
        print("\n--- PERTURBING THE MESH ---")
        # for discrete shape derivatives the mesh needs to be perturbed
        # read tags and coordinates of the mesh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # assign x,y,z coordinates to separate arrays
        xcoords = node_coords[0::3] # get x-coordinates
        ycoords = node_coords[1::3] # get y-coordinates
        zcoords = node_coords[2::3] # get z-coordinates
        # perturb the chosen mesh points slightly in x direction
        for i, x in enumerate(xcoords):
            if x > 0.3: # only perturb parts of mesh that lie behind the flame located at 0.25m
                xcoords[i] += (x - 0.3)/(self.tube_length - 0.3) * self.perturbation
        #xcoords += xcoords * perturbation #(case for perturbing the whole duct disregarding the flame)
        # update node x coordinates in mesh to the perturbed points
        node_coords[0::3] = xcoords
        # update node positions
        for tag, new_coords in zip(node_tags, node_coords.reshape(-1,3)):
            gmsh.model.mesh.setNode(tag, new_coords, [])
        # update point positions
        gmsh.model.setCoordinates(self.p3, self.perturbation + self.tube_length, self.tube_height, 0)
        gmsh.model.setCoordinates(self.p4, self.perturbation + self.tube_length, 0, 0)
        # optionally launch GUI to see the results
        #if '-nopopup' not in sys.argv:
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
        T_pert = rparams.temperature_step_function(self.perturbed_mesh, rparams.x_f, rparams.T_in, self.T_out) # the central variable that affects is T_out! if changed to T_in we get the correct homogeneous starting case
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


    def calculate_continuous_derivative(self):
        print("\n--- CALCULATING CONTINUOUS SHAPE DERIVATIVES ---")
        # 1 for inlet
        # 2 for outlet
        physical_facet_tags = {1: 'inlet', 2: 'outlet'}
        selected_facet_tag = 2 # tag of the wall to be displaced
        selected_boundary_condition = self.boundary_conditions[selected_facet_tag]
        print("BOUNDARY:", selected_boundary_condition)
        # [-1,0] for inlet
        # [1,0] for outlet
        norm_vector_inlet = [1,0] # normal outside vector of the wall to be displaced
        # calculate the shape derivatives for the border
        print("- calculating shape derivative")
        self.derivative = ShapeDerivativesFFDRectFullBorder(self.MeshObject, selected_facet_tag, selected_boundary_condition, norm_vector_inlet, self.omega_dir, self.p_dir, self.p_adj, self.c, self.matrices, self.D)


    def log(self):
        # print most important parameters and results of the calculation
        print("\n")
        if self.omega_dir.imag > 0: 
            stability = 'instable'
        else:
            stability = 'stable'
        print(f"---> \033[1mMesh Resolution =\033[0m {self.mesh_resolution}")
        print(f"---> \033[1mDimensions =\033[0m {self.tube_length}m, {self.tube_height} m")
        print(f"---> \033[1mPolynomial Degree of FEM =\033[0m {self.degree}")
        print(f"---> \033[1mPerturbation Distance =\033[0m {self.perturbation} m")
        print(f"---> \033[1mTarget =\033[0m {self.frequ} Hz ")
        print(f"---> \033[1mEigenfrequency =\033[0m {round(self.omega_dir.real/2/np.pi,4)} + {round(self.omega_dir.imag/2/np.pi,4)}j Hz ({stability})")
        print(f"---> \033[1mDiscrete Shape Derivative =\033[0m {round(self.derivative.real/2/np.pi,8)} + {round(self.derivative.imag/2/np.pi,8)}j")
        if self.type == 'discrete':
            print(f"---> \033[1mNormalized Discrete Shape Derivative =\033[0m {round(self.derivative.real/2/np.pi/self.perturbation,8)} + {round(self.derivative.imag/2/np.pi/self.perturbation,8)}j") 


