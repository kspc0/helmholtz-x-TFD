'''
python class for computing test cases in the helmholtzX-TFD project
'''
# standard python libraries
import datetime
import gmsh
import sys
import numpy as np
import logging
from petsc4py import PETSc
from dolfinx.fem import Function, FunctionSpace, Expression, form
# HelmholtzX utilities
import helmholtz_x.distribute_params as dist_params
from helmholtz_x.io_utils import XDMFReader, dict_writer, xdmf_writer, write_xdmf_mesh # to write mesh data as files
from helmholtz_x.parameters_utils import sound_speed # to calculate sound speed from temperature
from helmholtz_x.acoustic_matrices import AcousticMatrices # to assemble the acoustic matrices for discrete Helm. EQU
from helmholtz_x.flame_transfer_function import nTau, stateSpace # to define the flame transfer function
from helmholtz_x.flame_matrices import DistributedFlameMatrix # to define the flame matrix for discrete Helm. EQU
from helmholtz_x.eigensolvers import newtonSolver # to solve the system
from helmholtz_x.eigenvectors import normalize_adjoint
from helmholtz_x.petsc4py_utils import vector_matrix_vector, conjugate, conjugate_function
from helmholtz_x.shape_derivatives import ShapeDerivativeFullBorder, ffd_displacement_vector_full_border # to calculate shape derivatives



# class for computing different test cases
class TestCase:
    # constructor
    def __init__(self, name, type, hom_case, path):
        # mark the processing time
        self.start_time = datetime.datetime.now()
        # initialize standard parameters of object
        self.name = name # Kornilov or RijkeTube
        self.type = type # discrete or continuous shape derivative
        self.path = path # path to the current directory
        # save the python script parameter import as a public variable
        if self.name == "/KornilovCase":
            import KornilovCase.kparams as params
        elif self.name == "/RijkeTube":
            import RijkeTube.rparams as params
        else:
            logging.error("required parameter file not found")
        self.par = params # set public variable
        # main parameters of geometry
        self.mesh_resolution = self.par.mesh_resolution
        self.mesh_refinement_factor = self.par.mesh_refinement_factor
        self.length = self.par.length
        self.height = self.par.height
        # test case parameters
        self.degree = self.par.degree
        self.perturbation = self.par.perturbation
        self.target = self.par.frequ
        self.boundary_conditions = self.par.boundary_conditions
        self.homogeneous_case = hom_case # fast change between homogeneous and inhomogeneous case
        # initialize input/output values for homogeneous or inhomogeneous case
        if self.homogeneous_case: # homogeneous case
            self.T_out = self.par.T_in # no temperature gradient
            self.Rho_out = self.par.rho_u # no density gradient
        else: # inhomogeneous case
            self.T_out = self.par.T_out
            self.Rho_out = self.par.rho_d

    # build the mesh of the kornilov case
    def create_kornilov_mesh(self):
        logging.debug("\n--- CREATING MESH ---")
        gmsh.initialize() # start the gmsh session
        gmsh.option.setNumber('General.Terminal', 0) # disable terminal output
        gmsh.model.add(self.name) # add the model name
        factor = 40        
        self.offset = 0#(1e-3)*factor # positive shift of the geometry in x direction to prevent negative coordinates
        self.slit = (1e-3)*factor # measure of the slit
        self.combustion_chamber_height = (2.5e-3)*factor # height of the combustion chamber
        # locate the points of the 2D geometry: [m]
        p1 = gmsh.model.geo.addPoint(self.offset, 0, 0, self.mesh_resolution)  
        p2 = gmsh.model.geo.addPoint(self.offset, self.height, 0, self.mesh_resolution)
        p3 = gmsh.model.geo.addPoint(self.offset+self.length, self.height, 0, self.mesh_resolution)
        p4 = gmsh.model.geo.addPoint(self.offset+self.length, self.slit, 0, self.mesh_resolution/self.mesh_refinement_factor)
        p5 = gmsh.model.geo.addPoint(self.offset+self.length+self.slit, self.slit, 0, self.mesh_resolution/self.mesh_refinement_factor)
        p6 = gmsh.model.geo.addPoint(self.offset+self.length+self.slit, self.combustion_chamber_height, 0, self.mesh_resolution)
        p7 = gmsh.model.geo.addPoint(self.offset+self.length*3+self.slit, self.combustion_chamber_height, 0, self.mesh_resolution)
        p8 = gmsh.model.geo.addPoint(self.offset+self.length*3+self.slit, 0, 0, self.mesh_resolution)
        # p1 = gmsh.model.geo.addPoint(-self.length-self.slit, 0, 0, self.mesh_resolution)  
        # p2 = gmsh.model.geo.addPoint(-self.length-self.slit, self.height, 0, self.mesh_resolution)
        # p3 = gmsh.model.geo.addPoint(-self.slit, self.height, 0, self.mesh_resolution)
        # p4 = gmsh.model.geo.addPoint(-self.slit, self.slit, 0, self.mesh_resolution/self.mesh_refinement_factor)
        # p5 = gmsh.model.geo.addPoint(0, self.slit, 0, self.mesh_resolution/self.mesh_refinement_factor)
        # p6 = gmsh.model.geo.addPoint(0, self.combustion_chamber_height, 0, self.mesh_resolution)
        # p7 = gmsh.model.geo.addPoint(self.length*2, self.combustion_chamber_height, 0, self.mesh_resolution)
        # p8 = gmsh.model.geo.addPoint(self.length*2, 0, 0, self.mesh_resolution)
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
        logging.debug("\n--- LOADING MESH ---")
        self.MeshObject = XDMFReader(self.path+"/Meshes"+self.name)
        self.mesh, self.subdomains, self.facet_tags = self.MeshObject.getAll() # mesh, domains and tags
        self.MeshObject.getInfo()
        # save original positions of nodes and points for perturbation later
        self.original_node_tags, self.original_node_coords , _ = gmsh.model.mesh.getNodes()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    # build the mesh of the rijke tube case
    def create_rijke_tube_mesh(self):
        logging.debug("\n--- CREATING MESH ---")
        gmsh.initialize() # start the gmsh session
        gmsh.option.setNumber('General.Terminal', 0) # disable terminal output
        gmsh.model.add(self.name) # add the model name
        # locate the points of the 2D geometry: [m]
        p1 = gmsh.model.geo.addPoint(0, 0, 0, self.mesh_resolution)  
        p2 = gmsh.model.geo.addPoint(0, self.height, 0, self.mesh_resolution)
        p3 = gmsh.model.geo.addPoint(self.length/4, self.height, 0, self.mesh_resolution/self.mesh_refinement_factor) # refined at flame
        p4 = gmsh.model.geo.addPoint(self.length, self.height, 0, self.mesh_resolution)
        p5 = gmsh.model.geo.addPoint(self.length, 0, 0, self.mesh_resolution)
        p6 = gmsh.model.geo.addPoint(self.length/4, 0, 0, self.mesh_resolution/self.mesh_refinement_factor) # refined at flame
        # create outlines by connecting points
        l1 = gmsh.model.geo.addLine(p1, p2) # inlet boundary
        l2 = gmsh.model.geo.addLine(p2, p3) # upper wall
        l3 = gmsh.model.geo.addLine(p3, p4) # upper wall
        l4 = gmsh.model.geo.addLine(p4, p5) # outlet boundary
        l5 = gmsh.model.geo.addLine(p5, p6) # lower wall
        l6 = gmsh.model.geo.addLine(p6, p1) # lower wall
        # create curve loops for surface
        loop1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4,l5,l6]) # entire geometry
        # create surfaces from the curved loops
        surface1 = gmsh.model.geo.addPlaneSurface([loop1]) # surface of entire geometry
        # assign physical tags for 1D boundaries
        gmsh.model.addPhysicalGroup(1, [l1], tag=1) # inlet boundary
        gmsh.model.addPhysicalGroup(1, [l4], tag=2) # outlet boundary
        gmsh.model.addPhysicalGroup(1, [l5, l6], tag=3) # lower wall
        gmsh.model.addPhysicalGroup(1, [l2, l3], tag=4) # upper wall
        # assign physical tag for 2D surface
        gmsh.model.addPhysicalGroup(2, [surface1], tag=1)
        # create 2D mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        # optionally launch GUI to see the results
        # if '-nopopup' not in sys.argv:
        #   gmsh.fltk.run() 
        # save data in /Meshes directory
        gmsh.write("{}.msh".format(self.path+"/Meshes"+self.name)) # save as .msh file
        write_xdmf_mesh(self.path+"/Meshes"+self.name, dimension=2) # save as .xdmf file
        logging.debug("\n--- LOADING MESH ---")
        self.MeshObject = XDMFReader(self.path+"/Meshes"+self.name)
        self.mesh, self.subdomains, self.facet_tags = self.MeshObject.getAll() # mesh, domains and tags
        self.MeshObject.getInfo()
        # save original positions of nodes and points for perturbation later
        self.original_node_tags, self.original_node_coords , _ = gmsh.model.mesh.getNodes()
        self.p3 = p4 # end nodes of the tube need to be perturbed later
        self.p4 = p5

    # assemble the passive matrices and the flame matrix
    def assemble_matrices(self):
        logging.debug("\n--- ASSEMBLING PASSIVE MATRICES ---")
        # distribute temperature gradient as function on the geometry
        T = dist_params.step_function(self.mesh, self.par.x_f, self.par.T_in, self.T_out)
        # calculate the sound speed function from temperature
        self.c = sound_speed(T)
        # calculate the passive acoustic matrices
        self.matrices = AcousticMatrices(self.mesh, self.facet_tags, self.boundary_conditions, T , self.degree) # very large, sparse matrices
        logging.debug("\n--- ASSEMBLING FLAME MATRIX ---")
        if self.name == "/KornilovCase":
            # using statespace model to define the flame transfer function
            FTF = stateSpace(self.par.S1, self.par.s2, self.par.s3, self.par.s4)
        elif self.name == "/RijkeTube":
            # using nTau model to define the flame transfer function
            FTF = nTau(self.par.n, self.par.tau)
        # define input functions for the flame matrix
        # density function:
        rho = dist_params.tanh_function(self.mesh, self.par.x_f, self.par.a_f, self.Rho_out, self.par.rho_u)
        # use differnet functions for heat release h(x) and measurement w(x)
        if self.homogeneous_case:
            w = dist_params.homogeneous_function(self.mesh, self.par.x_r)
            h = dist_params.homogeneous_function(self.mesh, self.par.x_f)
        else:
            w = dist_params.gauss_function(self.mesh, self.par.x_r, self.par.a_r)
            h = dist_params.gauss_function(self.mesh, self.par.x_f, self.par.a_f)
        # calculate the flame matrix
        self.D = DistributedFlameMatrix(self.mesh, w, h, rho, T, self.par.q_0, self.par.u_b, FTF, self.degree, gamma=self.par.gamma)

    # solve the eigenvalue problem for the given target frequency
    def solve_eigenvalue_problem(self):
        logging.info("\n--- COMPUTING EIGENSOLUTION ---") 
        # unit of target: ([Hz])*2*pi = [rad/s] 
        target = (self.target)*2*np.pi
        logging.info(f"---> \033[1mTarget\033[0m: {target.real:.2f}  {target.imag:.2f}j")
        try:
            # direct problem
            logging.debug("\n- DIRECT PROBLEM -")
            self.D.assemble_submatrices('direct') # assemble direct flame matrix
            # calculate the eigenvalues and eigenvectors
            omega_dir, p_dir, p_adj = newtonSolver(self.matrices, self.degree, self.D, target, nev=1,
                                                    i=0, tol=1e-2, maxiter=70, problem_type='direct', print_results= False)
            #print("- omega_dir:", omega_dir)
            omega_adj = np.conj(omega_dir) # conjugate eigenvalue
        except IndexError:
            logging.error("IndexError: convergence of target failed in given range of iterations and tolerance")
        else:
            logging.debug("-> Iterations done - target converged successfully")
        logging.debug("\n--- SAVING EIGENSOLUTION ---")
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

    def compute_residual(self):
        logging.debug("\n--- TESTING EIGENSOLUTION ---")
        MAT = (self.matrices.A + self.omega_dir**2 * self.matrices.C - self.D.get_derivative(self.omega_dir))
        residual_dir = self.p_dir.vector.copy()
        MAT.mult(self.p_dir.vector, residual_dir)
        logging.debug("- residual of the direct eigenvalue problem computed")
        # save residual vector as xdmf file
        V = FunctionSpace(self.mesh, ("CG", self.degree))
        resi_dir = Function(V)
        resi_dir.vector.setArray(residual_dir)
        resi_dir.x.scatter_forward()
        xdmf_writer(self.path+"/Results"+"/residual", self.mesh, resi_dir)

    # slightly perturb the kornilov mesh to get perturbed matrices
    def perturb_kornilov_mesh(self, pert_method):
        logging.debug("\n--- PERTURBING THE MESH ---")
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
        if pert_method == "y": # increase of plenum height
            logging.info("- perturbation method: y-direction")
            # choose points which have smaller x coordinate then 10mm 
            # or have x coordinate of 10mm and y coordinate greater than 1mm
            # these are all the points in the plenum without the slit entry
            for i in range(len(xcoords)):
                if xcoords[i] <= self.offset+self.length:
                    plenum_node_indices.append(i) # store the index of the plenum nodes in this array
            # perturb the chosen mesh points slightly in y direction
            # perturbation is percent based on the y-coordinate
            ycoords[plenum_node_indices] += ycoords[plenum_node_indices] / self.height * self.perturbation
            # update node y coordinates in mesh from the perturbed points and the unperturbed original points
            perturbed_node_coordinates[1::3] = ycoords
        elif pert_method == "x": # change in inlet direction
            logging.info("- perturbation method: x-direction")
            for i in range(len(xcoords)):
                if xcoords[i] < 1.5e-3:
                    plenum_node_indices.append(i)
            xcoords[plenum_node_indices] += (xcoords[plenum_node_indices]-self.length-self.offset) / (self.length) * self.perturbation
            perturbed_node_coordinates[0::3] = xcoords
        else:
            logging.error("Warning: unknown perturbation method")
        # update node positions
        for tag, new_coords in zip(self.original_node_tags, perturbed_node_coordinates.reshape(-1,3)):
            gmsh.model.mesh.setNode(tag, new_coords, [])
        # update point positions
        if pert_method == "y": 
            gmsh.model.setCoordinates(self.p2, self.offset, self.perturbation + self.height, 0)
            gmsh.model.setCoordinates(self.p3, self.offset+self.length, self.perturbation + self.height, 0)
        elif pert_method == "x":
            gmsh.model.setCoordinates(self.p1, -self.perturbation+self.offset, 0, 0)
            gmsh.model.setCoordinates(self.p2, -self.perturbation+self.offset, self.height, 0)
        # optionally launch GUI to see the results
        # if '-nopopup' not in sys.argv:
        #   gmsh.fltk.run()
        # save perturbed mesh data in /Meshes directory
        gmsh.write("{}.msh".format(self.path+"/Meshes"+self.name+"_perturbed")) # save as .msh file
        write_xdmf_mesh(self.path+"/Meshes"+self.name+"_perturbed",dimension=2) # save as .xdmf file
        logging.debug("\n--- RELOADING MESH ---")
        # reload to the perturbed mesh
        MeshObject_perturbed = XDMFReader(self.path+"/Meshes"+self.name+"_perturbed")
        self.perturbed_mesh, perturbed_subdomains, self.perturbed_facet_tags = MeshObject_perturbed.getAll() # mesh, domains and tags
        MeshObject_perturbed.getInfo()
        logging.debug("\n--- REASSEMBLING PASSIVE MATRICES ---")
        # recalculate the acoustic matrices for the perturbed mesh
        # define temperature gradient function in geometry
        T_pert = dist_params.step_function(self.perturbed_mesh, self.par.x_f, self.par.T_in, self.T_out)
        # calculate the passive acoustic matrices
        self.perturbed_matrices = AcousticMatrices(self.perturbed_mesh, self.perturbed_facet_tags,
                                                    self.boundary_conditions, T_pert , self.degree) # very large, sparse matrices

    # slightly perturb the rijketube mesh to get perturbed matrices
    def perturb_rijke_tube_mesh(self, pert_method="linear"):
        logging.debug("\n--- PERTURBING THE MESH ---")
        # copy the original node coordinates in order to prevent overwriting
        node_coords = np.array(self.original_node_coords, dtype=np.float64)
        # for discrete shape derivatives the mesh needs to be perturbed
        # assign x,y,z coordinates to separate arrays
        xcoords = node_coords[0::3] # get x-coordinates
        ycoords = node_coords[1::3] # get y-coordinates
        # choose which perturbation method is applied
        if pert_method == "linear": # standard
            logging.info("- perturbation method: linear")
            # perturb the chosen mesh points slightly in x direction
            for i, x in enumerate(xcoords):
                if x > 0.3: # only perturb parts of mesh that lie behind the flame located at 0.25m
                    xcoords[i] += (x - 0.3)/(self.length - 0.3) * self.perturbation
        elif pert_method == "inside": # perturbation only inside the mesh, not the border
            logging.info("- perturbation method: inside")
            # perturb the chosen mesh points only inside the mesh domain, not modifying the borders
            for i in range(len(xcoords)):
                # slightly randomly perturb all points inside the mesh
                if 0.3 < xcoords[i] < 1:
                    xcoords[i] += self.perturbation
        elif pert_method == "random":
            #if 0.3 < xcoords[i] < 1 and 0 < ycoords[i] < 0.047:  # Only perturb points inside the mesh
                # Slightly randomly perturb all points inside the mesh in a random direction
                # random_perturbation_x = np.random.uniform(-self.perturbation, self.perturbation)
                # random_perturbation_y = np.random.uniform(-self.perturbation, self.perturbation)
                # xcoords[i] += random_perturbation_x
                # ycoords[i] += random_perturbation_y
            logging.warning("- perturbation method: random - not implemented yet!")
        elif pert_method == "border":
            logging.warning("- perturbation method: border - not implemented yet!")
        else:
            logging.error("Warning: unknown perturbation method")
        #xcoords += xcoords * perturbation #(case for perturbing the whole duct disregarding the flame)
        # update node x coordinates in mesh to the perturbed points
        perturbed_node_coordinates = node_coords
        perturbed_node_coordinates[0::3] = xcoords
        perturbed_node_coordinates[1::3] = ycoords
        # update node positions
        for tag, new_coords in zip(self.original_node_tags, perturbed_node_coordinates.reshape(-1,3)):
            gmsh.model.mesh.setNode(tag, new_coords, [])
        # update point positions
        if pert_method == "linear": # standard
            gmsh.model.setCoordinates(self.p3, self.perturbation + self.length, self.height, 0)
            gmsh.model.setCoordinates(self.p4, self.perturbation + self.length, 0, 0)
        # optionally launch GUI to see the results
        # if '-nopopup' not in sys.argv:
        #   gmsh.fltk.run()
        # save perturbed mesh data in /Meshes directory
        gmsh.write("{}.msh".format(self.path+"/Meshes"+self.name+"_perturbed")) # save as .msh file
        write_xdmf_mesh(self.path+"/Meshes"+self.name+"_perturbed",dimension=2) # save as .xdmf file
        logging.debug("\n--- RELOADING MESH ---")
        # reload to the perturbed mesh
        MeshObject_perturbed = XDMFReader(self.path+"/Meshes"+self.name+"_perturbed")
        self.perturbed_mesh, perturbed_subdomains, self.perturbed_facet_tags = MeshObject_perturbed.getAll() # mesh, domains and tags
        MeshObject_perturbed.getInfo()
        logging.debug("\n--- REASSEMBLING PASSIVE MATRICES ---")
        # recalculate the acoustic matrices for the perturbed mesh
        # define temperature gradient function in geometry
        T_pert = dist_params.step_function(self.perturbed_mesh, self.par.x_f, self.par.T_in, self.T_out)
        # calculate the passive acoustic matrices
        self.perturbed_matrices = AcousticMatrices(self.perturbed_mesh, self.perturbed_facet_tags, 
                                                   self.boundary_conditions, T_pert , self.degree) # very large, sparse matrices

    # calculate the shape derivative using the discrete formula
    def calculate_discrete_derivative(self):
        logging.info("\n--- COMPUTING DISCRETE SHAPE DERIVATIVES ---")
        logging.debug("- calculate difference perturbed matrices")
        diff_A = self.perturbed_matrices.A - self.matrices.A
        diff_C = self.perturbed_matrices.C - self.matrices.C
        logging.debug("- assembling numerator matrix")
        Mat_n = diff_A + self.omega_dir**2 * diff_C
        # normalize the adjoint eigenvector with the same measure from the continuous approach
        self.p_adj_norm = normalize_adjoint(self.omega_dir, self.p_dir, self.p_adj, self.matrices, self.D)
        # multiply numerator matrix with direct and adjoint conjugate eigenvector
        # vector_matrix_vector automatically conjugates transposes p_adj
        self.derivative = vector_matrix_vector(self.p_adj_norm.vector, Mat_n, self.p_dir.vector) / self.perturbation
    
    # calculate the shape derivative using continuous formula
    def calculate_continuous_derivative(self, tag):
        logging.info("\n--- COMPUTING CONTINUOUS SHAPE DERIVATIVES ---")
        if tag == "inlet":
            selected_facet_tag = 1 # tag of the wall to be displaced
            norm_vector = [-1,0] # normal vector of the wall to be displaced
        elif tag == "outlet":
            selected_facet_tag = 2
            norm_vector = [1,0]
        elif tag == "upper plenum":
            selected_facet_tag = 5
            norm_vector = [0,1]
        else:
            logging.error("Error: unknown facet tag")
        selected_boundary_condition = self.boundary_conditions[selected_facet_tag]
        logging.debug("- boundary: %s", selected_boundary_condition)
        # calculate the shape derivatives for the border
        self.derivative = ShapeDerivativeFullBorder(self.MeshObject, selected_facet_tag, selected_boundary_condition, norm_vector,
                                                     self.omega_dir, self.p_dir, self.p_adj, self.c, self.matrices, self.D)

    # terminal log of most important parameters and results of the calculation
    def log(self):
        logging.info("--- LOGGING ---")
        if self.omega_dir.imag > 0: 
            stability = 'instable'
        else:
            stability = 'stable'
        logging.info(f"---> \033[1mMesh Resolution =\033[0m {self.mesh_resolution}")
        logging.info(f"---> \033[1mDimensions =\033[0m {self.length}m, {self.height} m")
        logging.info(f"---> \033[1mPolynomial Degree of FEM =\033[0m {self.degree}")
        logging.info(f"---> \033[1mPerturbation Distance =\033[0m {self.perturbation} m")
        logging.info(f"---> \033[1mTarget =\033[0m {self.target} Hz ")
        logging.info(f"---> \033[1mOmega =\033[0m {self.omega_dir.real} + {self.omega_dir.imag}j ({stability})")
        logging.info(f"---> \033[1m(Physical Frequency) =\033[0m {self.omega_dir.real/-2/np.pi} Hz") # needs negative sign to fit the physical frequency
        logging.info(f"---> \033[1m(Physical Growth Rate) =\033[0m {self.omega_dir.imag/2/np.pi} 1/s")
        #logging.info(f"---> \033[1mEigenfrequency =\033[0m {round(self.omega_dir.real/2/np.pi,4)} + {round(self.omega_dir.imag/2/np.pi,4)}j Hz ({stability})")
        logging.info(f"---> \033[1m{self.type.capitalize()} Shape Derivative =\033[0m {round(self.derivative.real/2/np.pi,8)} + {round(self.derivative.imag/2/np.pi,8)}j")
        print("Total Execution Time: ", datetime.datetime.now()-self.start_time)

    # write the input functions for the paraview visualization
    def write_input_functions(self):
        logging.info("\n--- WRITING INPUT FUNCTIONS ---")
        # assemble the functions again
        h_func = dist_params.gauss_function(self.mesh, self.par.x_f, self.par.a_f)
        w_func = dist_params.gauss_function(self.mesh, self.par.x_r, self.par.a_r)
        rho_func = dist_params.tanh_function(self.mesh, self.par.x_f, self.par.a_f, self.par.rho_d, self.par.rho_u)
        T_func = dist_params.step_function(self.mesh, self.par.x_f, self.par.T_in, self.par.T_out)
        c_func = sound_speed(T_func)
        # displacement vector for the continuous shape derivative calculation
        V_ffd = ffd_displacement_vector_full_border(self.MeshObject, 1, [-1,0], deg=1)
        # save the functions in the InputFunctions directory as .xdmf files used to examine with paraview  
        xdmf_writer("InputFunctions/rho", self.mesh, rho_func)
        xdmf_writer("InputFunctions/w", self.mesh, w_func)
        xdmf_writer("InputFunctions/h", self.mesh, h_func)
        xdmf_writer("InputFunctions/c", self.mesh, c_func)
        xdmf_writer("InputFunctions/T", self.mesh, T_func)
        xdmf_writer(self.path+"/InputFunctions/V_ffd", self.mesh, V_ffd)