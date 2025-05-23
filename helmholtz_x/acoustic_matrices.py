from dolfinx.fem import Function, FunctionSpace, dirichletbc, form, locate_dofs_topological, Constant, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix
from .parameters_utils import sound_speed_variable_gamma, gamma_function
from .solver_utils import info
from ufl import Measure, TestFunction, TrialFunction, grad, inner
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import logging


# find the acoustic matrices A, B and C for discrete helmholtz equation
class AcousticMatrices:
    # constructor
    def __init__(self, mesh, facet_tags, boundary_conditions,
                 parameter, degree=1):
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.boundary_conditions = boundary_conditions
        self.parameter = parameter
        self.degree = degree
        self.dimension = self.mesh.topology.dim
        self.fdim = self.mesh.topology.dim - 1
        self.dx = Measure('dx', domain=mesh) # volume/area
        self.ds = Measure('ds', domain=mesh, subdomain_data=facet_tags) # boundaries
        # creating the function space
        # degree gives the base of the used polynomial for the test function later
        self.V = FunctionSpace(self.mesh, ("Lagrange", degree))
        # trial and test function definition
        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)
        
        self.AreaConstant = Constant(mesh, PETSc.ScalarType(1))
        self.bcs_Dirichlet = []
        self.integrals_R = []
        # define placeholder for matrices
        self.a_form = None
        self.b_form = None
        self.c_form = None
        self._A = None
        self._B = None
        self._B_adj = None
        self._C = None

        if MPI.COMM_WORLD.rank == 0:
            logging.debug("- Degree of basis functions: %d", self.degree)

        # usually temperature is the case, as we create the matrices from distribution of T
        if self.parameter.name == "temperature":
            self.c = sound_speed_variable_gamma(self.mesh, parameter, degree=degree)
            self.T = self.parameter
            self.gamma = gamma_function(self.T)
            logging.debug("- Temperature function is used for passive flame matrices.")
        else:
            self.c = parameter
            self.gamma = self.c.copy()
            self.gamma.x.array[:] = 1.4
            logging.debug("- Speed of sound function is used for passive flame matrices.")

        # Boundary Conditions:
        for boundary in boundary_conditions:
            if 'Neumann' in boundary_conditions[boundary]:
                logging.debug("- Neumann boundaries on boundary "+str(boundary))
                
            if 'Dirichlet' in boundary_conditions[boundary]:
                u_bc = Function(self.V)
                facets = np.array(self.facet_tags.indices[self.facet_tags.values == boundary])
                dofs = locate_dofs_topological(self.V, self.fdim, facets)
                bc = dirichletbc(u_bc, dofs)
                self.bcs_Dirichlet.append(bc)
                logging.debug("- Dirichlet boundary on boundary "+str(boundary))

            if 'Robin' in boundary_conditions[boundary]:
                R = boundary_conditions[boundary]['Robin']
                Z = (1+R)/(1-R)
                integrals_Impedance = 1j * self.c / Z * inner(self.phi_i, self.phi_j) * self.ds(boundary)
                self.integrals_R.append(integrals_Impedance)
                logging.debug("- Robin boundary on boundary "+str(boundary))

            if 'ChokedInlet' in boundary_conditions[boundary]:
                A_inlet = MPI.COMM_WORLD.allreduce(assemble_scalar(form(self.AreaConstant * self.ds(boundary))), op=MPI.SUM)
                gamma_inlet_form = form(self.gamma/A_inlet* self.ds(boundary))
                gamma_inlet = MPI.COMM_WORLD.allreduce(assemble_scalar(gamma_inlet_form), op=MPI.SUM)

                Mach = boundary_conditions[boundary]['ChokedInlet']
                R = (1-gamma_inlet*Mach/(1+(gamma_inlet-1)*Mach**2))/(1+gamma_inlet*Mach/(1+(gamma_inlet-1)*Mach**2))
                Z = (1+R)/(1-R)
                integral_C_i = 1j * self.c / Z * inner(self.phi_i, self.phi_j) * self.ds(boundary)
                self.integrals_R.append(integral_C_i)
                logging.debug("- Choked inlet boundary on boundary "+str(boundary))

            if 'ChokedOutlet' in boundary_conditions[boundary]:
                A_outlet = MPI.COMM_WORLD.allreduce(assemble_scalar(form(self.AreaConstant * self.ds(boundary))), op=MPI.SUM)
                gamma_outlet_form = form(self.gamma/A_outlet* self.ds(boundary))
                gamma_outlet = MPI.COMM_WORLD.allreduce(assemble_scalar(gamma_outlet_form), op=MPI.SUM)

                Mach = boundary_conditions[boundary]['ChokedOutlet']
                R = (1-0.5*(gamma_outlet-1)*Mach)/(1+0.5*(gamma_outlet-1)*Mach)
                Z = (1+R)/(1-R)
                integral_C_o = 1j * self.c / Z * inner(self.phi_i, self.phi_j) * self.ds(boundary)
                self.integrals_R.append(integral_C_o)
                logging.debug("- Choked outlet boundary on boundary "+str(boundary))

        logging.debug("- Passive matrices are assembling..")

        # Assemble the matrix A - stiffness matrix
        self.a_form = form(-self.c**2* inner(grad(self.phi_i), grad(self.phi_j))*self.dx) # symbolic form
        A = assemble_matrix(self.a_form, bcs=self.bcs_Dirichlet) # actual matrix assembly NxN
        A.assemble() # finalizing assembly
        logging.debug("- Matrix A is assembled.")
        self._A = A

        # assemble matrix B - boundary matrix
        if self.integrals_R:
            self.b_form = form(sum(self.integrals_R))
            B = assemble_matrix(self.b_form)
            B.assemble()
            # find adjoint matrix B
            B_adj = B.copy()
            B_adj.transpose()
            B_adj.conjugate()
            logging.debug("- Matrix B is assembled.")
            self._B = B
            self._B_adj = B_adj

        # assemble matrix C - mass matrix
        self.c_form = form(inner(self.phi_i , self.phi_j) * self.dx)
        C = assemble_matrix(self.c_form, self.bcs_Dirichlet)
        C.assemble()
        logging.debug("- Matrix C is assembled.")
        self._C = C

    @property
    def A(self):
        return self._A
    @property
    def B(self):
        return self._B
    @property
    def B_adj(self):
        return self._B_adj
    @property
    def C(self):
        return self._C