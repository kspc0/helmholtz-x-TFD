from ufl import Measure, TestFunction, TrialFunction, inner, as_vector, grad, dx
from dolfinx.cpp.geometry import determine_point_ownership
from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem  import FunctionSpace, Expression, form
from .parameters_utils import gamma_function
from .dolfinx_utils import distribute_vector_as_chunks, broadcast_vector
from .solver_utils import info
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import logging

# assembly of the flame matrix D(omega)
class FlameMatrix:
    def  __init__(self, mesh, h, q_0, u_b, FTF, degree, bloch_object=None, tol=1e-5):

        self.mesh = mesh
        self.h = h
        self.q_0 = q_0 
        self.u_b = u_b
        self.FTF = FTF
        self.degree = degree
        self.tol = tol
        
        # create function space
        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        # definition of trial and test function
        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)
        self.gdim = self.mesh.geometry.dim

        # matrix utils
        self.global_size = self.V.dofmap.index_map.size_global
        self.local_size = self.V.dofmap.index_map.size_local

        # vector for reference direction
        if self.gdim == 1:
            self.n_r = as_vector([1])
        elif self.gdim == 2:
            self.n_r = as_vector([1,0])
        else:
            self.n_r = as_vector([0,0,1])

        # placeholder objects for flame matrix
        self._D_ij = None
        self._D_ij_adj = None
        self._D = None
        self._D_adj = None

    @property
    def matrix(self):
        return self._D
    @property
    def submatrices(self):
        return self._D_ij
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def adjoint_submatrices(self):
        return self._D_ij_adj
    
    # method for finding non-zero indices and values of the matrix
    def indices_and_values(self, form):

        temp = assemble_vector(form)
        temp.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        temp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        packed = temp.array
        packed.real[abs(packed.real) < self.tol] = 0.0
        packed.imag[abs(packed.imag) < self.tol] = 0.0

        indices = np.array(np.flatnonzero(packed),dtype=np.int32)
        global_indices = self.dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, packed[indices]))
        return packed

    @staticmethod
    # prepare data for constructing sparse matrix
    def get_sparse_matrix_data(left, right, problem_type='direct'):
        if problem_type=='direct':
            row = [item[0] for item in left]
            col = [item[0] for item in right]

            row_vals = [item[1] for item in left]
            col_vals = [item[1] for item in right]
        
        elif problem_type=='adjoint':
            row = [item[0] for item in right]
            col = [item[0] for item in left]

            row_vals = [item[1] for item in right]
            col_vals = [item[1] for item in left]

        product = np.outer(row_vals,col_vals)
        val = product.flatten()

        return row, col, val
    
    # multiply matrix with Flame Transfer Function (FTF) - last step of matrix assembly
    def assemble_matrix(self, omega, problem_type):

        if problem_type == 'direct':
            self._D = self._D_ij * self.FTF(omega) 
            logging.debug("- Direct matrix D is assembling...")
       
        elif problem_type == 'adjoint':
            self._D_adj = self._D_ij_adj * np.conj(self.FTF(np.conj(omega)))
            logging.debug("- Adjoint matrix D is assembling...")
        else:
            ValueError("The problem type should be specified as 'direct' or 'adjoint'.")
        
        logging.debug("- Matrix D is assembled.")
    
    # assemble the derivative of the matrix
    def get_derivative(self, omega):
        logging.debug("- Assembling derivative of matrix D..")
        logging.debug("- FTF derivative: %s", self.FTF.derivative(omega))
        dD_domega = self.FTF.derivative(omega) * self._D_ij
        logging.debug("- Derivative of matrix D is assembled.")
        return dD_domega


# assembly of flame matrix over entire domain
class DistributedFlameMatrix(FlameMatrix):

    def __init__(self, mesh, w, h, rho, T, q_0, u_b, FTF, degree=1, bloch_object=None, gamma=None, tol=1e-5):
        super().__init__(mesh, h, q_0, u_b, FTF, degree, bloch_object, tol)

        if gamma==None: # variable gamma depends on temperature
            gamma = gamma_function(T) 

        # assemble the flame matrix symbolically
        # split into left and right side because it simplifies separation of adjoint and direct calculation
        self.left_form = form((gamma - 1) * q_0 / u_b * self.phi_i * h *  dx)
        self.right_form = form(inner(self.n_r,grad(self.phi_j)) / rho * w * dx)
    
    # assembly vectors and enable splitting computation between different processes
    def _assemble_vectors(self, problem_type='direct'):
       
        left_vector = self.indices_and_values(self.left_form)
        right_vector = self.indices_and_values(self.right_form)

        # invert left and right vector for adjoint and direct problem
        if problem_type == 'direct':
            left_vector = distribute_vector_as_chunks(left_vector)
            right_vector = broadcast_vector(right_vector)
        elif problem_type == 'adjoint':
            right_vector = distribute_vector_as_chunks(right_vector)
            left_vector = broadcast_vector(left_vector)
        else:
            logging.error("The problem type should be specified as 'direct' or 'adjoint'.")

        return left_vector, right_vector

    # assemble final matrix by combining left and right form
    def assemble_submatrices(self, problem_type='direct'):

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(self.local_size, self.global_size), (self.local_size, self.global_size)])
        mat.setType('mpiaij')

        left, right = self._assemble_vectors(problem_type=problem_type)
        row,col,val = self.get_sparse_matrix_data(left, right, problem_type=problem_type)

        logging.debug("- Generating matrix D..")

        ONNZ = len(col)*np.ones(self.local_size,dtype=np.int32)
        mat.setPreallocationNNZ([ONNZ, ONNZ])
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        logging.debug("- Distributed Submatrix D is Assembled.")

        # choose if left of right submatrix is assembled
        if problem_type == 'direct':
            self._D_ij = mat
        elif problem_type == 'adjoint':
            self._D_ij_adj = mat
        else:
            logging.error("The problem type should be specified as 'direct' or 'adjoint'.")