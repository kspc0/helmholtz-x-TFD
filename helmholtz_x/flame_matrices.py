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

# assembly of the flame matrix D(omega)
class FlameMatrix:
    def  __init__(self, mesh, h, q_0, u_b, FTF, degree, bloch_object=None, tol=1e-5):

        self.mesh = mesh
        self.h = h
        self.q_0 = q_0 
        self.u_b = u_b
        self.FTF = FTF
        self.degree = degree
        self.bloch_object=bloch_object
        self.tol = tol
        
        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        # definition of trial and test function
        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)
        self.gdim = self.mesh.geometry.dim

        # Matrix utils
        self.global_size = self.V.dofmap.index_map.size_global
        self.local_size = self.V.dofmap.index_map.size_local

        # Vector for reference direction
        # why is n_r defined like this? why these directions?
        if self.gdim == 1:
            self.n_r = as_vector([1])
        elif self.gdim == 2:
            self.n_r = as_vector([1,0])
        else:
            self.n_r = as_vector([0,0,1])

        # Utility objects for flame matrix
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
    
    # what is this?
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
    # first create a quite empty matrix?
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
    
    # multiply matrix with Flame Transfer Function (FTF)
    def assemble_matrix(self, omega, problem_type):

        if problem_type == 'direct':
            self._D = self._D_ij*self.FTF(omega) 
            info("- Direct matrix D is assembling...")
       
        elif problem_type == 'adjoint':
            self._D_adj = self._D_ij_adj * np.conj(self.FTF(np.conj(omega)))
            info("- Adjoint matrix D is assembling...")
        else:
            ValueError("The problem type should be specified as 'direct' or 'adjoint'.")
        
        info("- Matrix D is assembled.")
    
    def get_derivative(self, omega):
        info("- Assembling derivative of matrix D..")
        print("- FTF derivative:", self.FTF.derivative(omega))
        dD_domega = self.FTF.derivative(omega) * self._D_ij
        info("- Derivative of matrix D is assembled.")
        return dD_domega

    # is this even used?
    def blochify(self, problem_type='direct'):

        if problem_type == 'direct':
            D_ij_bloch = self.bloch_object.blochify(self.submatrices)
            self._D_ij = D_ij_bloch

        elif problem_type == 'adjoint':
            D_ij_adj_bloch = self.bloch_object.blochify(self.adjoint_submatrices)
            self._D_ij_adj = D_ij_adj_bloch
        else:
            ValueError("The problem type should be specified as 'direct' or 'adjoint'.")
    


### TYPE 1: pointwise (not used)
class PointwiseFlameMatrix(FlameMatrix):

    def __init__(self, mesh, subdomains, x_r, h, rho_u, q_0, u_b, FTF, degree=1, bloch_object=None, gamma=1.4, tol=1e-10):

        super().__init__(mesh, h, q_0, u_b, FTF, degree, bloch_object, tol)
        self.x_r = x_r
        self.rho_u = rho_u
        self.gamma = gamma
        self.dx = Measure("dx", subdomain_data=subdomains)

    def _assemble_vectors(self, flame, point):
        # Assemble the flame matrix D(omega) if pointwise
        left_form = form((self.gamma - 1) * self.q_0 / self.u_b * inner(self.h, self.phi_j)*self.dx(flame))
        left_vector = self.indices_and_values(left_form)

        # he?
        _, _, owning_points, cell = determine_point_ownership( self.mesh._cpp_object, point)#, 1e-10) manually changed because gives ERROR
        right_vector = []

        if len(cell) > 0: # Only add contribution if cell is owned 
            cell_geometry = self.mesh.geometry.x[self.mesh.geometry.dofmap[cell[0]], :self.gdim]
            point_ref = self.mesh.geometry.cmaps[0].pull_back([point], cell_geometry)
            right_form = Expression(inner(grad(TestFunction(self.V)), self.n_r), point_ref, comm=MPI.COMM_SELF)
            dphij_x_rs = right_form.eval(self.mesh, cell)[0]           
            right_values = dphij_x_rs / self.rho_u
            global_dofs = self.dofmaps.index_map.local_to_global(self.dofmaps.cell_dofs(cell[0]))
            for global_dof, right_value in zip(global_dofs, right_values):
                right_vector.append([global_dof, right_value ])

        right_vector = broadcast_vector(right_vector)

        return left_vector, right_vector

    # why are submatrices necessary?
    def assemble_submatrices(self, problem_type='direct'):

        info("- Generating matrix D..")

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(self.local_size, self.global_size), (self.local_size, self.global_size)])
        mat.setType('aij') 
        mat.setUp()

        for flame, point in enumerate(self.x_r):
            
            left, right = self._assemble_vectors(flame, point)
            row,col,val = self.get_sparse_matrix_data(left, right, problem_type=problem_type)

            mat.setValues(row,col,val, addv=PETSc.InsertMode.ADD_VALUES)
            info("- Matrix contribution of flame "+str(flame)+" is computed.")

        mat.assemblyBegin()
        mat.assemblyEnd()

        info ("- Pointwise Submatrix D is Assembled.")

        if problem_type == 'direct':
            self._D_ij = mat
        elif problem_type == 'adjoint':
            self._D_ij_adj = mat
        else:
            ValueError("The problem type should be specified as 'direct' or 'adjoint'.")


### TYPE 2: distributed
class DistributedFlameMatrix(FlameMatrix):

    def __init__(self, mesh, w, h, rho, T, q_0, u_b, FTF, degree=1, bloch_object=None, gamma=None, tol=1e-5):
        super().__init__(mesh, h, q_0, u_b, FTF, degree, bloch_object, tol)

        if gamma==None: # Variable gamma depends on temperature
            gamma = gamma_function(T) 

        # Assemble the flame matrix symbolically
        self.left_form = form((gamma - 1) * q_0 / u_b * self.phi_i * h *  dx)
        self.right_form = form(inner(self.n_r,grad(self.phi_j)) / rho * w * dx)
    
    def _assemble_vectors(self, problem_type='direct'):
       
        left_vector = self.indices_and_values(self.left_form)
        right_vector = self.indices_and_values(self.right_form)

        if problem_type == 'direct':
            left_vector = distribute_vector_as_chunks(left_vector)
            right_vector = broadcast_vector(right_vector)
        elif problem_type == 'adjoint':
            right_vector = distribute_vector_as_chunks(right_vector)
            left_vector = broadcast_vector(left_vector)
        else:
            ValueError("The problem type should be specified as 'direct' or 'adjoint'.")

        return left_vector, right_vector

    # why is this necessary?
    def assemble_submatrices(self, problem_type='direct'):

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(self.local_size, self.global_size), (self.local_size, self.global_size)])
        mat.setType('mpiaij')

        left, right = self._assemble_vectors(problem_type=problem_type)
        row,col,val = self.get_sparse_matrix_data(left, right, problem_type=problem_type)

        info("- Generating matrix D..")

        ONNZ = len(col)*np.ones(self.local_size,dtype=np.int32)
        mat.setPreallocationNNZ([ONNZ, ONNZ])
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        info ("- Distributed Submatrix D is Assembled.")

        if problem_type == 'direct':
            self._D_ij = mat
        elif problem_type == 'adjoint':
            self._D_ij_adj = mat
        else:
            ValueError("The problem type should be specified as 'direct' or 'adjoint'.")