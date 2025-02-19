from .petsc4py_utils import multiply, vector_matrix_vector, matrix_vector, FixSign
from dolfinx.fem import Function, FunctionSpace, Expression, form
from ufl import dx, VectorElement, grad, inner, sqrt
from dolfinx.fem.assemble import assemble_scalar
from .solver_utils import info
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

# normalize eigenvector with L2 norm (used in Newton solver)
def normalize_eigenvector(mesh, obj, i, absolute=False, degree=1, which='right', matrices=None, print_eigs=True):
    """ 
    normalized eigensolution such that \int (p p dx) = 1
    """

    A = obj.getOperators()[0]
    # create placeholder vectors to store the eigenvector
    vr, vi = A.createVecs()
    
    # check which type of eigenvalue problem there was PEP or EPS
    # obtain the eigenvector and eigenvalue from the matrix
    if isinstance(obj, SLEPc.EPS):
        eig = obj.getEigenvalue(i)
        omega = np.sqrt(eig)
        if which == 'right':
            obj.getEigenvector(i, vr, vi)
        elif which == 'left':
            obj.getLeftEigenvector(i, vr, vi)
    elif isinstance(obj, SLEPc.PEP):
        eig = obj.getEigenpair(i, vr, vi)
        omega = eig

    # usually matrices are not given as input parameter
    if matrices:
        V = matrices.V
    # so we need to create an empty function space
    else:
        V = FunctionSpace(mesh, ("CG", degree))

    # create a function to store the eigenvector
    p = Function(V)
    FixSign(vr) # ensure consistent sign of the eigenvector
    p.vector.setArray(vr.array) # set the function to store eigenvector
    p.x.scatter_forward() # distribute the values
    # calculate normalizing measure of the function with sqrt(integral(p^2)dx=1)
    meas = np.sqrt(mesh.comm.allreduce(assemble_scalar(form(p*p*dx)), op=MPI.SUM)) # this does not affect shape derivative because is cancelled later
    # only print normalization measure for right eigenvector
    if which == 'right':
        print("- measure of normalization for eigenvector: m = ", round(meas.real,3))

    temp = vr.array
    # normalize eigenvector by the measure
    temp = temp/meas
    
    # only if absolute is true, we normalize the eigenvector by the maximum value
    if absolute: # not used in newton solver
        abs_temp = abs(temp)
        max_temp = mesh.comm.allreduce(np.amax(abs_temp), op=MPI.MAX)
        temp = abs_temp/max_temp

    # create a function to store the normalized eigenvector
    p_normalized = Function(V) # Required for Parallel runs
    p_normalized.vector.setArray(temp)
    p_normalized.x.scatter_forward()

    # print the eigenvalue and eigenfrequency
    if MPI.COMM_WORLD.rank == 0 and print_eigs:
        print(f"Eigenvalue = {omega:.3f} | \033[1mEigenfrequency= {omega/(2*np.pi):.3f}\033[0m")
    return omega, p_normalized


# normalizes adjoint eigenvector for shape optimization with B + 2 \omega C - D'(\omega)
def normalize_adjoint(omega_dir, p_dir, p_adj, matrices, D=None):
    info("- Normalizing the adjoint eigenvector to calculate shape derivatives..")

    B = matrices.B

    p_dir_vec = p_dir.vector
    p_adj_vec = p_adj.vector

    # normalize according to derivation of shape derivative
    if not B and not D:
        # + 2 \omega C
        dL_domega = matrices.C * (2 * omega_dir)
        print("- using normalization: 2 omega C")
    elif B and not D:
        # B + 2 \omega C
        dL_domega = (B +
                     matrices.C * (2 * omega_dir))
        print("- using normalization: B + 2 omega C")
    elif D and not B:
        # 2 \omega C - D'(\omega)
        dL_domega = (matrices.C * (2 * omega_dir) -
                    D.get_derivative(omega_dir))
        print("- using normalization: 2 omega C - D'")
    else:
        # B + 2 \omega C - D'(\omega)
        dL_domega = (B +
                     matrices.C * (2 * omega_dir) -
                     D.get_derivative(omega_dir))
        print("- using normalization: B + 2 omega C - D'")

    # measure of the adjoint eigenfunction
    meas = vector_matrix_vector(p_adj_vec, dL_domega, p_dir_vec)
    print("- measure of the shape derivative normalization: m =", meas)
    p_adj_vec = multiply(p_adj_vec, 1 / meas) # normalize by division of measure

    # create new vector to store the normalized adjoint eigenvector
    p_adj1 = p_adj # ? need to add .copy() if calculate parallel two derivatives for inlet and outlet
    p_adj1.name = "p_adj"
    p_adj1.vector.setArray(p_adj_vec.getArray())
    p_adj1.x.scatter_forward()

    # check normalization
    integral = vector_matrix_vector(p_adj1.vector, dL_domega, p_dir_vec)
    if MPI.COMM_WORLD.rank == 0:
        print("! Normalization Check: ", integral)

    return p_adj1