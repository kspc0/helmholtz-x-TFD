from .petsc4py_utils import vector_matrix_vector
from .eigenvectors import normalize_eigenvector
from .solver_utils import info
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import scipy

def results(E):
    if MPI.COMM_WORLD.Get_rank()==0:
        print()
        print("******************************")
        print("*** SLEPc Solution Results ***")
        print("******************************")
        print()

        its = E.getIterationNumber()
        print("Number of iterations of the method: %d" % its)

        eps_type = E.getType()
        print("Solution method: %s" % eps_type)

        nev, ncv, mpd = E.getDimensions()
        print("Number of requested eigenvalues: %d" % nev)

        tol, maxit = E.getTolerances()
        print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

        nconv = E.getConverged()
        print("Number of converged eigenpairs %d" % nconv)

        A = E.getOperators()[0]
        vr, vi = A.createVecs()

        if nconv > 0:
            print()
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            print("%15f, %15f" % (k.real, k.imag))
        print()

# EPS: eigenvalue problem solver 
# for standard eigenvalue problem
def eps_solver(A, C, target, nev, two_sided=False, print_results=False):
    """
    This function defines solved instance for
    A + w^2 C = 0
    """
    E = SLEPc.EPS().create(MPI.COMM_WORLD)

    C = - C
    E.setOperators(A, C)

    # spectral transformation
    st = E.getST()
    st.setType('sinvert') # shiftinvert
    eps_target = target
    E.setTarget(eps_target)
    # find eigenvalues closest to a given target
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # MAGNITUDE, TARGET_REAL or TARGET_IMAGINARY
    E.setTwoSided(two_sided)

    E.setDimensions(nev, SLEPc.DECIDE)
    E.setTolerances(1e-15)
    E.setFromOptions()

    info("- EPS solver started.")
    E.solve()
    info("- EPS solver converged. Eigenvalue computed.")
    if print_results and MPI.COMM_WORLD.rank == 0:
        results(E)

    return E

# PEP : polynomial eigenvalue problem
# eigenvalue problem of polynomial degree (complex)
def pep_solver(A, B, C, target, nev, print_results=False):
    """
    This function defines solved instance for
    A + wB + w^2 C = 0
    """
    Q = SLEPc.PEP().create(MPI.COMM_WORLD)
    operators = [A, B, C]
    Q.setOperators(operators)

    # spectral transformation
    st = Q.getST()
    st.setType('sinvert')

    Q.setTarget(target)
    # find eigenvalues closest to a given target
    Q.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)  # MAGNITUDE, TARGET_REAL or TARGET_IMAGINARY
    Q.setDimensions(nev, SLEPc.DECIDE)
    Q.setTolerances(1e-15)
    Q.setFromOptions()

    info("- PEP solver started.")
    Q.solve()
    info("- PEP solver converged. Eigenvalue computed.")
    if print_results and MPI.COMM_WORLD.rank == 0:
        results(Q)

    return Q


# Newton solver for eigenvalue problem
def newtonSolver(operators, degree, D, init, nev, i, tol, maxiter, problem_type, print_results=False):
    """
    The convergence strongly depends/relies on the initial value assigned to omega.
    Targeting zero in the shift-and-invert (spectral) transformation or, more in general,
    seeking for the eigenvalues nearest to zero might also be problematic.
    Parameters:
    # operators: A, C, B acoustic matrices
    # degree: degree of the function space
    # D: flame matrix D
    # init: initial target for the eigenvalue
    # nev: number of eigenvalues to find in close range to target
    # i: index of the eigenvalue (i=0 is closest eigenvalue to target)
    # tol: tolerance of the solution
    # maxiter: maximum number of iterations

    The implementation uses the TwoSided option to compute the adjoint eigenvector.
    """
    A = operators.A
    C = operators.C
    B = operators.B

    omega = np.zeros(maxiter, dtype=complex)
    omega[0] = init
    domega = 2 * tol
    k = 0

    # formatting
    tol_ = "{:.0e}".format(tol)
    tol_ = int(tol_[-2:])
    s = "{{:+.{}f}}".format(tol_)

    relaxation = 1.0

    info("-> Newton solver started.")

    while abs(domega) > tol:
        D.assemble_matrix(omega[k], problem_type)
        # choose which type of problem to solve - direct or adjoint
        if problem_type == 'direct':
            D_Mat = D.matrix
        elif problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix
        # choose if boundary matrix is included or not
        if not B:
            L = A + omega[k] ** 2 * C - D_Mat
            print("- no boundary matrix")
            dL_domega = 2 * omega[k] * C - D.get_derivative(omega[k])
        else:
            L = A + omega[k] * B + omega[k] ** 2 * C  - D_Mat
            dL_domega = B + (2 * omega[k] * C) - D.get_derivative(omega[k])
            print("- boundary matrix included")

        # solve the eigenvalue problem L(\omega) * p = \lambda * C * p
        # usually we solve L(\omega) * p = \lambda*Id*p, but here we set matrix C, to give the convergence a headstart
        # mass matrix C defines the scale of the problem, which makes it easier to converge
        E = eps_solver(L, - C, 0, nev, two_sided=True, print_results=print_results) # why never use pep solver?
        eig = E.getEigenvalue(i)
        # normalize the eigenvectors
        # p is either direct or adjoint eigenvector, depending on which matrix D was assembled earlier
        omega_dir, p = normalize_eigenvector(operators.mesh, E, i, degree=degree, which='right', print_eigs=False)
        # however a second ajoint "left" eigenvector is required to calculate the convergence of the eigenvalue
        omega_adj, p_adj = normalize_eigenvector(operators.mesh, E, i, degree=degree, which='left', print_eigs=False)

        # convert into PETSc.Vec type
        p_vec = p.vector
        p_adj_vec = p_adj.vector

        # numerator and denominator
        num = vector_matrix_vector(p_adj_vec, dL_domega, p_vec)
        den = vector_matrix_vector(p_adj_vec, C, p_vec)

        deig = num / den
        domega = - relaxation * eig / deig
        relaxation *= 0.8

        omega[k + 1] = omega[k] + domega
        
        if MPI.COMM_WORLD.rank == 0:
            print('iter = {:2d},  omega = {}  {}j,  |domega| = {:.2e}'.format(
                    k, s.format(omega[k + 1].real), s.format(omega[k + 1].imag), abs(domega)))
        k += 1
        del E

    return omega[k], p