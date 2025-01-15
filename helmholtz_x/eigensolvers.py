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

#### EPS: eigenvalue problem solver 
# - just for standard eigenvalue problem (simple)
def eps_solver(A, C, target, nev, two_sided=False, print_results=False):

    E = SLEPc.EPS().create(MPI.COMM_WORLD)

    C = - C
    E.setOperators(A, C)


    # spectral transformation
    st = E.getST()
    st.setType('sinvert') # shiftinvert
    # E.setKrylovSchurPartitions(1) # MPI.COMM_WORLD.Get_size()
    #print("target right before square", target)
    #eps_target = target**2
    #eps_target = target.real**2 + target.imag**2 *1j # MIGHT THIS BE THE PROBLEM??
    eps_target = target
    #print("---------------------target in eps_solver", eps_target) # why is target purely imaginary???? because if abs(Re)=abs(Im)
    E.setTarget(eps_target)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # TARGET_REAL or TARGET_IMAGINARY
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

#### PEP : polynomial eigenvalue problem
#  - for more complex eigenvalue problem of polynomial degree (complex)
def pep_solver(A, B, C, target, nev, print_results=False):
    """
    This function defines solved instance for
    A + wB + w^2 C = 0

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Matrix of Grad term
    B : petsc4py.PETSc.Mat
        Empty Matrix
    C : petsc4py.PETSc.Mat
        Matrix of w^2 term.
    target : float
        targeted eigenvalue
    nev : int
        Requested number of eigenvalue
    print_results : boolean, optional
        Prints the results. The default is False.

    Returns
    -------
    Q : slepc4py.SLEPc.PEP
        Solution instance of eigenvalue problem.

    """
    #print("target in pep_solver", target)

    Q = SLEPc.PEP().create(MPI.COMM_WORLD)
    operators = [A, B, C]
    Q.setOperators(operators)

    # spectral transformation
    st = Q.getST()
    st.setType('sinvert')

    Q.setTarget(target)
    # find eigenvalues closest to a given target
    Q.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)  #MAGNITUDE, TARGET_REAL or TARGET_IMAGINARY
    Q.setDimensions(nev, SLEPc.DECIDE)
    Q.setTolerances(1e-15)
    Q.setFromOptions()

    info("- PEP solver started.")

    Q.solve()

    info("- PEP solver converged. Eigenvalue computed.")

    if print_results and MPI.COMM_WORLD.rank == 0:
        results(Q)

    return Q

## solution algorithm for EPS
def fixed_point_iteration_eps(operators, D, target, nev, i,
                              tol, maxiter,
                              print_results=False,
                              problem_type='direct',
                              two_sided=False):

    A = operators.A
    B = operators.B
    #B = np.zeros(A.size, dtype=complex) ?? try to zero B
    C = operators.C
    if problem_type == 'adjoint':
        B = operators.B_adj

    print("- searching for",nev,"eigenvalues")
    print("- using the",i,"th eigenvalue to update FTF")
    print("- doing max",maxiter,"iterations")
    print("- with a tolerance of",tol)
    
    # calculate possible start eigenvalues
    omega = np.zeros(maxiter, dtype=complex)
    f = np.zeros(maxiter, dtype=complex)
    alpha = np.zeros(maxiter, dtype=complex)

    #info("--> Fixed point iteration started.")
    # print("-------------------dimensions of A:", A.shape)
    # print("-------------------dimensions of C:", C.shape)
    # solve starting configuration without flame matrix
    E = eps_solver(A, C, target, nev, print_results=print_results)
    eig = E.getEigenvalue(i) # ---------------------------START EIGENVALUE
    #print("------------eigenvalue ", E.getEigenvalue(3))
    for iter in range(nev):
        print_eig = E.getEigenpair(iter)
        rounded = round(print_eig.real,2) + round(print_eig.imag,2)*1j
        print(f"- Start Eigenvalue {iter}: {rounded}")

    # print(i)
    # tmp =i
    # while round(np.sqrt(abs(eig.real)),2)+ round(np.sqrt(abs(eig.imag)),2)*1j==(1+0j):
    #     tmp+=1
    #     eig = E.getEigenvalue(tmp)
    #print("- eigenvalue ", round(eig.real,2)+ round(eig.imag,2)*1j)
    omega[0] = np.sqrt(eig)
    #print("- root of eig:", round(omega[0].real,2)+ round(omega[0].imag,2)*1j)
    alpha[0] = 0.5
    domega = 2 * tol
    k = - 1

    # formatting
    s = "{:.0e}".format(tol)
    s = int(s[-2:])
    s = "{{:+.{}f}}".format(s)

    # starting eigenvalue
    if MPI.COMM_WORLD.rank == 0:
        print("+ \033[1mTake START eigenvalue\033[0m: {}  {}j. ".format(
                 round(omega[k + 1].real,4), round(omega[k + 1].imag,4)))

    # if eigenvalue 0+1j is detected, stop iteration to prevent endless fixed point diverging
    if round(omega[k + 1].real)==0 and abs(round(omega[k+1].imag))==1:
        print("CONVERGED-STOP")
        #eig = E.getEigenvalue(i+1)
        domega = 0
    
    info("-> Iterations starting.")
    while abs(domega) > tol:

        k += 1
        E.destroy()
        if MPI.COMM_WORLD.rank == 0:
            print("* iter = {:2d}".format(k+1))

        # Assembly of flame matrix - update of the FTF(omega) matrix
        D.assemble_matrix(omega[k], problem_type) # where flame matrix is put together

        if problem_type == 'direct':
            D_Mat = D.matrix
        elif problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix
        else:
            raise ValueError("The problem type should be specified as 'direct' or 'adjoint'.")

        if not B:
            D_Mat = A - D_Mat
        else:
            D_Mat = A + (omega[k] * B) - D_Mat

        # viewer = PETSc.Viewer().createASCII('FlameMatrixD.txt', mode=PETSc.Viewer.Mode.WRITE)
        # viewer(D_Mat)
        # SOLVE
        E = eps_solver(D_Mat, C, target, nev, two_sided=two_sided, print_results=print_results)
        
        del D_Mat
        eig = E.getEigenvalue(i)
        # print possible eigenvalues near the target
        for iter2 in range(nev):
            print_eig = E.getEigenpair(iter2)
            rounded = round(np.sqrt(abs(print_eig.real)),2)+ round(np.sqrt(abs(print_eig.imag)),2)*1j
            print(f"- Eigenvalue {iter2}: {rounded}")

        #print("- eigenvalue:", round(eig.real,2)+ round(eig.imag,2)*1j)
        f[k] = np.sqrt(eig)
        #print("- root of eig:", round(f[k].real,2)+ round(f[k].imag,2)*1j)

        if k != 0:
            alpha[k] = 1/(1 - ((f[k] - f[k-1])/(omega[k] - omega[k-1])))
            
        omega[k+1] = alpha[k] * f[k] + (1 - alpha[k]) * omega[k]

        domega = omega[k+1] - omega[k]
        if MPI.COMM_WORLD.rank == 0:
            print('+ omega = {}  {}j,  |domega| = {:.2e}'.format(
                 round(omega[k + 1].real,4), round(omega[k + 1].imag,4), abs(domega)
            ))
        #print("alpha: ", alpha)
    return E

## solution algorithm for PEP
def fixed_point_iteration_pep( operators, D,  target, nev, i,
                                    tol, maxiter,
                                    print_results=False,
                                    problem_type='direct'):
    # import matrices
    A = operators.A
    C = operators.C
    B = operators.B
    if problem_type == 'adjoint':
        B = operators.B_adj

    # display info
    print("- searching for",nev,"eigenvalues")
    print("- using the",i,"th eigenvalue to update FTF")
    print("- doing max",maxiter,"iterations")
    print("- with a tolerance of",tol)

    # initalize empty lists
    omega = np.zeros(maxiter, dtype=complex) # empty list of eigenvalues with size of max iterations
    f = np.zeros(maxiter, dtype=complex)
    alpha = np.zeros(maxiter, dtype=complex)

    # calculate the homogeneous solution without a flame matrix
    E = pep_solver(A, B, C, target, nev, print_results=print_results)    # possible start eigenvalues
    vr, vi = A.getVecs() # storage for real and imaginary parts of the eigenvector obtained from matrix A
    #print("vectors:", vr, vi)
    print("eig converged:", E.getConverged())

    # print possible found eigenvalues near the target
    for iter in range(nev):
        print_eig = E.getEigenpair(iter) # get eigenvalue
        #rounded = round(np.sqrt(abs(print_eig.real))/2/np.pi,2)+ round(np.sqrt(abs(print_eig.imag))/2/np.pi,2)*1j
        #rounded = round(print_eig.real/2/np.pi,2)+ round(print_eig.imag/2/np.pi,2)*1j
        rounded = round(print_eig.real,2) + round(print_eig.imag,2)*1j
        #rounded = round(print_eig.real,2)+ round(print_eig.imag,2)*1j
        print(f"- Start Eigenvalue {iter}: {rounded}")

    # pick a starting eigenvalue
    # note: (i, vr, vi) is the same as just (i)
    eig = E.getEigenpair(i, vr, vi) # why only take first eigenvalue? i=0 eig is always 0+0j
    #print("-------------EIGPPAIR:", eig, "i:", i)
    omega[0] = eig # start with the first eigenvalue
    
    # update difference calculation
    alpha[0] = 0.5 # step in direction of the eigenvalue
    domega = 2 * tol
    k = - 1 # start from omega[0] in the list because first k is -1+1=0

    # print starting eigenvalue
    if MPI.COMM_WORLD.rank == 0:
        print("+ \033[1mTake START Eigenvalue\033[0m: {}  {}j. ".format(
                 round(omega[k + 1].real,4), round(omega[k + 1].imag,4)))
        
    # formatting tolerance
    s = "{:.0e}".format(tol)
    s = int(s[-2:])
    s = "{{:+.{}f}}".format(s)

    # if eigenvalue 0+1j is detected, stop iteration to prevent endless fixed point diverging
    if round(omega[k + 1].real)==0 and abs(round(omega[k+1].imag))==1:
        print("CONVERGED-STOP")
        #eig = E.getEigenvalue(i+1)
        domega = 0

    info("-> Iterations starting.")
    while abs(domega) > tol: # iterate till tolerance is met

        k += 1
        E.destroy() # empty solution matrix
        if MPI.COMM_WORLD.rank == 0:
            print("* iter = {:2d}".format(k+1)) # print current iteration

        # Assembly of flame matrix - update of the FTF(omega) matrix
        D.assemble_matrix(omega[k], problem_type)
        # specify if direct or adjoint
        if problem_type == 'direct':
            D_Mat = D.matrix
        elif problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix
        else:
            raise ValueError("The problem type should be specified as 'direct' or 'adjoint'.")

        # update matrices of power omega^0 by subtracting the flame matrix current matrix
        D_Mat = A - D_Mat
        #D_Mat = A - 0.9* D_Mat # test decreasing influence of flame matrix so only slight variation from passive case

        # SOLVE new matrix system
        E = pep_solver(D_Mat, B, C, target, nev, print_results=print_results)
        #print("E: ", E)
        D_Mat.destroy()
        eig = E.getEigenpair(i, vr, vi) # why only take first eigenvalue? i=0 eig is always 0+0j

        # print possible found eigenvalues near the target
        for iter2 in range(nev):
            print_eig = E.getEigenpair(iter2)
            #rounded = round(np.sqrt(abs(print_eig.real))/2/np.pi,2)+ round(np.sqrt(abs(print_eig.imag))/2/np.pi,2)*1j
            #rounded = round(print_eig.real/2/np.pi,2)+ round(print_eig.imag/2/np.pi,2)*1j
            #rounded = round(print_eig.real,2)+ round(print_eig.imag,2)*1j
            rounded = round(print_eig.real,2)+ round(print_eig.imag,2)*1j
            print(f"- Eigenvalue {iter2}: {rounded}")
        
        # what does this algorithm do?
        f[k] = eig
        if k != 0:
            alpha[k] = 1 / (1 - ((f[k] - f[k-1]) / (omega[k] - omega[k-1])))
        # append new eigenvalue to list
        omega[k+1] = alpha[k] * f[k] + (1 - alpha[k]) * omega[k]

        # update the residual
        domega = omega[k+1] - omega[k]
        # print the current eigenvalue
        if MPI.COMM_WORLD.rank == 0:
            print('+ omega = {}  {}j,  |domega| = {:.2e}'.format(
                 round(omega[k + 1].real,4), round(omega[k + 1].imag,4), abs(domega) #round(omega[k + 1].real,4)
            ))

    #print("List of omegas:")
    #print(np.size(omega)) # there are many zeros in omega, because only a couple iterations are required to converge 
    return E

# decide which solution algorithm is required depending on PEP or EPS problem
def fixed_point_iteration(operators, D,  target, nev, i,
                                    tol=1e-3, maxiter=50,
                                    print_results=False,
                                    problem_type='direct'):
    #operators.B.destroy() # test to delete the B matrix and force eps solver
    if operators.B:
        E = fixed_point_iteration_pep( operators, D,  target, nev=nev, i=i,
                                    tol=tol, maxiter=maxiter,
                                    print_results=print_results,
                                    problem_type=problem_type)
        #print ("---Running PEP fixed point")
    else:
        E = fixed_point_iteration_eps( operators, D,  target, nev=nev, i=i,
                                    tol=tol, maxiter=maxiter,
                                    print_results=print_results,
                                    problem_type=problem_type)
        #print("---Running EPS fixed point")
    
    return E



# another solver, other than the two fixed point iterations
def newtonSolver(operators, D, init, nev, i, tol, degree, maxiter, print_results=False):
    """
    The convergence strongly depends/relies on the initial value assigned to omega.
    Targeting zero in the shift-and-invert (spectral) transformation or, more in general,
    seeking for the eigenvalues nearest to zero might also be problematic.
    The implementation uses the TwoSided option to compute the adjoint eigenvector.
    """
    A = operators.A
    C = operators.C
    B = operators.B

    #real_converge_route = []
    #imag_converge_route = []

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
        D.assemble_matrix(omega[k])
        if not B:
            L = A + omega[k] ** 2 * C - D.matrix
            #print("- no boundary matrix")
            dL_domega = 2 * omega[k] * C - D.get_derivative(omega[k])
        else:
            L = A + omega[k] * B + omega[k]** 2 * C  - D.matrix
            dL_domega = B + (2 * omega[k] * C) - D.get_derivative(omega[k])

        # solve the eigenvalue problem L(\omega) * p = \lambda * C * p
        # set the target to zero (shift-and-invert)
        E = eps_solver(L, - C, 0, nev, two_sided=True, print_results=print_results)
        eig = E.getEigenvalue(i)
        # print("eig", eig)   
        # normalize the eigenvectors
        # note that "p" is either direct or adjoint, depending on which matrix D was assembled earlier
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
            #real_converge_route.append(omega[k + 1].real/2/np.pi)
            #imag_converge_route.append(omega[k + 1].imag/2/np.pi)

        k += 1

        del E

    return omega[k], p #, real_converge_route, imag_converge_route