from .petsc4py_utils import multiply, vector_matrix_vector, matrix_vector, FixSign
from dolfinx.fem import Function, FunctionSpace, Expression, form
from ufl import dx, VectorElement, grad, inner, sqrt
from dolfinx.fem.assemble import assemble_scalar
from .solver_utils import info
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

# normalize EV
# print eigenvalues
def normalize_eigenvector(mesh, obj, i, absolute=False, degree=1, which='right',BlochRemapper=None, matrices=None, print_eigs=True):
    """ 
    This function normalizes the eigensolution vr which is obtained from complex slepc build
     (vi is set to zero in complex build of slepc) 

    Returns:
        [<class 'dolfinx.fem.function.Function'>]: normalized eigensolution such that \int (p p dx) = 1
    """

    A = obj.getOperators()[0]
    # create vectors to store the eigenvector
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

    # (bloch condition not used)
    if BlochRemapper:
        vr = matrix_vector(BlochRemapper,vr)
    
    # usually matrices are not given as input parameter
    if matrices:
        V=matrices.V
    # so we need to create a function space
    else:
        V = FunctionSpace(mesh, ("CG", degree))

    # create a function to store the eigenvector
    p = Function(V)
    FixSign(vr) # ensure consistent sign of the eigenvector
    p.vector.setArray(vr.array) # set the eigenvector to the function from the matrix
    p.x.scatter_forward()
    # calculate measure of the function with sqrt(integ(p^2)dx=1)
    meas = np.sqrt(mesh.comm.allreduce(assemble_scalar(form(p*p*dx)), op=MPI.SUM))
    
    temp = vr.array
    # normalize by dividing by the measure
    temp = temp/meas
    
    # only if absolute is true, we normalize the eigenvector by the maximum value
    if absolute:
        abs_temp = abs(temp)
        max_temp = mesh.comm.allreduce(np.amax(abs_temp), op=MPI.MAX)
        temp = abs_temp/max_temp

    p_normalized = Function(V) # Required for Parallel runs
    p_normalized.vector.setArray(temp)
    p_normalized.x.scatter_forward()

    # print the eigenvalue and eigenfrequency
    if MPI.COMM_WORLD.rank == 0 and print_eigs:
        print(f"Eigenvalue = {omega:.3f} | \033[1mEigenfrequency= {omega/(2*np.pi):.3f}\033[0m")
        # add a print for squareroot of eigenvalue:
        #rounded = round(np.sqrt(abs(omega.real))/2/np.pi,2)+ round(np.sqrt(abs(omega.imag))/2/np.pi,2)*1j
        #print(f"Eigenfrequency obtained with squareroot: {rounded}\n")
    return omega, p_normalized


# calc velocity eigenfunction after solving system
def velocity_eigenvector(mesh, p, omega, rho, degree=1, normalize=True, absolute=False):
    """
    This function calculates velocity eigenfunction using momentum equation;
        -i \omega u \rho + \nabla p = 0 -> Eq (2b) in PRF paper

        mesh ([dolfinx.cpp.mesh.Mesh]): mesh of the domain
        p ([dolfinx.fem.function.Function]): acoustic pressure eigenfunction
        omega ([complex]): eigenvalue
        rho ([dolfinx.fem.function.Function]): density field
        degree (int, optional): degree of finite elements. Defaults to 1.

    Returns:
        [<class 'dolfinx.fem.function.Function'>]: normalized eigenfunction such that \int (u * u * dx) = 1
    
    """
    if mesh.topology.dim ==1:
        Q = FunctionSpace(mesh, ("CG",degree))
    else:
        v_cg = VectorElement("CG", mesh.ufl_cell(), degree)
        Q = FunctionSpace(mesh, v_cg)
   
    v_h = Function(Q)
    v_h.name = "U"

    if isinstance(rho, Function):
        velocity_expr = Expression(grad(p)/rho, Q.element.interpolation_points())
        v_h.interpolate(velocity_expr)
        v_h.x.array[:] /= (1j*omega)
    else:
        velocity_expr = Expression(grad(p), Q.element.interpolation_points())
        v_h.interpolate(velocity_expr)
        v_h.x.array[:] /= (1j*omega*rho)

    if normalize:
        meas = np.sqrt(mesh.comm.allreduce(assemble_scalar(form(inner(v_h,v_h)*dx)), op=MPI.SUM))
        v_h.x.array[:] /= meas

    if absolute:
        
        if len(v_h)==1:
            mag = v_h
        elif len(v_h)==2:
            mag = sqrt(v_h[0]**2 + v_h[1]**2)
        elif len(v_h)==3:
            mag = sqrt(v_h[0]**2 + v_h[1]**2 + v_h[2]**2)

        V = FunctionSpace(mesh,("CG", degree))
        vs_mag = Function(V)
        vs_mag_expr = Expression(mag, V.element.interpolation_points())
        vs_mag.interpolate(vs_mag_expr) 
        vs_mag.x.array[:] = np.abs(vs_mag.x.array[:])
        max_mag = mesh.comm.allreduce(np.amax(vs_mag.x.array), op=MPI.MAX) 

        v_h.x.array[:] = np.abs(v_h.x.array[:]) / max_mag    

    v_h.x.scatter_forward()
    
    return v_h



# Normalizes adjoint eigenfunction for shape optimization.
def normalize_adjoint(omega_dir, p_dir, p_adj, matrices, D=None):
    """
    Args:
        omega_dir ([complex]): direct eigenvalue
        p_dir ([<class 'dolfinx.fem.function.Function'>]): direct eigenfunction
        p_adj ([<class 'dolfinx.fem.function.Function'>]): adjoint eigenfunction
        matrices ([type]): passive_flame object
        D ([type], optional): active flame matrix

    Returns:
        [<class 'dolfinx.fem.function.Function'>]: [description]
    """
    
    info("- Normalizing the adjoint eigenvector to calculate shape derivatives..")

    B = matrices.B

    p_dir_vec = p_dir.vector
    p_adj_vec = p_adj.vector

    if not B and not D:
        # + 2 \omega C
        dL_domega = matrices.C * (2 * omega_dir)
    elif B and not D:
        # B + 2 \omega C
        dL_domega = (B +
                     matrices.C * (2 * omega_dir))
    elif D and not B:
        # 2 \omega C - D'(\omega)
        dL_domega = (matrices.C * (2 * omega_dir) -
                     D.get_derivative(omega_dir))
    else:
        # B + 2 \omega C - D'(\omega)
        dL_domega = (B +
                     matrices.C * (2 * omega_dir) -
                     D.get_derivative(omega_dir))

    # measure of the adjoint eigenfunction
    meas = vector_matrix_vector(p_adj_vec, dL_domega, p_dir_vec)

    p_adj_vec = multiply(p_adj_vec, 1 / meas)

    p_adj1 = p_adj
    p_adj1.name = "p_adj"
    p_adj1.vector.setArray(p_adj_vec.getArray())
    p_adj1.x.scatter_forward()

    integral = vector_matrix_vector(p_adj1.vector, dL_domega, p_dir_vec)
    
    if MPI.COMM_WORLD.rank == 0:
        print("! Normalization Check: ", integral)

    return p_adj1