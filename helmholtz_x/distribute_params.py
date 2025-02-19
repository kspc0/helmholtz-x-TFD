import numpy as np
from dolfinx.fem import Function, FunctionSpace
from helmholtz_x.dolfinx_utils import normalize


# density distribution
def tanh_function(mesh, x_f, a_f, rho_d, rho_u, degree=1):
    V = FunctionSpace(mesh, ("CG", degree)) # create function space
    rho = Function(V) # create function
    x_f = x_f[0] # get the first element
    # distribute the density function onto the cells
    rho.interpolate(lambda x: density(x[0], x_f, a_f/2, rho_d, rho_u))
    return rho
# tanh-function starting from rho_d and smoothely transitioning up to rho_u
# at coordinate x_f with thickness sigma
def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))

# heat release rate and measurement distribution
def gauss_function(mesh, x_f, a_f, degree=1):
    V = FunctionSpace(mesh, ("CG", degree)) # create function space
    w = Function(V) # create function
    x_f = x_f[0] # get the first element
    ndim = mesh.geometry.dim # get the dimension of the mesh
    # distribute the density function onto the cells
    w.interpolate(lambda x: gauss(x[0], x_f, a_f, ndim))
    # normalize so integral of domain equals 1 
    w = normalize(w)
    return w
# gauss-function with amplitude 1 and sigma as standard deviation
def gauss(x, x_ref, sigma, n):
    if n==1:
        spatial = (x-float(x_ref))**2
    elif n==2:
        spatial = (x-x_ref)**2 + (x-x_ref)**2
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))
    return amplitude*spatial_term

# homogeneous heat release rate and measurement distribution
def homogeneous_function(mesh, x_f, degree=1):
    V = FunctionSpace(mesh, ("CG", degree)) # create function space
    w = Function(V) # create function
    x_f = x_f[0] # get the first element
    w.interpolate(lambda x: np.zeros_like(x[0])) # create homogeneous function
    return w

# temperature distribution
def step_function(mesh, x_f, T_u, T_d):
    V = FunctionSpace(mesh, ("CG", 1)) # create function space
    c = Function(V) # create function
    c.name = "temperature" # define which parameter the acoustic matrices use
    x = V.tabulate_dof_coordinates()
    x_f = x_f[0] # get the first element
    axis = 0 # loop through the x values
    # loop through each degree of freedom coordinate
    for i in range(x.shape[0]):
        midpoint = x[i,:] # coordinates of current point
        ref_value = x_f # set new threshold
        # step function depending on the x coordinate threshold
        if midpoint[axis] < ref_value:
            c.vector.setValueLocal(i, T_u)
        else:
            c.vector.setValueLocal(i, T_d)
    c.x.scatter_forward() # update the function
    return c
