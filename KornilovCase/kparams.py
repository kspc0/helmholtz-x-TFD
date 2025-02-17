from math import sqrt
import numpy as np
import os

from dolfinx.fem import Function, FunctionSpace
import gmsh
from helmholtz_x.dolfinx_utils import normalize, unroll_dofmap
from helmholtz_x.parameters_utils import sound_speed
from helmholtz_x.shape_derivatives import ShapeDerivativesFFDRectFullBorder, ffd_displacement_vector_rect_full_border # to calculate shape derivatives

# set path to write files
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

#--------------------------------PARAMETERS------------------------------------#
### classic physical constants and variables
r_gas = 287.1  # [J/kg/K] ideal gas constant
gamma = 1.4  # [/] ratio of specific heat capacities cp/cv

### properties of the surroundings
p_amb = 1e5  # [Pa] ambient pressure
T_amb = 293  # [K] ambient temperature
rho_amb = p_amb/(r_gas*T_amb)  # [kg/m^3] ambient density
c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s] ambient speed of sound

### density, temperature and speed of sound
# input and output density
rho_u = rho_amb  # [kg/m^3]
rho_d = 0.17  # [kg/m^3]
# input and output temperature
T_in = p_amb/(r_gas*rho_u)  # [K]
T_out = p_amb/(r_gas*rho_d)  # [K]
# input and output speed of sound
c_in = sqrt(gamma*p_amb/rho_u)  # [kg/m^3]
c_out = sqrt(gamma*p_amb/rho_d)  # [kg/m^3]

# No reflection coefficients for boundaries
R_in = 0  # [/]
R_out = 0  # [/] 

### Flame transfer function
u_b = 0.4 # [m/s] mean flow velocity/ bulk velocity
q_0 = 2577.16 # [W] heat flux density: integrated value dQ from open foam
# load the state-space matrices from data saved as csv tables
S1 = np.loadtxt(parent_path+'/FTFMatrices/S1.csv', delimiter=',') # A
s2 = np.loadtxt(parent_path+'/FTFMatrices/s2.csv', delimiter=',') # b
s3 = np.loadtxt(parent_path+'/FTFMatrices/s3.csv', delimiter=',') # c
s4 = np.array([[0]]) # d

d = 1e-3 # used to dimension the mesh between scale 1m and scale 1e-3m
### Flame location
# flame location in 2D
x_f = np.array([[13*d, 0.0, 0.0]])  # [m] flame located at roughly 13mm
a_f = 0.5*d # [m] thickness of flame
# reference point coordinates - needed?
x_r = np.array([[5*d, 0., 0.]])  # [m] reference located at 5mm
a_r = 0.5*d  # [m] thickness of reference
# gauss function dimensions
sig = 0.8*d # [m] thickness of curved flame gauss function
amplitude = 4*d # [m] height of curved flame gauss function
limit = 1*d # [m] horizontal separation for breakpoint of curved density function


#--------------------------------FUNCTIONS------------------------------------#
# tanh function for density 
# starts from rho_d and goes smoothely up to rho_u at coordinate x_f with thickness sigma
def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))
# now distribute the 1D density function onto the 2D mesh
def curved_tanh_function(mesh, x_f, a_f, rho_d, rho_u,amplitude, sig,limit, degree=1):
    # create a continuous function space V
    V = FunctionSpace(mesh, ("CG", degree))
    rho = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    # implement the gauss shape by making parameter y dependend
    func = lambda y: np.where(y > limit, 9*x_f/10, -3 * y + 2.7*x_f/(2))  #1/(7e-3*y-40e-3)+2.2e-3-0.2e-3*y ##
    # other functions: #-3.5 *y+x_f+x_f/2 #amplitude*np.exp(-(y**2)/(2*sig**2))+x_f # update the y coordinate to a rotated gauss function
    # distribute the function onto the cells
    rho.interpolate(lambda x: density(x[0], func(x[1]), a_f/2, rho_d, rho_u))
    return rho
# now distribute the 1D density function onto the 2D
def plane_tanh_function(mesh, x_f, a_f, rho_d, rho_u, degree=1):
    # create a continuous function space V
    V = FunctionSpace(mesh, ("CG", degree))
    rho = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    # distribute the function onto the cells
    rho.interpolate(lambda x: density(x[0], x_f, a_f/2, rho_d, rho_u))
    return rho

# for heat release rate
# shape values like gauss function
def curved_gauss(x, x_ref, sigma, n):
    if len(x_ref)==1:
        x_ref = x_ref[0] 
    if n==1:
        spatial = (x-float(x_ref))**2
    elif n==2:
        spatial = (x-x_ref)**2 + (x-x_ref)**2
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))
    return amplitude*spatial_term
# flame shaped gauss function
def curved_gauss_function(mesh, x_f, a_f, amplitude, sig,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    ndim = mesh.geometry.dim # comment this to 1 for Kornilov Case because its 2D and we want 1D gauss functions
    # implement the gauss shape by making parameter y dependend
    func = lambda y: amplitude*np.exp(-(y**2)/(2*sig**2))+x_f
    w.interpolate(lambda x: curved_gauss(x[0],func(x[1]),a_f,ndim))
    w = normalize(w)
    return w
# for measurement function
def point_gauss(x, x_ref, sigma, n):
    if len(x_ref)==1:
        x_ref = x_ref[0]
    if   n==1:
        spatial = (x[0]-float(x_ref[0]))**2
    elif n==2:
        spatial = (x[0]-x_ref[0])**2 + (x[1]-x_ref[1])**2
    elif n==3:
        spatial = (x[0]-x_ref[0])**2 + (x[1]-x_ref[1])**2 + (x[2]-x_ref[2])**2
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))
    return amplitude*spatial_term
# used for function w for flame matrix
def point_gauss_function(mesh, x_r, a_r, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    ndim = mesh.geometry.dim 
    w.interpolate(lambda x: point_gauss(x,x_r,a_r,ndim))
    w = normalize(w)
    return w

# for heat release rate and measurement function
# shape values like gauss function
# without checking dimension of n - used for plane gauss function
def plane_gauss(x, x_ref, sigma, n):
    if n==1:
        spatial = (x-float(x_ref))**2
    elif n==2:
        spatial = (x-x_ref)**2 + (x-x_ref)**2
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))
    return amplitude*spatial_term
# plane gauss function gaussianFunctionHplane
def plane_gauss_function(mesh, x_f, a_f, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    ndim = mesh.geometry.dim # comment this to 1 for Kornilov Case because its 2D and we want 1D gauss functions
    w.interpolate(lambda x: plane_gauss(x[0],x_f,a_f,ndim))
    w = normalize(w)
    return w
# for homogeneous case no need for gauss functions - just zero
def homogeneous_flame_functions(mesh, x_f,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    w.interpolate(lambda x: np.zeros_like(x[0]))
    return w

# temperature and sound speed functions
def curved_step_function(mesh, x_f, c_u, c_d, amplitude, sig):
    V = FunctionSpace(mesh, ("CG", 1))
    c = Function(V)
    c.name = "soundspeed"
    x = V.tabulate_dof_coordinates()
    x_f = x_f[0]
    x_f = x_f[0]
    axis = 0 # loop through the x values
    # loop through each degree of freedom coordinate
    for i in range(x.shape[0]):
        midpoint = x[i,:] # coordinates of current point
        #print("Mid:",midpoint)
        #print('MidY', midpoint[1])
        func = lambda y: amplitude*np.exp(-(y**2)/(2*sig**2))+x_f # update the y coordinate to a rotated gauss function
        ref_value = func(midpoint[1]) # calculate the new threshold
        #print('ref Value', ref_value)
        # Step function based on comparison with updating reference value from lambda
        if midpoint[axis] < ref_value:  # adjust axis if needed (e.g., midpoint[0] for x, midpoint[1] for y)
            c.vector.setValueLocal(i, c_u)
        else:
            c.vector.setValueLocal(i, c_d)
    c.x.scatter_forward()
    return c
def plane_step_function(mesh, x_f, T_u, T_d):
    V = FunctionSpace(mesh, ("CG", 1))
    c = Function(V)
    c.name = "temperature" # important! -> defines which parameter the acoustic matrices use!
    x = V.tabulate_dof_coordinates()
    x_f = x_f[0]
    x_f = x_f[0]
    axis = 0 # loop through the x values
    # loop through each degree of freedom coordinate
    for i in range(x.shape[0]):
        midpoint = x[i,:] # coordinates of current point
        ref_value = x_f # calculate the new threshold
        #print('ref Value', ref_value)
        # Step function based on comparison with updating reference value from lambda
        if midpoint[axis] < ref_value:  # adjust axis if needed (e.g., midpoint[0] for x, midpoint[1] for y)
            c.vector.setValueLocal(i, T_u)
        else:
            c.vector.setValueLocal(i, T_d)
    c.x.scatter_forward()
    return c


## MOVED INTO CLASS
#---------------------------------SAVING------------------------------------#
# only execute when the skript is run directly from terminal with python3 params.py:
# if __name__ == '__main__':
#     from helmholtz_x.io_utils import XDMFReader,xdmf_writer
#     # get mesh data from saved file
#     gmsh.initialize() # start the gmsh session
#     Kornilov = XDMFReader("Meshes/Kornilov")
#     mesh, subdomains, facet_tags = Kornilov.getAll()

#     # create CURVED flame functions
#     # rho_func = curved_tanh_function(mesh, x_f, a_f, rho_d, rho_u, amplitude, sig, limit)
#     # w_func = point_gauss_function(mesh, x_r, a_r)
#     # h_func = curved_gauss_function(mesh, x_f, a_f/2, amplitude, sig) 
#     # c_func = curved_step_function(mesh, x_f, c_in, c_out, amplitude, sig)
#     # T_func = curved_step_function(mesh, x_f,T_in, T_out, amplitude, sig)
#     # c_acoustic = sound_speed(T_func)

#     # create PLANE flame functions
#     h_func = plane_gauss_function(mesh, x_f, a_f)
#     w_func = plane_gauss_function(mesh, x_r, a_r)
#     rho_func = plane_tanh_function(mesh, x_f, a_f, rho_d, rho_u)
#     c_func = plane_step_function(mesh, x_f, c_in, c_out)
#     T_func = plane_step_function(mesh, x_f, T_in, T_out)
#     c_acoustic = sound_speed(T_func)

#     #V_ffd = ffd_displacement_vector_rect_full_border(Kornilov, 5, [0,1], deg=1)
#     # save the functions in the InputFunctions directory as .xdmf   
#     xdmf_writer("InputFunctions/rho", mesh, rho_func)
#     xdmf_writer("InputFunctions/w", mesh, w_func)
#     xdmf_writer("InputFunctions/h", mesh, h_func)
#     xdmf_writer("InputFunctions/c", mesh, c_func)
#     xdmf_writer("InputFunctions/T", mesh, T_func)
#     xdmf_writer("InputFunctions/c_acoustic", mesh, c_acoustic)
#     #xdmf_writer(path+"/InputFunctions/V_ffd", mesh, V_ffd)

