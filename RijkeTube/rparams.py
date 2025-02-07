from math import sqrt
import numpy as np
import os

from dolfinx.fem import Function, FunctionSpace
from helmholtz_x.dolfinx_utils import normalize
from helmholtz_x.parameters_utils import sound_speed # to calculate sound speed from temperature

# set path to write files
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

#--------------------------------PARAMETERS------------------------------------#
### classic physical constants and variables
r_gas = 287.1  # [J/kg/K] ideal gas constant
gamma = 1.4  # [/] ratio of specific heat capacities cp/cv

### properties of the surroundings
p_amb = 1e5  # [Pa] ambient pressure
T_amb = 293 # [K] ambient temperature
rho_amb = 1.22  # [kg/m^3] ambient density
c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s] ambient speed of sound

### density, temperature and speed of sound
# input and output density
rho_u = rho_amb  # [kg/m^3]
rho_d = 0.85 # inhomogenous case  # [kg/m^3]
# input and output temperature
T_in = 285.6  # [K]
T_out = 409.92  # [K]
# input and output speed of sound
c_in = sqrt(gamma*p_amb/rho_u)  # [m/s]
c_out = sqrt(gamma*p_amb/rho_d)  # [m/s]

# reflection coefficients for boundaries if Robin boundary given
#R_in = -0.975-0.05j # [/]
#R_out = -0.975-0.05j # [/] 

### Flame transfer function
u_b = 0.1006 # [m/s] mean flow velocity (bulk velocity)
# scale q_0 down from 3D cylinder to 2D plane
q_0 = -27.0089*4/(np.pi*0.047) #/(0.047)/1 # [W] heat flux density: integrated value dQ from open foam
# n-tau model parameters
n = 0.1 # interaction index
tau = 0.0015 #086 # 0.0015
# dimensioning of the flame
d = 1e-3 # used to scale the mesh between meters and milimeters
### Flame location
# flame location - heat release rate
x_f = np.array([[250*d, 0.0, 0.0]])  # [m]
a_f = 25*d # [m] thickness of flame
# reference point coordinates
x_r = np.array([[200*d, 0., 0.]])  # [m]
a_r = 25*d  # [m] thickness of reference



#-------------------------DISTRIBUTE FUNCTIONS ON MESH-------------------------#
# DENSITY FUNCTION
# tanh function for density 
# starts from rho_d and goes smoothely up to rho_u at coordinate x_f with thickness sigma
def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))
# now distribute the 1D density function onto the 2D mesh
def rho_function(mesh, x_f, a_f, rho_d, rho_u, degree=1):
    # create a continuous function space V
    V = FunctionSpace(mesh, ("CG", degree))
    rho = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    # distribute the function onto the cells
    rho.interpolate(lambda x: density(x[0], x_f, a_f/2, rho_d, rho_u))
    return rho

# FLAME FUNCTIONS h and w
# for heat release rate used for plane gauss function
def gaussian_plane(x, x_ref, sigma, n):
    if n==1:
        spatial = (x-float(x_ref))**2
    elif n==2:
        spatial = (x-x_ref)**2 + (x-x_ref)**2
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))
    return amplitude*spatial_term
# plane gauss function
def flame_functions(mesh, x_f, a_f, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    ndim = mesh.geometry.dim
    w.interpolate(lambda x: gaussian_plane(x[0],x_f,a_f,ndim))
    w = normalize(w)
    return w
# for homogeneous case no need for gauss function - just zero
def homogeneous_flame_functions(mesh, x_f,x_r,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    ndim = mesh.geometry.dim
    w.interpolate(lambda x: np.zeros_like(x[0]))
    return w

# TEMPERATURE FUNCTION
def temperature_step_function(mesh, x_f, T_u, T_d, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    T = Function(V)
    T.name = "temperature"
    x = V.tabulate_dof_coordinates()
    x_f = x_f[0]
    x_f = x_f[0]
    axis = 0 # loop through the x values
    # loop through each degree of freedom coordinate
    for i in range(x.shape[0]):
        midpoint = x[i,:] # coordinates of current point
        ref_value = x_f # calculate the new threshold
        # step comparison with updating reference value from lambda
        # adjust axis if needed (e.g., midpoint[0] for x, midpoint[1] for y)
        if midpoint[axis] < ref_value:
            T.vector.setValueLocal(i, T_u)
        else:
            T.vector.setValueLocal(i, T_d)
    return T


#---------------------------------SAVING------------------------------------#
# only execute when the skript is run directly from terminal with python3 rparams.py:
if __name__ == '__main__':
    from helmholtz_x.io_utils import XDMFReader,xdmf_writer
    # get mesh data from saved file
    Rijke = XDMFReader("Meshes/RijkeMesh")
    mesh, subdomains, facet_tags = Rijke.getAll()
    # create the functions with plane flame shape
    h_func = flame_functions(mesh, x_f, a_f)
    w_func = flame_functions(mesh, x_r, a_r)
    rho_func = rho_function(mesh, x_f, a_f, rho_d, rho_u)
    T_func_temp = temperature_step_function(mesh, x_f, T_in, T_out)
    c_func = sound_speed(T_func_temp)
    # save the functions in the InputFunctions directory as .xdmf files used to examine with paraview  
    xdmf_writer("InputFunctions/rho", mesh, rho_func)
    xdmf_writer("InputFunctions/w", mesh, w_func)
    xdmf_writer("InputFunctions/h", mesh, h_func)
    xdmf_writer("InputFunctions/c", mesh, c_func)
    xdmf_writer("InputFunctions/T", mesh, T_func_temp)
