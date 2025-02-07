from math import sqrt
import numpy as np
import os

from dolfinx.fem import Function, FunctionSpace

from helmholtz_x.dolfinx_utils import normalize, unroll_dofmap
from helmholtz_x.parameters_utils import sound_speed_variable_gamma, sound_speed

# set path to write files
path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(path)

#--------------------------------PARAMETERS------------------------------------#
### classic physical constants and variables
r_gas = 287.1  # [J/kg/K] ideal gas constant
gamma = 1.4  # [/] ratio of specific heat capacities cp/cv

### properties of the surroundings
p_amb = 1e5  # [Pa] ambient pressure
T_amb = 293 #293  # [K] ambient temperature
rho_amb = 1.22  # [kg/m^3] ambient density
c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s] ambient speed of sound 338.7537429470791

### density, temperature and speed of sound
# input and output density
rho_u = rho_amb  # [kg/m^3] -> 1,189187904 ~1.2
rho_d = 0.85 # inhomogenous case  # [kg/m^3]
# input and output temperature
T_in = 285.6  # [K] -> 293
T_out = 409.92  # [K] -> 2049
# input and output speed of sound
c_in = sqrt(gamma*p_amb/rho_u)  # [kg/m^3] -> 338.7537429470791
c_out = sqrt(gamma*p_amb/rho_d)  # [kg/m^3] -> 405.8397249567139

# No reflection coefficients for boundaries
#R_in = -0.975-0.05j  #-0.23123+0.123j # [/]
#R_out = -0.975-0.05j   #-0.454-0.2932j  # [/] 

### Flame transfer function
u_b = 0.1006 # [m/s] mean flow velocity (bulk velocity)
# scale q_0 down from 3D cylinder to 2D plane
q_0 = -27.0089 #/(0.047)/1 # [W] heat flux density: integrated value dQ from open foam
# load the state-space matrices from data saved as csv tables
# FTF:
n = 0.1*4/(np.pi*0.047) #/(np.pi/4 * 0.047)#2.7 #2.7 = 0.1 / (np.pi * 0.047/4) # interaction index scaled for 2D case
tau = 0.0015#/1.744 #0.0015*338.7537429470791 #*4/(np.pi*0.047) #0015 #*338/0.047 #*4/(np.pi*0.047)
# 0.00086 fits perfectly for lenght = 1
# from Juniper paper - nondimensional index
#n = 0.014 / (np.pi/4 * 0.047) #0.161
#tau = 0.0015*338/1=0.508 # nondimensional time delay

d = 1e-3 # used to dimension the mesh between scale 1m and scale 1e-3m
### Flame location
# flame location in 2D
x_f = np.array([[250*d, 0.0, 0.0]])  # [m] flame located at roughly 13mm in Kornilov Case
a_f = 25*d # [m] thickness of flame
# reference point coordinates - needed?
x_r = np.array([[200*d, 0., 0.]])  # [m] reference located at 5mm in Kornilov Case
a_r = 25*d  # [m] thickness of reference
# gauss function dimensions for a side view gaussian function
sig = 80*d # [m] thickness of flame gauss function
amplitude = 4*d # [m] height of flame gauss function
limit = 1*d # [m] horizontal separation for breakpoint of density function


#--------------------------------FUNCTIONS------------------------------------#
# DENSITY FUNCTION
#  distribute functions across mesh
# create a density function
def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))
# -> its a tanh function which starts from rho_d and goes smoothely up
#    to rho_u at coordinate x_f with thickness sigma

# now distribute the 1D density function onto the 2D mesh
def rhoFunction(mesh, x_f, a_f, rho_d, rho_u,amplitude, sig,limit, degree=1):
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

# now distribute the 1D density function onto the 2D mesh
def rhoFunctionPlane(mesh, x_f, a_f, rho_d, rho_u,amplitude, sig,limit, degree=1):
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
def gaussianH(x, x_ref, sigma, n):
    if len(x_ref)==1:
        x_ref = x_ref[0] 
    if n==1:
        spatial = (x-float(x_ref))**2
    elif n==2:
        spatial = (x-x_ref)**2 + (x-x_ref)**2
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))
    return amplitude*spatial_term

# for heat release rate
# shape values like gauss function
# without checking dimension of n - used for plane gauss function
def gaussianHplane(x, x_ref, sigma, n):
    if n==1:
        spatial = (x-float(x_ref))**2
    elif n==2:
        spatial = (x-x_ref)**2 + (x-x_ref)**2
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))
    return amplitude*spatial_term

# HEAT RELEASE RATE FUNCTION
# flame shaped gauss function
def gaussianFunctionH(mesh, x_f, a_f, amplitude, sig,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    ndim = mesh.geometry.dim # comment this to 1 for Kornilov Case because its 2D and we want 1D gauss functions
    # implement the gauss shape by making parameter y dependend
    func = lambda y: amplitude*np.exp(-(y**2)/(2*sig**2))+x_f
    w.interpolate(lambda x: gaussianH(x[0],func(x[1]),a_f,ndim))
    w = normalize(w)
    return w

# plane gauss function
def gaussianFunctionHplane(mesh, x_f, a_f, amplitude, sig,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    ndim = mesh.geometry.dim # comment this to 1 for Kornilov Case because its 2D and we want 1D gauss functions
    w.interpolate(lambda x: gaussianHplane(x[0],x_f,a_f,ndim))
    w = normalize(w)
    return w

def gaussianFunctionHplaneHomogenous(mesh, x_f, a_f, amplitude, sig,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_f = x_f[0]
    x_f = x_f[0]
    ndim = mesh.geometry.dim # comment this to 1 for Kornilov Case because its 2D and we want 1D gauss functions
    w.interpolate(lambda x: np.zeros_like(x[0]))
    return w

# point function
def FunctionPoint(mesh, x_r, a_r, amplitude, sig,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    x_r = x_r[0]
    # set the point with index 3039, because that is the one which is at [x_r,0] in the gmsh file! check with paraview for property Id
    w.vector.setValueLocal(3039, 1)
    return w

# added new gaussian function to cut it along the X axis for Kornilov Case
def halfGaussianFunctionX(mesh,x_flame,a_flame,degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    h = gaussianFunction(mesh, x_flame, a_flame, degree=degree)
    if len(x_flame)==1:
        x_flame = x_flame[0]
    x_tab = V.tabulate_dof_coordinates()
    for i in range(x_tab.shape[0]):
        midpoint = x_tab[i,:]
        z = midpoint[0]
        if z<x_flame[0]:
            value = 0.
        else:
            value = h.x.array[i]
        h.vector.setValueLocal(i, value)
    h = normalize(h)
    return h

# SOUND SPEED FUNCTION
# added this for Kornilov Case
def c_step_gauss(mesh, x_f, c_u, c_d, amplitude, sig):
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

def c_step_gauss_plane(mesh, x_f, c_u, c_d, amplitude, sig):
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
        ref_value = x_f # calculate the new threshold
        #print('ref Value', ref_value)
        # Step function based on comparison with updating reference value from lambda
        if midpoint[axis] < ref_value:  # adjust axis if needed (e.g., midpoint[0] for x, midpoint[1] for y)
            c.vector.setValueLocal(i, c_u)
        else:
            c.vector.setValueLocal(i, c_d)
    c.x.scatter_forward()
    return c

# MEASUREMENT FUNCTION
# used for function w for flame matrix
def gaussianFunction(mesh, x_r, a_r, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    w = Function(V)
    ndim = mesh.geometry.dim 
    w.interpolate(lambda x: gaussian(x,x_r,a_r,ndim))
    w = normalize(w)
    return w

def gaussian(x, x_ref, sigma, n):
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

# TEMPERATURE FUNCTION
# added function for Kornilov Case
def temperature_step_gauss(mesh, x_f, T_u, T_d,amplitude, sig, degree=1):
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
        #print("Mid:",midpoint)
        #print('MidY', midpoint[1])
        func = lambda y: amplitude*np.exp(-(y**2)/(2*sig**2))+x_f # update the y coordinate to a rotated gauss function
        ref_value = func(midpoint[1]) # calculate the new threshold
        #print('ref Value', ref_value)
        # Step function based on comparison with updating reference value from lambda
        if midpoint[axis] < ref_value:  # adjust axis if needed (e.g., midpoint[0] for x, midpoint[1] for y)
            T.vector.setValueLocal(i, T_u)
        else:
            T.vector.setValueLocal(i, T_d)
    return T

def temperature_step_gauss_plane(mesh, x_f, T_u, T_d,amplitude, sig, degree=1):
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
        #print("Mid:",midpoint)
        #print('MidY', midpoint[1])
        ref_value = x_f # calculate the new threshold
        #print('ref Value', ref_value)
        # Step function based on comparison with updating reference value from lambda
        if midpoint[axis] < ref_value:  # adjust axis if needed (e.g., midpoint[0] for x, midpoint[1] for y)
            T.vector.setValueLocal(i, T_u)
        else:
            T.vector.setValueLocal(i, T_d)
    return T


#---------------------------------SAVING------------------------------------#
# only execute when the skript is run directly from terminal with python3 params.py:
# save the new input mesh functions
# this is not needed for running the code - only for postprocessing with paraview
# in general the parameters are imported by other scripts using "import params"
if __name__ == '__main__':
    from helmholtz_x.io_utils import XDMFReader,xdmf_writer
    # get mesh data from saved file
    Rijke = XDMFReader("Meshes/RijkeMesh")
    mesh, subdomains, facet_tags = Rijke.getAll()
    # create the functions with curved flames:
    # rho_func = rhoFunction(mesh, x_f, a_f, rho_d, rho_u, amplitude, sig, limit) # tanh density
    # w_func = gaussianFunction(mesh, x_r, a_r) # gauss measurement (original from HelmX)
    # h_func = gaussianFunctionH(mesh, x_f, a_f/2, amplitude, sig) # gauss heat release rate (modified to Kornilov)
    # c_func = c_step_gauss(mesh, x_f, c_in, c_out, amplitude, sig) # step sound speed (modified to Kornilov)
    # T_func = temperature_step_gauss(mesh, x_f,T_in, T_out, amplitude, sig) # step temperature (modified to Kornilov)

    # create the functions with plane flame
    h_func = gaussianFunctionHplane(mesh, x_f, a_f, amplitude, sig) # gauss heat release rate (modified to Kornilov)
    w_func = gaussianFunctionHplane(mesh, x_r, a_r, amplitude, sig) # gauss measurement (original from HelmX)
    rho_func = rhoFunctionPlane(mesh, x_f, a_f, rho_d, rho_u, amplitude, sig, limit) # tanh density
    c_func = c_step_gauss_plane(mesh, x_f, c_in, c_out, amplitude, sig) # step sound speed (modified to Kornilov)
    T_func_temp = temperature_step_gauss_plane(mesh, x_f, T_in, T_out, amplitude, sig) # step temperature (modified to Kornilov)
    T_func_rho = rhoFunctionPlane(mesh, x_f, a_f, T_out, T_in, amplitude, sig, limit) # step temperature (modified to Kornilov)
    
    # save the functions in the InputFunctions directory as .xdmf   
    xdmf_writer("InputFunctions/rho", mesh, rho_func)
    xdmf_writer("InputFunctions/w", mesh, w_func)
    xdmf_writer("InputFunctions/h", mesh, h_func)
    xdmf_writer("InputFunctions/c", mesh, c_func)
    xdmf_writer("InputFunctions/T_temp", mesh, T_func_temp)
    xdmf_writer("InputFunctions/T_rho", mesh, T_func_rho)

    # testing what T is doing in acoustic matrices:
    c_self_acoustic = sound_speed_variable_gamma(mesh, T_func_temp, degree=1)
    xdmf_writer("InputFunctions/C_acoustic", mesh, c_self_acoustic)

