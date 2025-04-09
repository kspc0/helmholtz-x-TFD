from math import sqrt
import numpy as np
import os

# mesh parameters 
mesh_resolution = 0.008 # specify mesh resolution (standard 0.008)
length = 1 # [m] length of the tube
height = 0.047 # [m] height of the tube

# eigenvalue problem parameters
degree = 2 # degree of FEM polynomials
frequ = 170 # [Hz] where to expect first mode ! initial value is negative the frequency

perturbation = 0.001 # [m] perturbation distance
# set boundary conditions
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Neumann'}, # outlet
                        3:  {'Neumann'}, # upper wall
                        4:  {'Neumann'}} # lower wall

# physical constants and variables
r_gas = 287.1  # [J/kg/K] ideal gas constant
gamma = 1.4  # [/] ratio of specific heat capacities cp/cv
p_amb = 1e5  # [Pa] ambient pressure
T_amb = 293 # [K] ambient temperature
rho_amb = 1.22  # [kg/m^3] ambient density
c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s] ambient speed of sound
# input/output parameters of system
rho_u = rho_amb  # [kg/m^3] input density
rho_d = 0.85 # [kg/m^3] output densitys
T_in = 285.6  # [K] input temperature
T_out = 409.92  # [K] output temperature
c_in = sqrt(gamma*p_amb/rho_u)  # [m/s] input speed of sound
c_out = sqrt(gamma*p_amb/rho_d)  # [m/s] output speed of sound
# No reflection coefficients for boundaries
R_in = 0  # [/]
R_out = 0  # [/] 

# Flame transfer function
u_b = 0.1006 # [m/s] mean flow velocity/ bulk velocity
# scale q_0 down from 3D cylinder to 2D plane
q_0 = -27.0089*4/(np.pi*0.047) # [W] heat flux density
# n-tau model parameters
n = 0.1 # interaction index
tau = 0.0015 # [s] time delay


# flame positioning
x_f = np.array([250e-3, 0.0, 0.0])  # [m] heat release rate function location
a_f = 25e-3 # [m] thickness of flame
# reference point coordinates
x_r = np.array([200e-3, 0., 0.])  # [m] measurement function location
a_r = 25e-3 # [m] thickness of reference