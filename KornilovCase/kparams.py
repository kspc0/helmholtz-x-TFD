from math import sqrt
import numpy as np
import os

# mesh parameters
mesh_resolution = 0.2e-3 # specify mesh resolution
mesh_refinement_factor = 2 # specify mesh refinement factor
length = 4e-3 # [m] length of the plenum
height = 2.5e-3 # [m] height of the plenum

# eigenvalue problem parameters
degree = 2 # degree of FEM polynomials
frequ = -10000 # [Hz] where to expect first mode (helmholtz requires negative frequency)

perturbation = 0.0001 # [m] perturbation distance
# set boundary conditions
boundary_conditions =  {1:  {'Neumann'}, # inlet
                        2:  {'Dirichlet'}, # outlet
                        3:  {'Neumann'}, # upper combustion chamber and slit wall
                        4:  {'Neumann'}, # lower symmetry boundary
                        5:  {'Neumann'}} # upper plenum wall

# physical constants and variables
r_gas = 287.1  # [J/kg/K] ideal gas constant
gamma = 1.4  # [/] ratio of specific heat capacities cp/cv
p_amb = 1e5  # [Pa] ambient pressure
T_amb = 293 # [K] ambient temperature
rho_amb = p_amb/(r_gas*T_amb)  # [kg/m^3] ambient density
c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s] ambient speed of sound
# input/output parameters of system
rho_u = rho_amb  # [kg/m^3] input density
rho_d = 0.17 # [kg/m^3] output densitys
T_in = p_amb/(r_gas*rho_u)  # [K] input temperature
T_out = p_amb/(r_gas*rho_d)  # [K] output temperature
c_in = sqrt(gamma*p_amb/rho_u)  # [m/s] input speed of sound
c_out = sqrt(gamma*p_amb/rho_d)  # [m/s] output speed of sound
# No reflection coefficients for boundaries
R_in = 0  # [/]
R_out = 0  # [/] 

# Flame transfer function
u_b = 0.4 # [m/s] mean flow velocity/ bulk velocity
q_0 = 2577.16 # [W] heat flux density: integrated value dQ from open foam
# set path to read FTF state space matrices
path = os.path.dirname(os.path.abspath(__file__))
# load the state-space matrices from data saved as csv tables
S1 = np.loadtxt(path+'/FTFMatrices/S1.csv', delimiter=',') # A
s2 = np.loadtxt(path+'/FTFMatrices/s2.csv', delimiter=',') # B
s3 = np.loadtxt(path+'/FTFMatrices/s3.csv', delimiter=',') # C
s4 = np.array([[0]]) # D
# flame positioning
x_f = np.array([8e-3, 0.0, 0.0])  # [m] heat release rate function location
a_f = 0.5e-3 # [m] thickness of flame
# reference point coordinates
x_r = np.array([3e-3, 0., 0.])  # [m] measurement function location
a_r = 0.5e-3 # [m] thickness of reference