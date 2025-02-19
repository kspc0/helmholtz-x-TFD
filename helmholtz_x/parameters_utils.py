from dolfinx.fem import FunctionSpace, Function, form, Constant, assemble_scalar, locate_dofs_topological
from .dolfinx_utils import normalize, unroll_dofmap
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from ufl import Measure

def gamma_function(temperature):
    r_gas =  287.1
    if isinstance(temperature, Function):
        V = temperature.function_space
        cp = Function(V)
        cv = Function(V)
        gamma = Function(V)
        cp.x.array[:] = 973.60091+0.1333*temperature.x.array[:]
        cv.x.array[:] = cp.x.array - r_gas
        gamma.x.array[:] = cp.x.array/cv.x.array
        gamma.x.scatter_forward()
    else:    
        cp = 973.60091+0.1333*temperature
        cv= cp - r_gas
        gamma = cp/cv
    return gamma

def sound_speed_variable_gamma(mesh, temperature, degree=1):
    # https://www.engineeringtoolbox.com/air-speed-sound-d_603.html
    # V = FunctionSpace(mesh, ("CG", degree))
    c = Function(temperature.function_space)
    c.name = "soundspeed"
    r_gas = 287.1
    if isinstance(temperature, Function):
        gamma = gamma_function(temperature)
        c.x.array[:] = np.sqrt(gamma.x.array[:]*r_gas*temperature.x.array[:])
    else:
        gamma_ = gamma_function(temperature)
        c.x.array[:] = np.sqrt(gamma_ * r_gas * temperature)
    c.x.scatter_forward()
    return c

def sound_speed(temperature):
    # https://www.engineeringtoolbox.com/air-speed-sound-d_603.html
    c = Function(temperature.function_space)
    c.name = "soundspeed"
    if isinstance(temperature, Function):
        c.x.array[:] =  20.05 * np.sqrt(temperature.x.array)
    else:
        c.x.array[:] =  20.05 * np.sqrt(temperature)
    c.x.scatter_forward()
    return c

