# List of functions used for Argon Project

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

# Basic functions
def norm(r, axis=0):
    '''Calculate the magnitude/norm of a vector r'''
    return np.sqrt(np.sum(r**2., axis))

def dist(i, j):
    '''Distance (magnitude) between particles i and j'''
    x = j[0]- i[0]
    y = j[1] - i[1]
    z = j[2] - i[2]
    distance = np.sqrt(x**2. + y**2. + z**2.)
    return distance

# Simulation related
def apply_boundary_conditions(x, L):
    '''apply boundary conditions for a box of size
    L x L x L to position x'''
    for d in range(3):
        if (x[d] < 0.0):
            x[d] = L - x[d]
        elif (x[d] > L):
            x[d] = x[d] - L
    return x

def get_r(state, t, particle):
    '''Determine the closest particle for a given particle at a given time (t) and
       return the distance between these two particles'''
    xi = state[particle, t, :3]
    xj = np.delete(state[:, t, :3], particle, axis=0) # exclude the particle itself
    #closest_part = (norm((xi - xj + L/2) % L - L/2, axis = 1)).argmin()
    closest_part = (norm(xi - xj, axis = 1)).argmin()
    closest_part_pos = xj[closest_part]
    return xi - closest_part_pos

# Kinematics & mechanics
def dUdr(r):
    '''Derivatice of Lennard-Jones potential wrt r
    in dimensionless units, no sigma or epislon
    function of magnitude distance r'''
    return -48./(r**13.) + 24./(r**7.)

def acceleration(r):
    '''Force between particles a distance r apart
    r is magnitude (dimensionless)'''
    nablaU = dUdr(norm(r)) * r/norm(r)
    return -nablaU

def current_velocity(x_next, x_prev, h):
    ''' Current veolicty of a particle
    next position x_next from function at time t+h
    previous position x_prev at time t-h'''
    v = (x_next - x_prev)/(2.*h)
    return v

def next_position(x, v, force, h, L):
    '''The next position value
    current position x at time t
    current velocity v
    force from acceleration at time t
    dimensionless units'''
    x_next = x + h*v + (h**2./2. * force)
    return apply_boundary_conditions(x_next, L)

def next_velocity(v, force, force_next, h):
    '''Finding the next velocity value
    current velocity at time t
    current force at time t
    next force at time t+h
    dimensionless units'''
    v_next = v + h/2.*(force + force_next)
    return v_next

def KE(v):
    '''Kinetive energy from velocity vector'''
    return 0.5 * v**2.

def LJP(r):
    '''Lennard-Jones potential formula
    Describes the potential of the system given a distance between
    two particles in dimensionless units
    function of magnitude distance r'''
    return 4.*(1./(r**12.) - 1./(r**6))

# Observables
def pair_correlation(n, r, Deltar, L, N):
    '''Pair correlation function for particle n
    at distance r (array), bin size Deltar,
    in simulation box of size L,
    and total number of particles N'''
    V = L**3.
    mean_n = np.mean(n)
    g = (2*V)/(N*(N-1)) * mean_n/(4*np.pi * r**2. * Deltar)
    return g
