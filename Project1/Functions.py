# List of functions used for Argon Project

import numpy as np
import matplotlib.pyplot as plt

# Basic functions
def norm(r, axis=0):
    '''Calculate the magnitude/norm of a vector r'''
    return np.sqrt(np.sum(r**2.0, axis))

def dist(i, j):
    '''Distance (magnitude) between particles i and j'''
    return norm(i - j)

def create_random_vector():
    '''Calculate a unitary vector in random direction'''
    vec = np.random.uniform(-1.0, 1.0, (3))
    return vec / norm(vec)

# Simulation related
def apply_boundary_conditions(x, L):
    '''apply boundary conditions for a box of size
    L x L x L to position x'''
    return x % L

def get_force(state, t, particle, L):
    '''Determine the closest particle for a given particle at a given time (t) and
       return the distance between these two particles'''
    xi = state[particle, t, :3]
    indices = np.arange(0, state.shape[0])
    indices = np.delete(indices, particle) # exclude the particle itself
    neighbours = state[indices, t, :3]
    distances = np.zeros(len(neighbours))
    F = np.zeros(3)
    P = 0
    
    # find for each particle the closest mirror
#     for j, xj in enumerate(neighbours):
#         r = (xi - xj + L / 2.0) % L - L / 2.0
#         r_norm = norm(r)
#         F += acceleration(r, r_norm)
#         P += LJP(r_norm)
#         distances[j] = r_norm
    
    # find for each particle the closest mirror
    r = (xi - neighbours + L / 2.0) % L - L / 2.0
    r_norm = norm(r, axis=1)
    r_norm_reshaped = np.reshape(np.repeat(r_norm, 3), (len(r_norm), 3))
    F = np.sum(acceleration(r, r_norm_reshaped), axis=0)
    P = np.sum(LJP(r_norm))
    distances = r_norm
    
    return F, P, distances

def rescale_velocities(state, t):
    lambda_factor = np.sqrt((num_part - 1) * 3.0 * T / np.sum(state[:, t, 3:] ** 2, axis=0))
    state[:, t, 3:] *= lambda_factor
    return state

# Kinematics & mechanics
def dUdr(r):
    '''Derivatice of Lennard-Jones potential wrt r
    in dimensionless units, no sigma or epislon
    function of magnitude distance r'''
    return -48./(r**13.) + 24./(r**7.)

def acceleration(r, r_norm):
    '''Force between particles a distance r apart
    r is magnitude (dimensionless)'''
    nablaU = dUdr(r_norm) * r / r_norm
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
    return v + h/2.0 * (force + force_next)

def KE(v):
    '''Kinetic energy from velocity vector'''
    return 0.5 * v**2.

def LJP(r):
    '''Lennard-Jones potential formula
    Describes the potential of the system given a distance between
    two particles in dimensionless units
    function of magnitude distance r'''
    return 4.*(1./(r**12.) - 1./(r**6))

# Observables
def pair_correlation(n, r, dr, L, N):
    '''Pair correlation function for n particles
    at distance r (array), bin size dr,
    in simulation box of size L and total number of particles N'''
    V = L**3.0
    g = (2.0*V)/(N*(N-1.0)) * n/(4.0 * np.pi * r**2.0 * dr)
    return g

def get_lambda(state, t, num_part, T):
    return np.sqrt((num_part - 1) * 3.0 * T / np.sum(norm(state[:, t, 3:], axis=1) ** 2.0))
