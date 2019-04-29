#!/usr/bin/env python
# coding: utf-8

# # Ising model

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint, choice, random
from IPython.display import clear_output, display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Calculate the energy for a spin by summing over the neighbour spins
# grid: grid of spins
# x, y: coordinates
# N:    number of spins in each dimension
# J:    degree of magnetization
# mu:   external magnetic field strength
def calculate_energy(grid, x, y, N, J, mu):
    return -J * grid[x, y] * (grid[x, (y+1)%N] +
                              grid[x, (y-1)%N] +
                              grid[(x+1)%N, y] +
                              grid[(x-1)%N, y]) - mu * grid[x, y]

# Calculate the total energy of the system
# grid: grid of spins
# J:    degree of magnetization
def calculate_total_energy(grid, J, mu):
    N = grid.shape[0]
    E = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            E[x, y] = calculate_energy(grid, x, y, N, J, mu)

    return E
    
# Simulate the Ising Model using the Monte Carlo method
# temperature:     temperature of the system
# eq_time:         equilibrium time of the system
# num_time_steps:  number of "time" steps
# N:               number of spins in each dimension
def simulate(temperature, eq_time, num_time_steps, N=16, J=1.0, mu=0.0):
    grid = choice([-1.0, 1.0], (N, N))
    magnetization, energy = np.zeros((num_time_steps - eq_time, N, N)), np.zeros((num_time_steps - eq_time, N, N))

    for t in range(num_time_steps):
        x = randint(0, N)
        y = randint(0, N)
        E = calculate_energy(grid, x, y, N, J, mu)

        if E > 0 or random() < np.exp(2.0 * E / temperature):
            grid[x, y] = grid[x, y] * -1
        
        if t >= eq_time:
            magnetization[t-eq_time, :, :] = grid
            energy[t-eq_time, :, :] = calculate_total_energy(grid, J, mu)
            
    return magnetization, energy


# In[7]:


temps = np.arange(1.6, 4.0, 0.05)       # Temperature range
results = np.zeros((len(temps), 4, 2))
time_steps = 10**5
equilibrium_time = int(n_T * 2 / 3)
N = 100                                 # Number of spins
mu = 0.0                                # External magnetic field strength

for i, T in enumerate(temps):
    clear_output(wait=True)
    print("{} of {}".format(i+1, len(temps)))
    
    # Run simulation
    simulation = simulate(T, equilibrium_time, time_steps, mu=mu)
    # Save magnetization and energy
    magnetizations = np.average(np.average(simulation[0], axis=1), axis=1)
    energies = np.average(np.average(simulation[1], axis=1), axis=1)
    # Save statistics
    results[i, 0, 0] = np.abs(np.average(magnetizations))
    results[i, 0, 1] = np.std(np.abs(magnetizations))
    results[i, 1, 0] = np.average(energies)
    results[i, 1, 1] = np.std(energies)
    
    spec_heat = np.zeros(N)
    susceptibility = np.zeros(N)
    
    # Resample the energies to calculate the specific heat for each timestep
    for n in range(N):
        random_indices = np.random.sample(len(energies)) * len(energies)
        E = energies[random_indices.astype(int)]
        spec_heat[n] = (np.average(E**2.0) - np.average(E)**2.0) / T**2.0
    
    # Resample the data of the magnetizations to calculate the susceptibility for each timestep
    for n in range(N):
        random_indices = np.random.sample(len(magnetizations)) * len(magnetizations)
        M = magnetizations[random_indices.astype(int)]
        susceptibility[n] = (np.average(M**2.0) - np.average(M)**2.0) / T
    
    # Save statistics
    results[i, 2, 0] = np.average(spec_heat)
    results[i, 2, 1] = np.sqrt(np.average(spec_heat ** 2.0) - np.average(spec_heat) ** 2.0)
    results[i, 3, 0] = np.average(susceptibility)
    results[i, 3, 1] = np.sqrt(np.average(susceptibility ** 2.0) - np.average(susceptibility) ** 2.0)


# In[7]:


# Plot observables
plt.title('Magnetization as a function of temperature')
plt.errorbar(temps, results[:, 0, 0], yerr=results[:, 0, 1], marker='o', capsize=2.5, color='b', ecolor='b')
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.grid()

plt.figure()
plt.title('Energy as a function of temperature')
plt.errorbar(temps, results[:, 1, 0], yerr=results[:, 1, 1], marker='o', capsize=2.5, color='b', ecolor='b')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.grid()

plt.figure()
plt.title('Specific heat as a function of temperature')
plt.errorbar(temps, results[:, 2, 0], yerr=results[:, 2, 1], marker='o', capsize=2.5, color='b', ecolor='b')
plt.xlabel('Temperature')
plt.ylabel('Specific heat')
plt.grid()

plt.figure()
plt.title('Susceptibility as a function of temperature')
plt.errorbar(temps, results[:, 3, 0], yerr=results[:, 3, 1], marker='o', capsize=2.5, color='b', ecolor='b')
plt.xlabel('Temperature')
plt.ylabel('Susceptibility $\chi$')
plt.grid()

plt.show()


# In[5]:


# Plot map
grid = simulate(1.5, 10**6-1, 10**6, N=128)[0][-1, :, :]
print(grid.shape)
plt.figure(figsize=(18, 10))
plt.imshow(grid, cmap='Greys_r')
plt.colorbar()

