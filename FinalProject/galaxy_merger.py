'''
Rachel Losacco
Erik Vroon
Computatinal Physics 2019
Final Project: Computational Astrophysics
Simulating the merger of the Milky Way and Andromeda
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
##matplotlib.use("Agg")

from amuse.units import units, constants
from amuse.lab import nbody_system
from amuse.ext.galactics_model import new_galactics_model
from amuse.lab import Gadget2
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits

def create_two_galaxies(M_MW, M_And, R_MW, R_And,
                        nhalo_MW, nhalo_And, nbulge_MW, nbulge_And,
                        ndisk_MW, ndisk_And):
    '''
    Output two galaxies with halo, disk, and bulge, and
    mass, position, and velocity
    For the Milky Way (MW) and Andromeda (And) include
    mass M, radius R, number n of particles in the halo 'nhalo,' bulge
    'nbulge,' and disk 'ndisk'
    '''
    # Make a converter to define units of the system
    converter = nbody_system.nbody_to_si(M_MW, R_MW)
    # Establish galaxy models
    MW_galaxy = new_galactics_model(nhalo_MW, converter,
                                   bulge_number_of_particles = nbulge_MW,
                                   disk_number_of_particles = ndisk_MW)
    And_galaxy = new_galactics_model(nhalo_And, converter,
                                     bulge_number_of_particles = nbulge_And,
                                     disk_number_of_particles = ndisk_And)
    # Include other parameters
    MW_galaxy.mass = M_MW
    And_galaxy.mass = M_And

    return MW_galaxy, And_galaxy

def simulate_merger(MW_galaxy, And_galaxy, nhalo_MW, nhalo_And,
                    M_MW, R_MW, time_step, t_end):
    '''
    Simulate the merger of MW_galaxy and And_galaxy until t_end
    These galaxies have a halo particle number nhalo_[galaxy]
    '''
    # Define units of the system
    converter_generic = ConvertBetweenGenericAndSiUnits(constants.c, units.s)
    converter = nbody_system.nbody_to_si(M_MW, R_MW)
    # Establish hydrodynamics solver
    hydro = Gadget2(converter)
    hydro.parameters.time_max = 2*converter_generic.to_generic(t_end)
    # Add galaxies to solver
    Milky_Way = hydro.particles.add_particles(MW_galaxy)
    Andromeda = hydro.particles.add_particles(And_galaxy)
    hydro.particles.move_to_center()

    # Channel the solver back to its particles
    hydro_channel_to_MW = hydro.particles.new_channel_to(MW_galaxy)
    hydro_channel_to_And = hydro.particles.new_channel_to(And_galaxy)

    # Separate the disk from the rest of the galaxy
    # We don't want to plot the halo
    MW_disk = Milky_Way[:nhalo_MW]
    And_disk = Andromeda[:nhalo_And]

    # Run simulation
    hydro.timestep = time_step
    time = 0.0 | units.Myr

    x_MW = []
    y_MW = []
    x_And = []
    y_And = []
    t = []
    while time < t_end:
        t.append(time)
        time += time_step
        # Evolve the simulation by one time step
        hydro.evolve_model(time)

        # Call channel
        hydro_channel_to_MW.copy()
        hydro_channel_to_And.copy()

        # Record observables
        x_MW.append(MW_disk.x.value_in(units.kpc))
        y_MW.append(MW_disk.y.value_in(units.kpc))
        x_And.append(And_disk.x.value_in(units.kpc))
        y_And.append(And_disk.y.value_in(units.kpc))

    hydro.stop()
    return x_MW, y_MW, x_And, y_And

if __name__ == '__main__':
    M_MW = 10**12. | units.MSun
    M_And = 10**12. | units.MSun
    R_MW = 30.660 | units.kpc
    R_And = 33.726 | units.kpc
    nhalo_MW = 2000
    nhalo_And = 2000
    nbulge_MW = 1000
    nbulge_And = 1000
    ndisk_MW = 1000
    ndisk_And = 1000
    time_step = 1 | units.Myr
    t_end = 10 | units.Myr

    MW_galaxy, And_galaxy = create_two_galaxies(M_MW, M_And, R_MW, R_And,
                        nhalo_MW, nhalo_And, nbulge_MW, nbulge_And,
                        ndisk_MW, ndisk_And)
    x_MW, y_MW, x_And, y_And = simulate_merger(MW_galaxy, And_galaxy, nhalo_MW, nhalo_And,
                    M_MW, R_MW, time_step, t_end)
    np.savetxt('out_x_MW.dat', x_MW)
    np.savetxt('out_y_MW.dat', y_MW)
    np.savetxt('out_x_And.dat', x_And)
    np.savetxt('out_y_And.dat', y_And)
