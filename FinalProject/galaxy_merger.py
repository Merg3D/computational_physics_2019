'''
Rachel Losacco
Erik Vroon
Computatinal Physics 2019
Final Project: Computational Astrophysics
Simulating the merger of the Milky Way and Andromeda
'''
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units, constants
from amuse.units.optparse import OptionParser
from amuse.lab import nbody_system
from amuse.ext.galactics_model import new_galactics_model
from amuse.lab import Gadget2
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits

def make_plot(disk1, disk2, state):
    plt.scatter(disk1.x.value_in(units.kpc), disk1.y.value_in(units.kpc),
                color='r', s=1, label='Milky Way')
    plt.scatter(disk2.x.value_in(units.kpc), disk2.y.value_in(units.kpc),
                color='b', s=1, label='Andromeda')
    plt.xlim(-300, 300)
    plt.xlabel('$x$ [kpc]')
    plt.ylim(-300, 300)
    plt.ylabel('$y$ [kpc]')
    plt.title('Merger at '+state)
    plt.legend()
    plt.savefig('merger_plot_'+state+'.png')
    plt.close()

def create_two_galaxies(M_MW, M_And, R_MW, R_And,
                        n_halo, n_bulge, n_disk):
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
    MW_galaxy = new_galactics_model(n_halo, converter,
                                   bulge_number_of_particles = n_bulge,
                                   disk_number_of_particles = n_disk)
    And_galaxy = new_galactics_model(n_halo, converter,
                                     bulge_number_of_particles = n_bulge,
                                     disk_number_of_particles = n_disk)
    # Include other parameters
    MW_galaxy.mass = M_MW
    And_galaxy.mass = M_And
    # Offset
    And_galaxy.position = MW_galaxy.position
    MW_galaxy.position += [100.0, 100.0, 0.0] | units.kpc
    MW_galaxy.rotate(0., np.pi/2, np.pi/4)
    MW_galaxy.velocity += [-10.0, 0.0, -10.0] | units.km/units.s
    
    And_galaxy.position -= [100.0, 0.0, 0.0] | units.kpc
    And_galaxy.rotate(np.pi/4., np.pi/4., 0.0)
    And_galaxy.velocity -= [0.0, 0.0, 0.0] | units.km/units.s

    return MW_galaxy, And_galaxy

def simulate_merger(MW_galaxy, And_galaxy, n_halo,
                    time_step, t_end):
    '''
    Simulate the merger of MW_galaxy and And_galaxy until t_end
    These galaxies have a halo particle number nhalo_[galaxy]
    '''
    # Define units of the system
    converter = nbody_system.nbody_to_si(1.0e12|units.MSun, 100|units.kpc)
    # Establish hydrodynamics solver
    hydro = Gadget2(converter)
    hydro.parameters.epsilon_squared = (100 | units.parsec)**2
    # Add galaxies to solver
    Milky_Way = hydro.particles.add_particles(MW_galaxy)
    Andromeda = hydro.particles.add_particles(And_galaxy)
    hydro.particles.move_to_center()

    # Separate the disk from the rest of the galaxy
    # We don't want to plot the halo
    MW_disk = Milky_Way[:n_halo]
    And_disk = Andromeda[:n_halo]

##    # Quick run
##    make_plot(MW_disk, And_disk, 'start')
##    hydro.evolve_model(t_end)
##    make_plot(MW_disk, And_disk, 'end')
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

        # Record observables
        x_MW.append(MW_disk.x.value_in(units.kpc))
        y_MW.append(MW_disk.y.value_in(units.kpc))
        x_And.append(And_disk.x.value_in(units.kpc))
        y_And.append(And_disk.y.value_in(units.kpc))

    hydro.stop()
##    make_plot(MW_disk, And_disk, 'end')

    return x_MW, y_MW, x_And, y_And

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("--MMW", unit=units.MSun,
                      dest="M_MW", default = 1.0e12 | units.MSun,
                      help="Milky Way mass [%default]")
    result.add_option("--RMW", unit=units.kpc,
                      dest="R_MW", default = 10 | units.kpc,
                      help="Milky Way size [%default]")
    result.add_option("--MAnd", unit=units.MSun,
                      dest="M_And", default = 1.0e12 | units.MSun,
                      help="Andromeda mass [%default]")
    result.add_option("--RAnd", unit=units.kpc,
                      dest="R_And", default = 10 | units.kpc,
                      help="Andromeda size [%default]")
    result.add_option("--n_bulge", dest="n_bulge", default = 10000,
                      help="number of stars in the bulge [%default]")
    result.add_option("--n_disk", dest="n_disk", default = 10000,
                      help="number of stars in the disk [%default]")
    result.add_option("--n_halo", dest="n_halo", default = 20000,
                      help="number of stars in the halo [%default]")
    result.add_option("--t_end", unit=units.Myr,
                      dest="t_end", default = 200|units.Myr,
                      help="End of the simulation [%default]")
    result.add_option("--t_step", unit=units.Myr,
                      dest="time_step", default = 2|units.Myr,
                      help="Time step for simulation [%default]")
    return result    

if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    MW_galaxy, And_galaxy = create_two_galaxies(o.M_MW, o.M_And,
                                                o.R_MW, o.R_And,
                                                o.n_halo, o.n_bulge, o.n_disk)
##    simulate_merger(MW_galaxy, And_galaxy, o.n_halo, o.n_halo,
##                    o.M_galaxy, o.R_galaxy, o.t_end)
##

    x_MW, y_MW, x_And, y_And = simulate_merger(MW_galaxy, And_galaxy,
                                               o.n_halo, o.time_step, o.t_end)
    np.savetxt('out_x_MW.dat', x_MW)
    np.savetxt('out_y_MW.dat', y_MW)
    np.savetxt('out_x_And.dat', x_And)
    np.savetxt('out_y_And.dat', y_And)
