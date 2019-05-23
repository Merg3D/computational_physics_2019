import numpy
from matplotlib import pyplot as plt
from amuse.units import units, constants
from amuse.units.optparse import OptionParser
from amuse.lab import nbody_system
from amuse.lab import Gadget2
from amuse.datamodel import Particles
from amuse.ext.galactics_model import new_galactics_model

def make_plot(disk1, disk2, filename):
    x_label = "X [kpc]"
    y_label = "Y [kpc]"
    plt.xlim(min(disk2.x.value_in(units.kpc)),
                    max(disk1.x.value_in(units.kpc)))
    plt.ylim(min(disk2.y.value_in(units.kpc)),
                    max(disk1.y.value_in(units.kpc)))

    plt.scatter(disk1.x.value_in(units.kpc), disk1.y.value_in(units.kpc),
                    alpha=1, s=1, lw=0)
    plt.scatter(disk2.x.value_in(units.kpc), disk2.y.value_in(units.kpc),
                   alpha=1, s=1, lw=0)
    plt.savefig(filename)
    plt.close()

def make_galaxies(M_galaxy, R_galaxy, n_halo, n_bulge, n_disk):
    converter=nbody_system.nbody_to_si(M_galaxy, R_galaxy)
    galaxy1 = new_galactics_model(n_halo,
                                  converter,
                                  #do_scale = True,
                                  bulge_number_of_particles=n_bulge,
                                  disk_number_of_particles=n_disk)
    galaxy2 = Particles(len(galaxy1))
    galaxy2.mass = galaxy1.mass.value_in(units.MSun) + 0.5 | units.MSun
    galaxy2.position = galaxy1.position
    galaxy2.velocity = galaxy1.velocity
    
    galaxy1.rotate(0., numpy.pi/2, numpy.pi/4)
    galaxy1.position += [500.0, 500, 0] | units.kpc
    galaxy1.velocity += [-10.0, 0.0, -10.0] | units.km/units.s

    galaxy2.rotate(numpy.pi/4, numpy.pi/4, 0.0)
    galaxy2.position -= [500.0, 0, 0] | units.kpc
    galaxy2.velocity -= [0.0, 0.0, 0] | units.km/units.s

    return galaxy1, galaxy2, converter

def simulate_merger(galaxy1, galaxy2, converter, n_halo, t_end):
    converter = nbody_system.nbody_to_si(1.0e12|units.MSun, 100|units.kpc)
    dynamics = Gadget2(converter, number_of_workers=4)
    dynamics.parameters.epsilon_squared = (100 | units.parsec)**2
    set1 = dynamics.particles.add_particles(galaxy1)
    set2 = dynamics.particles.add_particles(galaxy2)
    dynamics.particles.move_to_center()
    disk1 = set1[:n_halo]
    disk2 = set2[:n_halo]

    make_plot(disk1, disk2, 'start.png')
##    dynamics.evolve_model(t_end)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    KE = []
    PE = []
    TotE = []
    t = []
    time = 0.0 | units.Myr
    time_step = 20 | units.Myr
    while time < t_end:
        print(time)
        t.append(time.value_in(units.Myr))
        dynamics.evolve_model(time)
        x1.append(disk1.x.value_in(units.kpc))
        y1.append(disk1.y.value_in(units.kpc))
        x2.append(disk2.x.value_in(units.kpc))
        y2.append(disk2.y.value_in(units.kpc))
        print('positions set')
        kin = dynamics.kinetic_energy.value_in(units.J)
        KE.append(kin)
        print('ke set')
        pot = dynamics.potential_energy.value_in(units.J)
        PE.append(pot)
        print('pe set')
        TotE.append(kin + pot)
        print('tot e set')
        
        time += time_step
    make_plot(disk1, disk2, str(time)+'end.png')
    dynamics.stop()

    # Plot energies
    plt.plot(t, KE, label='Kinetic Energy')
    plt.plot(t, PE, label='Potential Energy')
    plt.plot(t, TotE, label='Total Energy')
    plt.plot((min(t), max(t)),(0,0), 'k--')
    plt.xlabel('Time (Myr)')
    plt.ylabel('Energy (J)')
    plt.title('Change In Energy of System')
    plt.legend()
    plt.savefig('energy_plot.png')
    plt.close()
    print('Energy plot complete')

    return x1, x2, y1, y2, KE, PE, TotE, t
    

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-M", unit=units.MSun,
                      dest="M_galaxy", default = 1.0e12 | units.MSun,
                      help="Galaxy mass [%default]")
    result.add_option("-R", unit=units.kpc,
                      dest="R_galaxy", default = 10 | units.kpc,
                      help="Galaxy size [%default]")
    result.add_option("--n_bulge", dest="n_bulge", default = 10,
                      help="number of stars in the bulge [%default]")
    result.add_option("--n_disk", dest="n_disk", default = 10,
                      help="number of stars in the disk [%default]")
    result.add_option("--n_halo", dest="n_halo", default = 20,
                      help="number of stars in the halo [%default]")
    result.add_option("--t_end", unit=units.Myr,
                      dest="t_end", default = 1400|units.Myr,
                      help="End of the simulation [%default]")
    return result

if __name__ == '__main__':
    o, arguments  = new_option_parser().parse_args()
    galaxy1, galaxy2, converter = make_galaxies(o.M_galaxy, o.R_galaxy,
                                                o.n_halo, o.n_bulge, o.n_disk)
    x1, x2, y1, y2, KE, PE, TotE, t = simulate_merger(galaxy1, galaxy2,
                                                      converter, o.n_halo,
                                                      o.t_end)
    numpy.save('MW_x.dat', x1)
    numpy.save('MW_y.dat', y1)
    numpy.save('And_x.dat', x2)
    numpy.save('And_y.dat', y2)
    numpy.save('time', t)
    numpy.save('KE', KE)
    numpy.save('PE', PE)
    numpy.save('Total_Energy', TotE)
