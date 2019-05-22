galaxy_merger.py currently outputs information into 4 .dat files tracking the
x and y positions of 40.000 particles in the Milky Way (MW) and Andromeda (And)
for 100 time steps. One file indicates the dimension and galaxy, then records
40.000 columns for each particle and 100 rows for each time step.

It also creates a plot of the kinetic, potential, and total energy in 'energy_plt.png'.

./scrap/ has test files. Other files are necessary for running AMUSE, do not
remove or move them from this directory.

data_processing.py processes the .dat out_*_*.dat files to a movie. It needs the program ffmpeg, which can be installed by for example `apt install ffmpeg`.
