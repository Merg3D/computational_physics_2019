import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
matplotlib.use("Agg")

# Import data
and_x = np.genfromtxt('out_x_And.dat', delimiter=' ')
and_y = np.genfromtxt('out_y_And.dat', delimiter=' ')
mw_x = np.genfromtxt('out_x_MW.dat', delimiter=' ')
mw_y = np.genfromtxt('out_y_MW.dat', delimiter=' ')

# Create animated plot of merger
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title = 'Galaxy Collision',
                artist = 'Computational Physics',
                comment = 'Computational Physcs')

writer = FFMpegWriter(fps=10, metadata=metadata)

fig = plt.figure()
l, = plt.plot([],[], 'k-o')
timesteps = 10
nsteps = 10
min_x = np.minimum(np.min(mw_x), np.min(and_x))
max_x = np.maximum(np.max(mw_x), np.max(and_x))
min_y = np.minimum(np.min(mw_y), np.min(and_y))
max_y = np.maximum(np.max(mw_y), np.max(and_y))

with writer.saving(fig, 'output.mp4', nsteps*timesteps):
    for x in range(nsteps):
        for i in range(timesteps):
            plt.clf()
            plt.xlim(min_x, max_x)
            plt.ylim(min_y, max_y)
            plt.grid()
            plt.title('Evolution of Andromeda and Milky Way')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.scatter(mw_x[i, :],  mw_y[i, :],  c='r', label='Milky Way')
            plt.scatter(and_x[i, :], and_y[i, :], c='b', label='Andromeda')

            writer.grab_frame()