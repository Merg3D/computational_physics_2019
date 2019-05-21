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

writer = FFMpegWriter(fps=25, metadata=metadata)

fig = plt.figure()
l, = plt.plot([],[], 'k-o')

timesteps = mw_x.shape[0]
print("timsteps: " + str(timesteps))

with writer.saving(fig, 'animation.mp4', timesteps):
    for i in range(timesteps):
        plt.clf()
        plt.xlim(-300, 300)
        plt.ylim(-300, 300)
        plt.title('Evolution of Andromeda and Milky Way')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(mw_x[i, :],  mw_y[i, :],  c='r', label='Milky Way', s=0.3)
        plt.scatter(and_x[i, :], and_y[i, :], c='b', label='Andromeda', s=0.3)
        plt.legend(loc='upper right')
        plt.grid()

        writer.grab_frame()