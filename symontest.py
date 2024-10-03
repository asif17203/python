import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Directory where your .dat files are located
data_dir = 'C:/coading/practice/tr_heat/heat_0.dat'

# Number of frames
num_frames = 103

# Create a figure and axis for the plot
fig, ax = plt.subplots()


def update(frame):
    # Load data from the .dat file
    filename = f'{data_dir}heat_{frame}.dat'
    data = np.loadtxt(filename)

    # Clear the axis and plot the new data
    ax.clear()
    c = ax.imshow(data, cmap='hot', interpolation='nearest')
    ax.set_title(f'Frame {frame}')
    return [c]


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)

# Save the animation as a GIF using Pillow
ani.save('heat_equation.gif', writer='pillow', fps=10)
