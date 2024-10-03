import numpy as np
import matplotlib.pyplot as plt

def bvp_green_function():
    # Define the BVP parameters
    a = 0  # left boundary
    b = 1  # right boundary

    # Set the inhomogeneous term
    x_squared = lambda x: x**2

    # Define the grid
    N = 100  # number of points
    x = np.linspace(a, b, N)

    # Calculate the Green's function
    G = calculate_green_function(x, a, b)

    # Convolve the Green's function with the inhomogeneous term (f = x^2)
    y = convolve_green_function(G, x_squared(x), x, a, b)

    # Plot the solution
    plt.figure()
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title("Solution of BVP with inhomogeneous term f(x) = x^2")
    plt.grid(True)
    plt.show()

def calculate_green_function(x, a, b):
    # Calculate the Green's function for the given BVP
    G = (x - a) * (b - x) / (2 * (b - a))
    return G

def convolve_green_function(G, f, x, a, b):
    # Convolve the Green's function with the inhomogeneous term
    delta_x = x[1] - x[0]
    y = delta_x * np.convolve(G, f, mode='full')[:len(x)]
    return y

# Call the main function
bvp_green_function()