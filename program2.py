import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
k = 1.0  # Thermal conductivity
rho = 1.0  # Density
cp = 1.0  # Specific heat capacity
dt = 0.1  # Time step size
spacing = 0.1  # Pin fin spacing
tolerance = 1e-6  # Convergence criterion

# Dimensions of the fin plate
plate_width = 1.0
plate_height = 1.0
plate_depth = 1.0

# Calculate the number of grid points along each dimension
nx = int(plate_width / spacing) + 1
ny = int(plate_height / spacing) + 1
nz = int(plate_depth / spacing) + 1


# Function to initialize the temperature array
def initialize_temperature():
    T = np.zeros((nx, ny, nz))
    return T


# Function to apply boundary conditions
def apply_boundary_conditions(T):
    # Set the boundary temperatures
    T[:, :, 0] = 100.0  # Front face
    T[:, :, -1] = 0.0  # Back face
    T[:, 0, :] = 75.0  # Bottom face
    T[:, -1, :] = 50.0  # Top face
    T[0, :, :] = 50.0  # Left face
    T[-1, :, :] = 25.0  # Right face
    return T


# Function to solve the heat equation using finite difference method
def solve_heat_equation(T):
    T_new = np.copy(T)
    error = 0.0
    max_error = 0.0
    iterations = 0

    # Perform iterations until convergence
    while True:
        # Copy the old temperature array to T_new
        T_new[:] = T[:]

        # Perform one iteration
        max_error = 0.0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    T[i, j, k] = T_new[i, j, k] + (k / (rho * cp)) * (
                            k * (T_new[i + 1, j, k] + T_new[i - 1, j, k] - 2 * T_new[i, j, k]) / (spacing * spacing) +
                            k * (T_new[i, j + 1, k] + T_new[i, j - 1, k] - 2 * T_new[i, j, k]) / (spacing * spacing) +
                            k * (T_new[i, j, k + 1] + T_new[i, j, k - 1] - 2 * T_new[i, j, k]) / (spacing * spacing)
                    )

                    error = abs(T[i, j, k] - T_new[i, j, k])
                    if error > max_error:
                        max_error = error

        iterations += 1

        if max_error <= tolerance:
            break

    print("Converged in", iterations, "iterations.")
    return T


# Function to plot the temperature distribution
def plot_temperature(T):
    x = np.linspace(0, plate_width, nx)
    y = np.linspace(0, plate_height, ny)
    z = np.linspace(0, plate_depth, nz)
    X, Y, Z = np.meshgrid(x, y, z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Temperature Distribution')

    # Plot the temperature as a surface plot
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.jet(T), linewidth=0, antialiased=False)
    fig.colorbar(surf)

    plt.show()


# Main function
def main():
    # Initialize the temperature array
    T = initialize_temperature()

    # Apply boundary conditions
    T = apply_boundary_conditions(T)

    # Solve the heat equation
    T = solve_heat_equation(T)

    # Plot the temperature distribution
    plot_temperature(T)


# Run the main function
main()
