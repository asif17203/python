import numpy as np
import pandas as pd

# Constants
Lx = Ly = Lz = 1.0  # Domain dimensions
Nx = Ny = Nz = 50  # Number of grid points in each dimension
dx = dy = dz = Lx / (Nx - 1)  # Grid spacing
dt = 0.01  # Time step
T_final = 0.1  # Final time
alpha = 0.01  # Thermal diffusivity

# Initial conditions
T = np.zeros((Nx, Ny, Nz))  # Temperature array
T[:, :, :] = 25  # Initial temperature everywhere

# Boundary conditions
T[:, :, 0] = 100  # Bottom boundary condition (heat flux)
T[:, :, -1] = 0.0  # Top boundary condition
T[:, 0, :] = 0.0  # Front boundary condition
T[:, -1, :] = 0.0  # Back boundary condition
T[0, :, :] = 0.0  # Left boundary condition
T[-1, :, :] = 0.0  # Right boundary condition

# FTCS method
for t in np.arange(0, T_final, dt):
    T_new = np.copy(T)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                # Specify specific y-positions for fin plates
                if j in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35] and k != 0:
                    # Adjust z-coordinate to create fins
                    z = 0  # Adjust this value based on your desired fin height
                else:
                    z = T[i, j, k] + alpha * dt * (
                        (T[i + 1, j, k] - 2 * T[i, j, k] + T[i - 1, j, k]) / dx**2 +
                        (T[i, j + 1, k] - 2 * T[i, j, k] + T[i, j - 1, k]) / dy**2 +
                        (T[i, j, k + 1] - 2 * T[i, j, k] + T[i, j, k - 1]) / dz**2
                    )
                T_new[i, j, k] = z
    T = np.copy(T_new)

# Create coordinates
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)

# Create DataFrame
data = {'x': [], 'y': [], 'z': [], 'T': []}
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            data['x'].append(x[i])
            data['y'].append(y[j])
            data['z'].append(z[k])
            data['T'].append(T[i, j, k])
df = pd.DataFrame(data)

# Save to Excel
df.to_csv('heat.csv', index=False)
