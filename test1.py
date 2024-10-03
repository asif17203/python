import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt


def create_laplacian(n):
    """Create a 2D Laplacian matrix using the finite difference method."""
    N = n * n
    diagonals = np.zeros((5, N))
    # Diagonal (-1, 0, 1)
    diagonals[0, n:] = 1  # -1 on the (i-1)th row
    diagonals[1, :] = -4  # 0 on the i-th row
    diagonals[2, :-n] = 1  # 1 on the (i+1)th row
    diagonals[3, 1:] = 1  # 1 on the (i+1)th column
    diagonals[4, :-1] = 1  # 1 on the (i-1)th column

    offsets = np.array([-n, 0, n, -1, 1])
    A = sp.diags(diagonals, offsets, shape=(N, N))

    return A


def apply_dirichlet_bc(A, b, n, u_bc):
    """Modify matrix A and vector b to apply Dirichlet boundary conditions."""
    # Flatten the boundary conditions array
    u_bc_flat = u_bc.flatten()

    # Calculate boundary indices
    top_row_indices = np.arange(n)

    # Convert A to CSR format for easier modification
    A = A.tocsr()

    # Apply Dirichlet boundary conditions to the top row
    for idx in top_row_indices:
        A[idx, :] = 0  # Zero out the row
        A[idx, idx] = 1  # Set the diagonal to 1
        b[idx] = u_bc_flat[idx]

    return A, b


def solve_laplace(n, u_bc):
    """Solve the Laplace equation on an n x n grid with Dirichlet boundary conditions."""
    A = create_laplacian(n)
    b = np.zeros(n * n)

    A, b = apply_dirichlet_bc(A, b, n, u_bc)

    # Solve the system
    u = linalg.spsolve(A, b).reshape((n, n))

    return u


def visualize_solution(u):
    """Visualize the solution using a heatmap."""
    plt.imshow(u, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature')
    plt.title('Solution to the Laplace Equation with Dirichlet BC')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():
    n = int(input("Enter the grid size (n): "))

    # Define Dirichlet boundary conditions (e.g., top boundary set to 200)
    u_bc = np.zeros((n, n))
    u_bc[0, :] = 200  # Top boundary

    # You can set other boundaries if needed, e.g.,
    u_bc[-1, :] =100  # Bottom boundary
    u_bc[:, 0] = 100  # Left boundary
    # u_bc[:, -1] = 0  # Right boundary

    u = solve_laplace(n, u_bc)
    visualize_solution(u)


if __name__ == "__main__":
    main()
