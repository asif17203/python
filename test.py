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
    diagonals[4, :-1] = 1  # -1 on the (i-1)th column

    offsets = np.array([-n, 0, n, -1, 1])
    A = sp.diags(diagonals, offsets, shape=(N, N))

    return A


def apply_dirichlet_bc(A, b, n, u_bc):
    """Modify matrix A and vector b to apply Dirichlet boundary conditions."""
    # Flatten the boundary conditions array
    u_bc_flat = u_bc.flatten()

    # Dirichlet boundary condition indices
    boundary_indices = np.concatenate([
        np.arange(n),  # Top row
        np.arange(n * (n - 1), n * n),  # Bottom row
        np.arange(n, n * (n - 1), n),  # Left column
        np.arange(n - 1, n * (n - 1) - 1, n)  # Right column
    ])

    # Remove duplicates
    boundary_indices = np.unique(boundary_indices)

    # Convert A to COO format for easier modification
    A = A.tocoo()

    # Modify A and b for Dirichlet boundary conditions
    for idx in boundary_indices:
        # Zero out the row
        row_start = A.row[A.row == idx]
        row_end   = A.row[A.row == idx]
        col_start = A.col[A.row == idx]
        A.data[np.isin(A.row, row_start) & np.isin(A.col, col_start)] = 0

        # Set the diagonal to 1
        A.data[np.logical_and(A.row == idx, A.col == idx)] = 1
        b[idx] = u_bc_flat[idx]

    A = A.tocsr()  # Convert back to CSR format for efficient solving
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

    # Define Dirichlet boundary conditions (example: all boundaries set to 100)
    u_bc = np.zeros((n, n))
    u_bc[0, :] = 0 # Top boundary
    u_bc[-1, :] =30  # Bottom boundary
    u_bc[:, 0] =30 # Left boundary
    u_bc[:, -1] = 30  # Right boundary

    u = solve_laplace(n, u_bc)
    visualize_solution(u)


if __name__ == "__main__":
    main()

