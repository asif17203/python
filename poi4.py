import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


def matrices(n, h):
    """
    Construct the Laplacian matrix using sparse diagonals.
    """
    d = np.ones(n ** 2)
    d[n - 1::n] = 0  # Correct the diagonal for periodic boundary condition
    e = np.ones(n ** 2)
    e[::n] = 0

    diagonals = [-d, -e, -4 * np.ones(n ** 2), -e, -d]
    lap = sp.diags(diagonals, [-n, -1, 0, 1, n], shape=(n ** 2, n ** 2))
    lap = lap * (-1 / h ** 2)

    return lap.tocsr()


def semilinear(lap, u):
    """
    Solve the semilinear problem.
    """
    y = spla.spsolve(lap, u)
    return y


def l2norm(vec):
    """
    Calculate the L2 norm of a vector.
    """
    return np.linalg.norm(vec)


def main():
    # Get user input for mesh points and regularization parameter
    n = int(input('Mesh points: '))
    h = 1 / (n + 1)
    alpha = float(input('Regularization parameter: '))

    # Generate coordinates
    x1, y1 = np.meshgrid(np.linspace(h, 1 - h, n), np.linspace(h, 1 - h, n))

    # Define the desired state
    z = (x1 * y1).reshape(n ** 2)

    # Generate Laplacian matrix
    lap = matrices(n, h)

    # Initial control
    u = np.zeros(n ** 2)

    # Initial state
    y = semilinear(lap, u)

    # Initial adjoint
    Y = sp.diags(y, 0)
    p = spla.spsolve(lap + 3 * Y.multiply(Y), y - z)

    # Optimization loop
    res = 1
    iter = 0

    while res >= 1e-3:
        iter += 1

        # Update matrices
        Y = sp.diags(y, 0)
        P = sp.diags(p, 0)
        A = sp.bmat([
            [sp.eye(n ** 2) - 6 * Y.multiply(P), None, lap + 3 * Y.multiply(Y)],
            [None, alpha * sp.eye(n ** 2), -sp.eye(n ** 2)],
            [lap + 3 * Y.multiply(Y), -sp.eye(n ** 2), None]
        ])

        # Right-hand side vector
        F = np.concatenate([
            np.zeros(n ** 2),
            -p - alpha * u,
            np.zeros(n ** 2)
        ])

        # Solve for delta
        delta = spla.spsolve(A.tocsr(), F)

        # Update control and state
        uprev = u.copy()
        u += delta[n ** 2:2 * n ** 2]
        y = semilinear(lap, u)

        # Update adjoint
        Y = sp.diags(y, 0)
        p = spla.spsolve(lap + 3 * Y.multiply(Y), y - z)

        # Calculate residual
        res = l2norm(u - uprev)
        print(f'Iteration: {iter}, Residual: {res}')

    # Visualization - 3D surface plot
    x = np.linspace(0, 1, n + 2)
    y = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(x, y)

    U = np.zeros((n + 2, n + 2))
    U[1:-1, 1:-1] = u.reshape((n, n))

    # Plot the control function as a 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, U, cmap='viridis')
    ax.set_title('Control Function u (3D)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()


if __name__ == "__main__":
    main()
