import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


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


def l2norm(vec):
    """
    Calculate the L2 norm of a vector.
    """
    return np.linalg.norm(vec)


def armijolq(u, y, p, z, alpha, lap):
    """
    Armijo line search algorithm.
    """
    countarm = 0
    gradcost = l2norm(p + alpha * u) ** 2
    cost1 = 0.5 * l2norm(y - z) ** 2 + alpha / 2 * l2norm(u) ** 2
    beta = 1
    armijo = 1e5

    while armijo > -1e-4 * beta * gradcost:
        beta = 0.5 ** countarm
        uinc = u - beta * (p + alpha * u)
        yinc = spla.spsolve(lap, uinc)
        cost2 = 0.5 * l2norm(yinc - z) ** 2 + alpha / 2 * l2norm(uinc) ** 2
        armijo = cost2 - cost1
        countarm += 1

    return beta


def plot_state(state, n, title):
    """
    Plot the state matrix.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(state.reshape((n, n)), cmap='viridis', origin='lower')
    plt.colorbar(label='State value')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


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

    # Visualization of the initial state
    plot_state(u, n, 'Initial State')

    # Optimization loop
    res = 1
    iter = 0
    tol = 1e-3
    residuals = []

    while res >= tol:
        iter += 1
        print(f"Iteration: {iter}")

        # State equation
        y = spla.spsolve(lap, u)

        # Adjoint solver
        p = spla.spsolve(lap, y - z)

        # Armijo line search
        beta = armijolq(u, y, p, z, alpha, lap)

        # Gradient step
        uprev = u.copy()
        u = u - beta * (p + alpha * u)

        # Residual computation
        res = l2norm(u - uprev)
        print(f"Residual: {res}")
        residuals.append(res)

    # Visualization of the final state
    plot_state(u, n, 'Final State')

    # Plot the residuals over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(residuals, marker='o')
    plt.title('Residuals over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
