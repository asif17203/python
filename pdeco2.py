import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def matrices(n, h):
    # Generate the Laplacian matrix using finite differences
    d = np.zeros(n ** 2)
    d[n - 1::n] = 1
    e = np.zeros(n ** 2)
    e[::n] = 1
    b = np.ones(n ** 2)
    diagonals = [b, b - d, -4 * b, b - e, b]
    lap = -1 / (h ** 2) * sp.diags(diagonals, [-n, -1, 0, 1, n], shape=(n ** 2, n ** 2))
    return lap


def semilinear(lap, u):
    # Placeholder for the semilinear PDE solver, assuming a linear solve for now
    y = spla.spsolve(lap, u)
    return y


def l2norm(x):
    return np.linalg.norm(x)


# Main script
n = int(input('Mesh points: '))
h = 1 / (n + 1)
alpha = float(input('Regularization parameter: '))

x1, y1 = np.meshgrid(np.linspace(h, 1 - h, n), np.linspace(h, 1 - h, n))  # Coordinates

# Desired state
z = (x1 * y1).reshape(n ** 2)

lap = matrices(n, h)  # Laplacian
u = sp.csr_matrix((n ** 2, 1))  # Initial control
y = semilinear(lap, u.toarray())  # Initial state
Y = sp.diags(y.flatten(), 0)
p = spla.spsolve(lap + 3 * Y.power(2), y - z)  # Initial adjoint

res = 1
iter = 0

while res >= 1e-3:
    iter += 1
    Y = sp.diags(y.flatten(), 0)
    P = sp.diags(p.flatten(), 0)

    A = sp.bmat([
        [sp.eye(n ** 2) - 6 * Y.multiply(P), None, lap + 3 * Y.power(2)],
        [None, alpha * sp.eye(n ** 2), -sp.eye(n ** 2)],
        [lap + 3 * Y.power(2), -sp.eye(n ** 2), None]
    ]).tocsc()  # Ensure A is in CSC format

    # Convert u to a dense array and ensure proper shape
    u_dense = u.toarray().reshape(-1, 1)

    # Convert p to a dense array and ensure proper shape
    p_dense = p.reshape(-1, 1)

    F = np.vstack([np.zeros((n ** 2, 1)), -p_dense - alpha * u_dense, np.zeros((n ** 2, 1))])
    delta = spla.spsolve(A, F)

    # Ensure delta is reshaped to match u's shape
    delta_u = delta[n ** 2:2 * n ** 2].reshape(-1, 1)

    uprev = u_dense
    u = u_dense + delta_u  # Control update

    y = semilinear(lap, u)  # State equation
    Y = sp.diags(y.flatten(), 0)
    p = spla.spsolve(lap + 3 * Y.power(2), y - z)  # Adjoint equation

    res = l2norm(u - uprev)  # Residual for convergence check

# Reshape the solution vectors back to the 2D grid
u_grid = u.reshape(n, n)
y_grid = y.reshape(n, n)
p_grid = p.reshape(n, n)

# Plot the results

# Plot the control variable u
plt.figure()
plt.contourf(x1, y1, u_grid, cmap='viridis')
plt.title('Control Variable u')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

# Plot the state variable y
plt.figure()
plt.contourf(x1, y1, y_grid, cmap='viridis')
plt.title('State Variable y')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

# Plot the adjoint variable p
plt.figure()
plt.contourf(x1, y1, p_grid, cmap='viridis')
plt.title('Adjoint Variable p')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

plt.show()

