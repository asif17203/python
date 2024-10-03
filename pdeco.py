import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def matrices(n, h):
    N = n * n
    diagonals = [np.ones(N), np.ones(N), -4 * np.ones(N), np.ones(N), np.ones(N)]
    offsets = [-n, -1, 0, 1, n]

    lap = sp.lil_matrix((N, N))
    lap.setdiag(diagonals[2])  # main diagonal
    lap.setdiag(diagonals[0][:-n], k=-n)  # upper n-diagonal
    lap.setdiag(diagonals[0][:-1], k=-1)  # upper 1-diagonal
    lap.setdiag(diagonals[3][:-1], k=1)  # lower 1-diagonal
    lap.setdiag(diagonals[4][:-n], k=n)  # lower n-diagonal

    for i in range(1, n):
        lap[i * n, i * n - 1] = 0
        lap[i * n - 1, i * n] = 0

    lap *= -1 / (h * h)
    return lap.tocsr()


def semilinear(lap, u):
    N = len(u)
    y = spla.spsolve(lap, u)
    return y


def l2norm(vec):
    return np.linalg.norm(vec)


# Parameters
n = int(input('Mesh points: '))
h = 1 / (n + 1)
alpha = float(input('Regularization parameter: '))

# Coordinates
x1, y1 = np.meshgrid(np.linspace(h, 1 - h, n), np.linspace(h, 1 - h, n))

# Desired state
desiredstate = lambda x, y: x * y
z = desiredstate(x1, y1).reshape(n * n)

# Laplacian
lap = matrices(n, h)

# Initial control
u = np.zeros(n * n)

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
    Y = sp.diags(y, 0)
    P = sp.diags(p, 0)

    A = sp.bmat([
        [sp.eye(n * n) - 6 * Y.multiply(P), None, lap + 3 * Y.multiply(Y)],
        [None, alpha * sp.eye(n * n), -sp.eye(n * n)],
        [lap + 3 * Y.multiply(Y), -sp.eye(n * n), None]
    ])

    F = np.concatenate([
        np.zeros(n * n),
        -p - alpha * u,
        np.zeros(n * n)
    ])

    delta = spla.spsolve(A, F)

    delta_y = delta[:n * n]
    delta_u = delta[n * n:2 * n * n]
    delta_pi = delta[2 * n * n:]

    uprev = u.copy()
    u += delta_u
    y = semilinear(lap, u)
    Y = sp.diags(y, 0)
    p = spla.spsolve(lap + 3 * Y.multiply(Y), y - z)

    res = l2norm(u - uprev)
    print(f'Iteration: {iter}, Residual: {res}')

# Visualization
x = np.linspace(0, 1, n + 2)
y = np.linspace(0, 1, n + 2)
X, Y = np.meshgrid(x, y)

U = np.zeros((n + 2, n + 2))
U[1:-1, 1:-1] = u.reshape((n, n))

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, U, 20, cmap='hot')
plt.colorbar()
plt.title('Control Function u')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
