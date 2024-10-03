import numpy as np
import matplotlib.pyplot as plt

def solver_FECS(I, U0, v, L, dt, C, time, user_action=None):
    Nt = int(round(time / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)
    dx = v * dt / C
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    u = np.zeros(Nx + 1)
    u_n = np.zeros(Nx + 1)
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])
    if user_action is not None:
        user_action(u_n, x, t, 0)
    for n in range(0, Nt):
        for i in range(1, Nx):
            u[i] = u_n[i] - 0.5 * C * (u_n[i + 1] - u_n[i - 1])
        u[0] = U0  # Set boundary condition
        if user_action is not None:
            user_action(u, x, t, n + 1)
        u_n[:] = u

def plot(u, x, t, n):
    plt.figure(1)
    if n == 0:
        plt.plot(x, u)
    else:
        plt.gca().get_lines()[0].set_ydata(u)
    plt.draw()
    plt.figure(2)
    m = 40
    if n % m != 0:
        return
    print('t=%g, n=%d, u in [%g, %g] w/%d points' % (t[n], n, u.min(), u.max(), x.size))
    if np.abs(u).max() > 3:
        return
    plt.plot(x, u)

def run_FECS(case):
    L = 1.0
    sigma = 0.02
    if case == 'gaussian':
        def I(x):
            return np.exp(-0.5 * ((x - L / 10) / sigma) ** 2)
    elif case == 'cosinehat':
        def I(x):
            return np.cos(np.pi * 5 / L * (x - L / 10)) if x <= L / 5 else 0

    solver_FECS(I=I, U0=0, v=1.0, L=L, dt=0.001, C=1, time=1, user_action=plot)
    plt.savefig('tmp.png')
    plt.savefig('tmp.pdf')
    plt.axis([0, L, -0.75, 1.1])
    plt.show()

run_FECS('gaussian')  # or run_FECS('cosinehat') to run the other case
