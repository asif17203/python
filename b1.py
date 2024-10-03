import numpy as np
import matplotlib.pyplot as plt

def solver_FECS(I, U0, v, L, dt, C, T, user_action=None):
    Nt = int(round(T / float(dt)))
    t = np.linspace(0, Nt * dt, Nt + 1)  # Mesh points in time
    dx = v * dt / C
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)  # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    u = np.zeros(Nx + 1)
    u_n = np.zeros(Nx + 1)
    # Set initial condition u(x, 0) = I(x)
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])
    if user_action is not None:
        user_action(u_n, x, t, 0)
    for n in range(0, Nt):
        # Compute u at inner mesh points
        for i in range(1, Nx):
            u[i] = u_n[i] - 0.5 * C * (u_n[i + 1] - u_n[i - 1])
        # Insert boundary condition
        u[0] = U0
        if user_action is not None:
            user_action(u, x, t, n + 1)
        # Switch variables before next step
        u_n, u = u, u_n

def run_FECS(case):
    """Special function for the FECScase."""
    if case == 'gaussian':
        def I(x):
            return np.exp(-0.5 * ((x - L / 10) / sigma)**2)
    elif case == 'cosinehat':
        def I(x):
            return np.cos(np.pi * 5 / L * (x - L / 10)) if x < L / 5 else 0

    L = 1.0
    sigma = 0.02
    legends = []

    def plot(u, x, t, n):
        """Animate and plot every m steps in the same figure."""
        plt.figure(1)
        if n == 0:
            lines = plt.plot(x, u)
        else:
            lines.set_ydata(u)
        plt.draw()
        # plt.savefig()
        plt.figure(2)
        m = 40
        if n % m != 0:
            return
        print('t=%g, n=%d, u in [%g, %g] w/%d points' % (t[n], n, u.min(), u.max(), x.size))
        if np.abs(u).max() > 3:  # Instability?
            return
        plt.plot(x, u)
        legends.append('t=%g' % t[n])
        if n > 0:
            plt.hold('on')
        plt.ion()

    U0 = 0
    dt = 0.001
    C = 1
    T = 1
    solver_FECS(I=I, U0=U0, v=1.0, L=1, dt=dt, C=C, T=T, user_action=plot)
    plt.legend(legends, loc='lowerleft')
    plt.savefig('tmp.png')
    plt.savefig('tmp.pdf')
    plt.axis([0, 1, -0.75, 1.1])
    plt.show()

run_FECS('gaussian')  # or run_FECS('cosinehat') to run the other case
