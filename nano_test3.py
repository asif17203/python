import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import math

# Parameters
Nr = 0.5
M = 0.1
E = 0.01
Pr = 5
p = 0.6
Nt = 0.5
Nb = 0.5
Bm = 0.1
Db = 0.1
d = 1.14  # radian
S = 0.0   # δ

def odes(eta, y):
    """
    y[0] = f
    y[1] = f'
    y[2] = f''
    y[3] = theta
    y[4] = theta'
    y[5] = phi
    y[6] = phi'
    """
    dy = np.zeros((7, eta.size))

    # f-equations
    dy[0] = y[1]
    dy[1] = y[2]
    dy[2] = -((y[3] - Nr*y[5]) * math.cos(d)
              - M*y[1]
              + (1/E) * np.exp(-0.5) * (y[1] + eta*y[2])
              + (1/Pr) * (y[1])**2)

    # theta-equations
    dy[3] = y[4]
    dy[4] = -((Nb*y[4]*y[6]) + Nt*(y[4])**2
              + (1/Bm)*np.exp(-0.5)*eta*y[4])

    # phi-equations
    dy[5] = y[6]
    dy[6] = -((Nt/Nb)*dy[4] + (1/Db)*np.exp(-0.5)*eta*y[4])

    return dy

def bc(ya, yb):
    return np.array([
        ya[0] - S,   # f(0) = 0
        ya[1] - S,   # f'(0) = 0
        ya[3] - 1,   # theta(0) = 1
        ya[5] - 1,   # phi(0) = 1
        yb[1],       # f'(∞) = 0
        yb[3],       # theta(∞) = 0
        yb[5]        # phi(∞) = 0
    ])

# Initial mesh and guess
eta = np.linspace(0, 10, 200)
y_init = np.zeros((7, eta.size))
y_init[0] = S
y_init[1] = 0.1
y_init[3] = np.exp(-eta)   # initial guess for theta
y_init[5] = np.exp(-eta)   # initial guess for phi

sol = solve_bvp(odes, bc, eta, y_init, max_nodes=10000)

if sol.success:
    print("BVP solved successfully ✅")
else:
    print("BVP solver failed ❌")

# Plot
eta_plot = np.linspace(0, 10, 300)
y_sol = sol.sol(eta_plot)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(eta_plot, y_sol[1], label="f'(η) - velocity")
plt.legend(); plt.grid()

plt.subplot(3, 1, 2)
plt.plot(eta_plot, y_sol[3], label="θ(η) - temperature", color='r')
plt.legend(); plt.grid()

plt.subplot(3, 1, 3)
plt.plot(eta_plot, y_sol[5], label="φ(η) - concentration", color='g')
plt.legend(); plt.grid()


plt.subplot(3, 1, 3)
plt.plot(eta_plot, y_sol[6], label="n(η) - concentration", color='g')
plt.legend(); plt.grid()


plt.xlabel("η")
plt.show()
