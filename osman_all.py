import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import math

# -------------------------------
# Parameters (set your base values here)
# -------------------------------
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
Dm = 0.5
Pe = 0.5
delta = 0.1
S = 0.0  # suction/injection parameter

# -------------------------------
# System of ODEs
# -------------------------------
def odes(eta, y, Nr, M, Pr):
    dy = np.zeros((9, eta.size))

    # f equations
    dy[0] = y[1]
    dy[1] = y[2]
    dy[2] = -((y[3] - Nr*y[5]) * math.cos(d)
              - M*y[1]
              + (1/E) * np.exp(-0.5) * (y[1] + eta*y[2])
              + (1/Pr) * (y[1])**2)

    # theta equations
    dy[3] = y[4]
    dy[4] = -((Nb*y[4]*y[6])
              + Nt*(y[4])**2
              + (1/Bm) * np.exp(-0.5) * eta * y[4])

    # phi equations
    dy[5] = y[6]
    dy[6] = -((Nt/Nb) * dy[4]
              + (1/Db) * np.exp(-0.5) * eta * y[4])

    # n equations
    dy[7] = y[8]
    dy[8] = -((1/Dm) * np.exp(-0.5) * eta * y[8]
              - Pe * ((delta + y[7]) * dy[6] + y[8]*y[4]))

    return dy

# -------------------------------
# Boundary Conditions
# -------------------------------
def bc(ya, yb):
    return np.array([
        ya[0] - S,   # f(0) = S
        ya[1] - S,   # f'(0) = S
        ya[3] - 1,   # θ(0) = 1
        ya[5] - 1,   # φ(0) = 1
        ya[7] - 1,   # n(0) = 1
        yb[1],       # f'(∞) = 0
        yb[3],       # θ(∞) = 0
        yb[5],       # φ(∞) = 0
        yb[7]        # n(∞) = 0
    ])

# -------------------------------
# Domain & Initial Guess
# -------------------------------
eta = np.linspace(0, 1, 200)
y_init = np.zeros((9, eta.size))
y_init[0] = S
y_init[1] = 1
y_init[3] = np.exp(-eta)  # θ initial guess
y_init[5] = np.exp(-eta)  # φ initial guess

# -------------------------------
# Variation of Pr
# -------------------------------
Pr_values = [5, 10, 15, 20]
plt.figure(figsize=(8, 6))

for Pr in Pr_values:
    sol = solve_bvp(lambda eta, y: odes(eta, y, Nr, M, Pr), bc, eta, y_init, max_nodes=10000)
    eta_plot = np.linspace(0, 1, 300)
    y_sol = sol.sol(eta_plot)
    plt.plot(eta_plot, y_sol[1], label=f'Pr = {Pr}')  # plotting θ (temperature)

plt.xlabel('η')
plt.ylabel('θ(η)')
plt.title('Effect of Pr on Temperature')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Variation of Nr
# -------------------------------
Nr_values = [0.1, 0.5, 1.0, 2.0]
plt.figure(figsize=(8, 6))

for Nr in Nr_values:
    sol = solve_bvp(lambda eta, y: odes(eta, y, Nr, M, Pr), bc, eta, y_init, max_nodes=10000)
    eta_plot = np.linspace(0, 1, 300)
    y_sol = sol.sol(eta_plot)
    plt.plot(eta_plot, y_sol[1], label=f'Nr = {Nr}')  # plotting θ (temperature)

plt.xlabel('η')
plt.ylabel('θ(η)')
plt.title('Effect of Nr on Temperature')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Variation of M
# -------------------------------
M_values = [0.1, 0.5, 1.0, 2.0]
plt.figure(figsize=(8, 6))

for M in M_values:
    sol = solve_bvp(lambda eta, y: odes(eta, y, Nr, M, Pr), bc, eta, y_init, max_nodes=10000)
    eta_plot = np.linspace(0, 1, 300)
    y_sol = sol.sol(eta_plot)
    plt.plot(eta_plot, y_sol[1], label=f'M = {M}')  # plotting θ (temperature)

plt.xlabel('η')
plt.ylabel('θ(η)')
plt.title('Effect of M on Temperature')
plt.legend()
plt.grid(True)
plt.show()
