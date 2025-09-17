import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# -------------------------------
# Fixed Parameters
# -------------------------------
A = 0.1
M = 0.3
K = 0.5
R = 0.2
Sc = 3.0
QH = 0.3
lam = 0.05
delta = 0.3
gamma = 0.3
S = 0.0


# -------------------------------
# System of ODEs (dependent on Pr)
# -------------------------------
def odes(eta, y, Pr):
    dy = np.zeros_like(y)

    # f equations
    dy[0] = y[1]
    dy[1] = y[2]
    dy[2] = -(y[0] * y[2] - 2 * y[1] ** 2 - A * np.exp(-eta) * (2 * y[1] + eta * y[2]) - (M + K) * y[1])

    # theta equations
    dy[3] = y[4]
    dy[4] = -(Pr * (y[0] * y[4] - y[1] * y[3]) - A * Pr * np.exp(-eta) * (4 * y[3] + eta * y[4]) + QH * Pr * np.exp(
        -eta) * y[3]) / (1 + 4 * R / 3)

    # phi equations
    dy[5] = y[6]
    dy[6] = -(Sc * (y[0] * y[6] - y[1] * y[5]) - A * Sc * np.exp(-eta) * (4 * y[5] + eta * y[6]))

    return dy


# -------------------------------
# Boundary Conditions
# -------------------------------
def bc(ya, yb):
    return np.array([
        ya[0] - S,
        ya[1] - (1 + lam * ya[2]),
        ya[3] - (1 + delta * ya[4]),
        ya[5] - (1 + gamma * ya[6]),
        yb[1],
        yb[3],
        yb[5]
    ])


# -------------------------------
# Domain & Initial Guess
# -------------------------------
eta = np.linspace(0, 2, 200)
y_init = np.zeros((7, eta.size))
y_init[0] = S
y_init[1] = 1
y_init[3] = np.exp(-eta)  # θ initial guess
y_init[5] = np.exp(-eta)  # φ initial guess

# -------------------------------
# Solve & Plot for different Pr
# -------------------------------
Pr_values = [0.7, 1.0, 2.0, 5.0]  # example Prandtl numbers
plt.figure(figsize=(8, 6))

for Pr in Pr_values:
    # Solve BVP
    sol = solve_bvp(lambda eta, y: odes(eta, y, Pr), bc, eta, y_init, max_nodes=10000)

    if sol.success:
        print(f"Pr={Pr}: BVP solved successfully ✅")
    else:
        print(f"Pr={Pr}: BVP solver failed ❌")

    eta_plot = np.linspace(0, 2, 300)
    y_sol = sol.sol(eta_plot)

    plt.plot(eta_plot, y_sol[3], label=f'Pr = {Pr}')

plt.xlabel('η')
plt.ylabel('θ(η) - Temperature Profile')
plt.title('Effect of Pr on Temperature')
plt.grid(True)
plt.legend()
plt.show()
