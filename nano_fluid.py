import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# -------------------------------
# Parameters (set your values here)
# -------------------------------
A = 0.5
M = 1.0
K = 1.0
Pr = 0.7
R = 0.5
Sc = 0.6
QH = 0.2
lam = 0.1  # λ
delta = 0.1  # δ
gamma = 0.1  # γ
S = 0.0  # suction/injection parameter


# -------------------------------
# System of ODEs (converted to 1st order)
# -------------------------------
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
    dy = np.zeros_like(y)

    # f equations
    dy[0] = y[1]
    dy[1] = y[2]
    dy[2] = -(y[3]-Nr*y[5] )*cos(d)

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
        ya[0] - S,  # f(0) = S
        ya[1] - (1 + lam * ya[2]),  # f'(0) = 1 + λ f''(0)
        ya[3] - (1 + delta * ya[4]),  # θ(0) = 1 + δ θ'(0)
        ya[5] - (1 + gamma * ya[6]),  # φ(0) = 1 + γ φ'(0)
        yb[1],  # f'(∞) = 0
        yb[3],  # θ(∞) = 0
        yb[5]  # φ(∞) = 0
    ])


# -------------------------------
# Initial Guess
# -------------------------------
eta = np.linspace(0, 10, 200)  # domain [0, 10], adjust if needed
y_init = np.zeros((7, eta.size))
y_init[0] = S  # f ≈ S at start
y_init[1] = 0 # f' ≈ 1 initially
y_init[3] = 0  # θ guess
y_init[5] = 0  # φ guess

# -------------------------------
# Solve the BVP
# -------------------------------
sol = solve_bvp(odes, bc, eta, y_init, max_nodes=10000)

if sol.success:
    print("BVP solved successfully ✅")
else:
    print("BVP solver failed ❌")

# -------------------------------
# Plot Results
# -------------------------------
eta_plot = np.linspace(0, 10, 300)
y_sol = sol.sol(eta_plot)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(eta_plot, y_sol[1], label="f'(η) - velocity")
plt.legend();
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(eta_plot, y_sol[3], label="θ(η) - temperature", color='r')
plt.legend();
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(eta_plot, y_sol[5], label="φ(η) - concentration", color='g')
plt.legend();
plt.grid()

plt.xlabel("η")
plt.show()