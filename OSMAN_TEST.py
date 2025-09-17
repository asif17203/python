import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


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


def eq(eta, y, Pr):
    dy = [
          y[1],  # dy[0
          y[2],  # dy[1
          -y[1] * y[2] + 2 * (y[2]) ** 2 + A * (2 * y[1] + eta * y[2]) + (M + K) * y[1],  # dy[2]
          y[4],  # dy[4]
           -Pr*(y[0]*y[1]-y[1]*y[3])+A*Pr*(4*y[3]+eta*y[4])-QH*Pr*y[3],
           y[6],
          -(Sc * (y[0] * y[6] - y[1] * y[5]) - A * Sc * np.exp(-eta) * (4 * y[5] + eta * y[6]))  #dy[5]

    ]
    return np.array(dy)
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
eta= np.linspace(0,1,200)
y_ini=np.zeros((7,eta.size))
y_ini[0]=S
y_ini[1]=0
y_ini[3]=0
y_ini[5]=0

Pr_values = [0.7, 1.0, 2.0, 5.0]
for Pr in Pr_values:
    sol = solve_bvp(lambda eta, y: eq(eta, y, Pr), bc, eta, y_ini, max_nodes=10000)
    if sol.success:
        print("Happy OSMAN")
    else:
        print("SAD OSMAN")
    eta_plot = np.linspace(0, 1, 300)
    y_sol = sol.sol(eta_plot)
    plt.plot(eta_plot, y_sol[3], label=f'Pr = {Pr}')

plt.xlabel('η')
plt.ylabel('θ(η) - Temperature Profile')
plt.title('Effect of Pr on Temperature')
plt.grid(True)
plt.legend()
plt.show()