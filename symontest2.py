import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import quad

x, j = smp.symbols('x j', real=True,positive= True)
f1 = 3*(smp.sin(2*x)*smp.cos(2*j))*0.5
f2=3*(smp.sin(2*j)*smp.cos(2*x))*0.5


f3 = (smp.integrate(f1, (j, 0, x))+smp.integrate(f2,(j,x,np.pi/2))).simplify()
print(f3)
f3_numeric = smp.lambdify(x, f3, 'numpy')
x_values = np.linspace(0, np.pi/2, 100)
y_values = f3_numeric(x_values)

plt.plot(x_values, y_values)
plt.title("Solution of BVP with inhomogeneous term f(x) =x ")
plt.grid(True)
plt.xlabel('x_values')
plt.ylabel('y_values ')
plt.show()
