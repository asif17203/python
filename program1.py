import numpy as np
import matplotlib.pyplot as plt
from numpy import sin ,exp,pi
x=np.linspace(0,1,100)
t=np.linspace(0,0.006,100)
def T(x,t): return np.exp(-np.pi**2*t)*sin(np.pi*x)
plt.plot(x,T(x,t))
plt.show()


