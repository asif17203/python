import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 1 if x<0 else -1

def a_0():
    return 1/np.pi*quad(lambda x:f(x),-np.pi,np.pi)[0]
def a_n(n):
    return 1/np.pi*quad(lambda x:f(x)*np.cos(n*x),-np.pi,np.pi)[0]
def b_n(n):
    return 1/np.pi*quad(lambda x:f(x)*np.sin(n*x),-np.pi,np.pi)[0]

def fourier_series(x,N):
    s=a_0()/2
    for n in range(1,N+1):
     s+= a_n(n)*np.cos(n*x)+b_n(n)*np.sin(n*x)
    return s
x=np.linspace(-np.pi,np.pi,1000)
f_vals = np.vectorize(f)(x)
plt.plot(x,f_vals)
for N in [1,2,3,4,21]:
    y=fourier_series(x,N)
    plt.plot(x,y,label=f'N={N}')


plt.show()