import numpy as np
import matplotlib.pyplot as plt
import fenics as fe
a = 1110
l=30
t=4
n= 100
dx = l/n-1
dt =( 0.5*dx**2)/a

u=np.zeros(n)+20
t_n=int(t/dt)
#vis
fig,axis =plt.subplots()
pcm = axis.pcolormesh([u],cmap=plt.cm.jet,vmin=0,vmax=100)
plt.colorbar(pcm,ax=axis)
axis.set_ylim([-2, 3])

#bcand ic
u[0]=100
u[n-1]=100
counter = 0
while counter < t:
       w=u.copy()
       for i in range(1,n-1):

        u[i]=w[i]+a*dt*(w[i+1]-2*w[i] +w[i-1])/dx**2


        counter += dt
        print("t: {:.3f} [s], Average temperature: {:.2f} Celcius".format(counter, np.average(u)))

        pcm.set_array([u])
        axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
        plt.pause(0.01)
plt.show()