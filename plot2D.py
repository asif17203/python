import numpy as np
import matplotlib.pyplot as plt

a = 111
l=50
t=4
n= 50
dx = l/n
dy=l/n
dt = min( dx**2/(4*a), dy**2/(4*a) )

u=np.zeros((n,n))+20
t_n=int(t/dt)
#vis
fig,axis =plt.subplots()
pcm = axis.pcolormesh(u,cmap=plt.cm.jet,vmin=0,vmax=100)
plt.colorbar(pcm,ax=axis)


#bcand ic
u[0,:]=np.linspace(0,100,n)
u[-1,:]=np.linspace(0,100,n)
u[:,0]=np.linspace(0,100,n)
u[:,-1]=np.linspace(0,100,n)
counter = 0
while counter < t:
       w=u.copy()
       for i in range(1,n-1):
        for j in range(1,n-1):
         dd_ux=(w[i-1,j]-2*w[i,j]+w[i+1,j])/dx**2
         dd_uy=(w[i,j-1]-2*w[i,j]+w[i,j+1])/dy**2
        u[i,j]=w[i,j]+a*dt*(dd_ux+dd_uy)


        counter += dt
        print("t: {:.3f} [s], Average temperature: {:.2f} Celcius".format(counter, np.average(u)))

        pcm.set_array(u)
        axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
        plt.pause(0.01)
plt.show()