import numpy as np 
import pylab as plt 
from scipy.spatial import Delaunay
import math 
from scipy import ndimage
plt.close()


def punkte(xi,yi,radius,samples):    
    num_samples = samples
    theta = np.linspace(0, 2*np.pi, num_samples)
    
    r = radius*np.random.rand((num_samples))
    x, y = r * np.cos(theta)+xi, r * np.sin(theta)+yi
    return x,y

def dist(x,y):  
     dist = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)  
     return dist  
 
def abstaende(x,y,z):
    erster_abstand_punkte = dist(x,y)
    zweiter_abstand_punkte = dist(y,z)
    dritter_abstand_punkte = dist(x,z)
    maxi = max(erster_abstand_punkte,zweiter_abstand_punkte,dritter_abstand_punkte)
    return maxi

num_samples = 1000

theta = np.linspace(0, 2*np.pi, num_samples)
a, b = 5 * np.cos(theta), 5 * np.sin(theta)

anzahl_punkte_pro_kreis = 1
x = np.zeros([num_samples,anzahl_punkte_pro_kreis])
y = np.zeros([num_samples,anzahl_punkte_pro_kreis])

anzahl = num_samples-250

for i in range(anzahl):
    x_tmp , y_tmp =punkte(a[i],b[i],1,anzahl_punkte_pro_kreis)
    x[i] = x_tmp
    y[i] = y_tmp
    
#plt.plot(a,b)
#plt.plot(x,y,'b.')

a = [x[:anzahl,0],y[:anzahl,0]]
a = np.reshape(a,(anzahl,2))

tri = Delaunay(a)

plt.triplot(x[:anzahl,0],y[:anzahl,0], tri.simplices)
plt.plot(x[:anzahl,0],y[:anzahl,0], 'o')
plt.show()


tmp = np.zeros(250)
for i in range(250):
    tmp[i] = abstaende(a[tri.simplices[i,0],:],a[tri.simplices[i,1],:],a[tri.simplices[i,2],:])



a_kurz = a[np.where(tmp<1),:]


for i in [np.arange(0,100,1),np.arange(100,200,1)]: print(i)






 
