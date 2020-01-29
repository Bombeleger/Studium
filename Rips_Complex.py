'''
Es werden zufällige Punkte generiert, es folgen Funktionen für das Rips Komplex, 
sowie für die Eulercharakteristik. 
Für ein Epsilon Array werden die Eulercharakteristiken bestimmt.
'''

import gudhi
import numpy as np
import matplotlib.pyplot as plt
plt.close()

##### eingabe #####
innen = 0.7
aussen = 1
pi = np.pi
num_samples = 10
punkte_plotten = 0
areas_plotten = 0
edges_plotten = 1
#
###### punkte aggregieren #####
points = np.zeros((num_samples,2))

phi = np.random.uniform(0, 2*pi, num_samples)
r = np.random.uniform(innen, aussen, num_samples)

x = r * np.cos(phi)
y = r * np.sin(phi)
#x, y = np.random.uniform(0,1,num_samples), np.random.uniform(0,1,num_samples)
points = np.vstack((x,y)).T

import gudhi

##### rips complex erstellen
def rips(points, epsilon):
    rips_complex = gudhi.RipsComplex(points = points, max_edge_length=epsilon)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    return simplex_tree

if punkte_plotten == 1: plt.plot(points[:,0],points[:,1],'o')

def euler(points, epsilon):
    simplex_tree = rips(points, epsilon)
    a = simplex_tree.get_filtration()
    
    count_areas = 0
    count_edges = 0
    for i in range(np.shape(a)[0]):
        if np.shape(a[i][0])[0] == 3:
            count_areas += 1
            if areas_plotten == 1:
                points_area = np.zeros((4,2))
                points_area[0:3] = points[a[i][0]]
                points_area[3] = points_area[0]
                plt.plot(points_area[:,0],points_area[:,1])
        if np.shape(a[i][0])[0] == 2:
            count_edges += 1 
            if edges_plotten == 1:
                points_edges = np.zeros((2,2))
                points_edges = points[a[i][0]]
                plt.plot(points_edges[:,0],points_edges[:,1])
    
    ecken = num_samples
    kanten = count_edges
    flaechen = count_areas
    
    euler_characteristics = ecken - kanten + flaechen
    
    return euler_characteristics

##### viele epsilons #####
epsilon = np.arange(0,2,0.01)
ergebnisse = np.zeros(np.shape(epsilon)[0])

for i,n in enumerate(epsilon):
    ergebnisse[i] = euler(points,n)

f, ax = plt.subplots()
ax.plot(epsilon,ergebnisse)

ax.set_xlabel("Epsilon")
ax.set_ylabel("Eulercharakteristik")
plt.title('Eulercharakteristik für verschiedene Epsilon')

##### festes epsilon #####
epsilon = 2
euler_characteristics = euler(points, epsilon)
print('Eulercharakteristik: ', euler_characteristics)















#import numpy as np
#import matplotlib.pyplot as plt
#
###### eingabe #####
#innen = 0.7
#aussen = 1
#pi = np.pi
#num_samples = 500
#
###### verarbeitung #####
#points = np.zeros((num_samples,2))
#phi = np.random.uniform(0, 2*pi, num_samples)
#r = np.random.uniform(innen, aussen, num_samples)
#x = r * np.cos(phi)
#y = r * np.sin(phi)
#points = np.vstack((x,y)).T
#
###### ausgabe #####
#fig = plt.figure()
#ax = fig.gca()
#ax.plot(points[:,0],points[:,1],'o')





