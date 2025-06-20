import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

import copy

from random import randint, random

from road import *
from utils import *


#---------PARAMETRES------------------

# ROUTE
#PIPI
WIDTH = 15 # largeur de la route
N_POINTS = 200 # nombre de points

# MODELISATION

N_SECTORS = 50 # nombre de points de contrôle sur la courbe solution
VMAX = 2500 # vitesse maximum

# DESCENTE DE GRADIENT

SCALE = 100 # coefficient du gradient
N_ITER = 1000 # nombre d'itérations

# COSMETIQUE

VERBOSE = True # affiche les infos dans la console



#---------TRAITEMENT ROUTE -----------

track_points = []
N = 0

with open("road.txt", 'r') as f:
    L = f.readlines()
    N = len(L)
    for l in L:
        l1 = l.removesuffix('\n').split(',')
        track_points.append((float(l1[0]), float(l1[1])))


spl = Road(N, track_points, WIDTH)

p2 = spl.compute_points2(N_POINTS,2)

POINTS = spl.compute_points2(N_SECTORS, 2)


#-----------RESOLUTION------------

curve_state = [0.5]*(N_SECTORS)
dx = 0.001

TIMES = []

def spline_from_state(state):
    return Spline(N_SECTORS+2, [POINTS[0][0]*(1-state[0])+ POINTS[0][1]*state[0]] + 
                  [POINTS[j][0]*(1-state[j])+ POINTS[j][1]*state[j] for j in range(N_SECTORS)] + 
                  [POINTS[-1][0]*(1-state[-1])+ POINTS[-1][1]*state[-1]])

def points_from_state(state):
    return [POINTS[i][0]*(1-state[i])+POINTS[i][1]*state[i] for i in range(len(state))]

def time_from_state2(state):
    controls = np.array(points_from_state(state))

    t = 0

    for i in range(len(state)-3):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        L3 = np.sqrt((controls[i+3,0]-controls[i+2,0])**2 + (controls[i+3,1]-controls[i+2,1])**2)
        
        theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2)) 
        phi = np.arccos(np.dot(controls[i+1,:]-controls[i+2,:], controls[i+3,:]-controls[i+2,:])/(L1*L3))
        alpha = ((phi-theta))

        #T2 += (log(phi/theta))/(alpha)
        t += 2*(np.sqrt(phi)-np.sqrt(theta))/alpha
    return t

def gradient_descent(state,scale,times):
    base_time = time_from_state2(state)

    times.append(base_time)

    gradient = [0]*N_SECTORS


    for i in range(N_SECTORS):
        state2 = copy.deepcopy(state)
        state2[i] = np.clip(state2[i]+dx, 0,1)

        new_time = time_from_state2(state2)

        gradient[i] = new_time-base_time
    
    for i in range(N_SECTORS):
        state[i] -= gradient[i]*scale
        state[i] = np.clip(state[i],0,1)


 
for i in range(N_ITER):
    gradient_descent(curve_state, SCALE, TIMES)
    if VERBOSE:
        print("Iter : ", i, " | Temps : ", TIMES[-1])

sol_points = np.array(points_from_state(curve_state))


#-------------AFFICHAGE----------------


fig,(ax1,ax2) = plt.subplots(1,2)

plot_points(p2,ax2)

ax1.set(xlabel='Itérations', ylabel='Temps')
ax1.plot([i for i in range(N_ITER)], TIMES)

ax2.plot(sol_points[:,0],sol_points[:,1], c='orange', linewidth=3)
ax2.plot(*sol_points.T, c='r', marker='*', markersize=6)


plt.show()
