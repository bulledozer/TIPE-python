import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

import matplotlib.animation as anim

import copy

from random import randint, random

from src.road import *
from src.utils import *
from src.car import *


#---------PARAMETRES------------------

global g
g = 9.81

# ROUTE

WIDTH = 3 # largeur de la route
N_POINTS = 400 # nombre de points

# MODELISATION

N_SECTORS = 100 # nombre de points de contrôle sur la courbe solution
VMAX = 50 # vitesse maximum

# DESCENTE DE GRADIENT

SCALE = 100 # coefficient du gradient
N_ITER = 1000 # nombre d'itérations

# COSMETIQUE

VERBOSE = True # affiche les infos dans la console



#---------TRAITEMENT ROUTE -----------

track_points = []
N = 0

with open("roads/Monza_centerline.csv", 'r') as f:
    L = f.readlines()
    N = len(L)
    for l in L:
        l1 = l.removesuffix('\n').split(',')
        track_points.append((float(l1[0]), float(l1[1])))


spl = Road(N, track_points, WIDTH, True)

p2 = spl.compute_points2(N_POINTS,2)

POINTS = spl.compute_points2(N_SECTORS, 2)


#-----------RESOLUTION------------

curve_state = [0.5]*(N_SECTORS)
dx = 0.0001

TIMES = []


def points_from_state(state):
    return np.array([POINTS[i][0]*(1-state[i])+POINTS[i][1]*state[i] for i in range(len(state))])

def time_from_state(state):
    controls = np.array(points_from_state(state))

    t = 0

    for i in range(len(state)-2):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        
        theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2)) 

        t += 1/min(np.abs(np.tan(theta/2)), VMAX)
    return t

def gradient_descent(state,scale,times,timef):
    base_time = timef(state)

    times.append(base_time)

    gradient = [0]*N_SECTORS


    for i in range(N_SECTORS):
        state2 = copy.deepcopy(state)
        state2[i] = np.clip(state2[i]+dx, 0,1)

        new_time = timef(state2)

        gradient[i] = new_time-base_time
    
    for i in range(N_SECTORS):
        state[i] -= gradient[i]*scale
        state[i] = np.clip(state[i],0,1)


min_state = curve_state
min_time = float('inf')

for i in range(N_ITER):
    gradient_descent(curve_state, SCALE, TIMES, time_from_state)
    if TIMES[-1] < min_time:
        min_state = curve_state

    if VERBOSE and not i%50:
        print("Iter : ", i, " | Temps : ", TIMES[-1])

sol_points = np.array(points_from_state(min_state))

#-------------AFFICHAGE----------------

f0 = plt.figure()
f1 = plt.figure()

ax0 = f0.add_subplot()
ax0.set_aspect('equal', 'datalim')

plot_points(p2, ax0, False, 'black')
ax0.plot(sol_points[:,0],sol_points[:,1], c='orange', linewidth=3)
ax0.scatter(*sol_points.T, c='r', marker='*', s=60, zorder=2)

ax1,ax2 = f1.subplots(1,2)


ax1.set(xlabel='Itérations', ylabel='Temps')
ax1.plot([i for i in range(N_ITER)], TIMES)

curvatures = []
controls = sol_points


for i in range(len(sol_points)-2):
    L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
    L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
    
    theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2))
    curvatures.append(np.tan(theta/2))

ax2.plot(np.linspace(0,1,len(curvatures)), curvatures)


plt.show()
