import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

import cma

import copy

from random import randint, random

from road import *
from utils import *


#---------COSMETIQUE (afficher la route) -----------

track_points = []
N = 0

with open("road.txt", 'r') as f:
    L = f.readlines()
    N = len(L)
    for l in L:
        l1 = l.removesuffix('\n').split(',')
        track_points.append((float(l1[0]), float(l1[1])))

#P = []
WIDTH = 15
spl = Road(N, track_points, WIDTH)

#points = np.array([spl.compute_point(i/R) for i in range(R+1)])
#curvature = np.array([1-1/(1+0.005*spl.compute_curvature(i/R)) for i in range(R+1)])

N_POINTS = 700

p2 = spl.compute_points2(N_POINTS,2)

N_SECTORS = 50

POINTS = spl.compute_points2(N_SECTORS, 2)

curve_state = [0.5]*(N_SECTORS)
dx = 0.01

VMAX = 2500

TIMES = []

def spline_from_state(state):
    return Spline(N_SECTORS+2, [POINTS[0][0]*(1-state[0])+ POINTS[0][1]*state[0]] + 
                  [POINTS[j][0]*(1-state[j])+ POINTS[j][1]*state[j] for j in range(N_SECTORS)] + 
                  [POINTS[-1][0]*(1-state[-1])+ POINTS[-1][1]*state[-1]])


def points_from_state(state):
    return np.array([POINTS[i][0]*(1-state[i])+POINTS[i][1]*state[i] for i in range(len(state))])

def time_from_state2(state):
    for i in state:
        if i < 0 or i > 1:
            return 1<<31
        
    controls = np.array(points_from_state(state))

    t = 0

    for i in range(len(state)-3):
        L1 = np.linalg.norm(controls[i+1]-controls[i+2])
        L2 = np.linalg.norm(controls[i+1]-controls[i])
        L3 = np.linalg.norm(controls[i+3]-controls[i+2])
        
        theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2)) 
        phi = np.arccos(np.dot(controls[i+1,:]-controls[i+2,:], controls[i+3,:]-controls[i+2,:])/(L1*L3))
        alpha = ((phi-theta))

        #T2 += (log(phi/theta))/(alpha)
        t += 2*(np.sqrt(phi)-np.sqrt(theta))/alpha
    return t

def time_from_state3(state):
    for i in state:
        if i < 0 or i > 1:
            return 1000000000000000000
        
    controls = np.array(points_from_state(state))

    t = 0

    for i in range(len(state)-1):
        t+= np.linalg.norm(controls[i]-controls[i+1])

    return t

def curvatures(state):
    controls = points_from_state(state)

    dx = np.gradient(controls[:,0])
    dy = np.gradient(controls[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    return ((dx*dx+dy*dy)**1.5)/np.abs(dx*d2y-d2x*dy)

def time_from_state4(state):
    for i in state:
        if i < 0 or i > 1:
            return 1000000000000000000000
    
    controls = points_from_state(state)
    t = 0

    for i in range(len(state)-3):
        L1 = np.linalg.norm(controls[i+1]-controls[i+2])
        L2 = np.linalg.norm(controls[i+1]-controls[i])

        F = np.linalg.norm(controls[i+2]-controls[i])
        
        theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2)) 

        #T2 += (log(phi/theta))/(alpha)
        #t += 1/(L1*np.tan(theta/2))
        t += 2*np.sin(theta)/F
    return t

def t(state):
    for i in state:
        if i < 0 or i > 1:
            return 1000000000000000000000
        
    return np.sum(1/curvatures(state))


es = cma.CMAEvolutionStrategy([0.5]*N_SECTORS, 0.1)
es.optimize(t, min_iterations=1000, iterations=3000)

VIS_RES = 1000

sol_points = points_from_state(es.result[0])

fig,(ax1,ax2) = plt.subplots(1,2)

plot_points(p2,ax2)

ax1.plot(np.linspace(0,1,N_SECTORS), np.log(np.log(curvatures(es.result[0]))))
ax1.set(xlabel='Itérations', ylabel='Temps')
#ax2.plot(points[:,0],points[:,1], c='r', linewidth=3)

ax2.plot(sol_points[:,0],sol_points[:,1], c='r', linewidth=3)
ax2.scatter(*sol_points.T, c=np.log(np.log(curvatures(es.result[0]))), marker='*', s=60, zorder=2)

plt.show()
