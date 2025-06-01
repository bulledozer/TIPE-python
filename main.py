import pylab as pl

import copy

from random import randint

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

p2 = spl.compute_points(N_POINTS,2)
plot_points(p2)

N_SECTORS = 50

POINTS = spl.compute_points(N_SECTORS, 2)

curve_state = [0.5]*(N_SECTORS)
dx = 0.01
RES = 50

VMAX = 20

def spline_from_state(state):
    return Spline(N_SECTORS+2, [POINTS[0][0]*(1-state[0])+ POINTS[0][1]*state[0]] + 
                  [POINTS[j][0]*(1-state[j])+ POINTS[j][1]*state[j] for j in range(N_SECTORS)] + 
                  [POINTS[-1][0]*(1-state[-1])+ POINTS[-1][1]*state[-1]])

def gradient_descent(state,scale):
    base_spline = spline_from_state(state)
    base_time = base_spline.compute_time(RES, VMAX)

    gradient = [0]*N_SECTORS

    for i in range(N_SECTORS):
        state2 = copy.deepcopy(state)
        state2[i] = pl.clip(state2[i]+dx, 0,1)

        new_spline = spline_from_state(state2)
        new_time = new_spline.compute_time(RES, VMAX)

        gradient[i] = new_time-base_time
    
    for i in range(N_SECTORS):
        state[i] -= gradient[i]
        state[i] = pl.clip(state[i],0,1)

SCALE = 10
N_ITER = 500

for i in range(N_ITER):
    gradient_descent(curve_state, SCALE)

sol_spline = spline_from_state(curve_state)


VIS_RES = 1000

points = np.array([sol_spline.compute_point(i/(VIS_RES)) for i in range(VIS_RES+1)])

for i in range(len(points)-1):
    pl.plot((points[i][0],points[i+1][0]), (points[i][1],points[i+1][1]), c='r', linewidth=3)
for i in range(N_SECTORS+1):
    pl.plot(sol_spline.compute_point(i/N_SECTORS)[0], sol_spline.compute_point(i/N_SECTORS)[1], 'ro')


pl.show()