import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyswarm as ps

from numba import njit

from alive_progress import alive_bar

from src.road import *
from src.utils import *

@njit(cache=True)
def points_from_state(state, points):
    controls = np.zeros((len(state),2))

    for i in range(len(state)):
        controls[i] = points[i][0]*(1-state[i])+points[i][1]*state[i]

    return controls

@njit(cache=True)
def time_from_state(state, points):
    controls = points_from_state(state, points)

    t = 0

    for i in range(len(state)-2):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        
        theta = np.arccos(np.dot(controls[i]-controls[i+1], controls[i+2]-controls[i+1])/(L1*L2)) 

        t += 1/np.abs(np.tan(theta/2))
    return t

@njit(cache=True)
def time_from_state2(state,points, mu, g):
    controls = points_from_state(state, points)

    t = 0.0
    for i in range(len(state)-2):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)

        theta = np.arccos(np.dot(controls[i]-controls[i+1], controls[i+2]-controls[i+1])/(L1*L2))

        t += L1/np.sqrt(np.tan(theta/2)*mu*g)
    return t

@njit(cache=True)
def speeds_from_state(state, points, mu, g):
    controls = points_from_state(state, points)

    speeds = np.zeros(len(state)-2)
    ds = np.zeros(len(state)-2)

    for i in range(len(state)-2):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        
        theta = np.arccos(np.dot(controls[i]-controls[i+1], controls[i+2]-controls[i+1])/(L1*L2)) 

        speeds[i] = np.sqrt(np.abs(np.tan(theta/2))*mu*g+0.00001)
        ds[i] = (L2)
    return speeds,ds

@njit(cache=True)
def gen_speed_profile(speeds, ds, start_speed, accel, decel):
    speed_prof = np.zeros(len(speeds))
    for i in range(len(speeds)):
        if np.isfinite(speeds[i]):
            speed_prof[i] = speeds[i]
        else:
            speed_prof[i] = 10e10

    speed_prof[0] = start_speed 
    

    for i in range(len(speed_prof)-1):
        nsp = np.sqrt(speed_prof[i]**2 + 2*accel*ds[i])
        speed_prof[i+1] = min(speed_prof[i+1], nsp)

    for i in range(len(speed_prof)-1):
        nsp = np.sqrt(speed_prof[len(speed_prof)-i-1]**2 + 2*ds[len(speed_prof)-i-2]*decel)
        speed_prof[len(speed_prof)-i-2] = min(speed_prof[len(speed_prof)-i-2], nsp)

    return speed_prof

def solve(N_iter, swarm_size, points, w, phip, phig, n_sectors, start_speed, accel, decel, times):
    res, time = ps.pso(lambda x : time_from_state(x, points), [0]*n_sectors, [1]*n_sectors, maxiter=2000, minfunc=1e-20, minstep=1e-20, swarmsize=1000)
    
    sol_state = np.clip(res,0,1)
    sol_points = points_from_state(sol_state, points)

    sol_id_speed_prof, ds = speeds_from_state(sol_state, points,mu,g)
    sol_speed_prof = gen_speed_profile(sol_id_speed_prof, ds, start_speed, accel, decel)
    s = np.cumsum(ds)
    
    return {"state":sol_state, "points":sol_points, "speed_prof":sol_speed_prof, "s":s, "time":time, "id_speed_prof":sol_id_speed_prof}

def testPSO(points, wR, phipR, phigR, n_sectors, res):
    w = np.linspace(-wR, wR, res)
    phip = np.linspace(-phipR, phipR, res)
    phig = np.linspace(-phigR, phigR, res)

    min_time = np.inf
    min_conf = (0,0,0)
    for a in w:
        for b in phip:
            for c in phig:
                t = ps.pso(lambda x : time_from_state(x, points), [0]*n_sectors, [1]*n_sectors)[1]
                if t >= min_time:
                    continue
                min_time = t
                min_conf = (a,b,c)
    return min_conf

if __name__ == "__main__":

#-----------------------------------------------------------
#---------PARAMETRES----------------------------------------
#-----------------------------------------------------------
    g = 9.81
    # ROUTE

    WIDTH = 2.2 # largeur de la route
    N_POINTS = 800 # nombre de points
    mu = 1.3

    # MODELISATION

    N_SECTORS = 60 # nombre de points de contrôle sur la courbe solution

    # PSO

    W = -0.3
    PHIP = 1.16
    PHIG = -0.94
    SWARM_SIZE = 200

    START_SPEED = 0.1
    ACCEL = 5
    DECEL = 7

    N_ITER = 500 # nombre d'itérations

    # COSMETIQUE

    VERBOSE = False # affiche les infos dans la console
    SHOW_LINE = False # attention : Montréal et Shanhai n'ont pas de trajectoire idéale
    ROAD_NAME = "Monza"

    VEL_PROFILE = False 

#-----------------------------------------------------------
#---------TRAITEMENT ROUTE ---------------------------------
#-----------------------------------------------------------
    track_points = pd.read_csv("roads/" + ROAD_NAME + "_centerline.csv").values[:,:2]
    N = track_points.shape[0]

    spl = Road(N, track_points, WIDTH, True)

    POINTS = spl.compute_points2(N_SECTORS, 2)
    VIS_POINTS = spl.compute_points2(N_POINTS, 2)
    track_points.shape
#-----------------------------------------------------------
#-----------RESOLUTION--------------------------------------
#-----------------------------------------------------------

    dx = 0.0001
    curve_state = [0.5]*(N_SECTORS)

    TIMES = [0.0]

    data = solve(N_ITER, SWARM_SIZE, POINTS, W, PHIP, PHIG, N_SECTORS, START_SPEED, ACCEL, DECEL, TIMES)
    sol_points = data["points"]
    s = data["s"]
    sol_speed_prof = data["speed_prof"]
    sol_id_speed_prof = data["id_speed_prof"]

#-----------------------------------------------------------
#-------------AFFICHAGE-------------------------------------
#-----------------------------------------------------------

    f0 = plt.figure()
    f1 = plt.figure()

    ax0 = f0.add_subplot()
    ax0.set_aspect('equal', 'datalim')

    plot_points(POINTS, ax0, True, 'white', 'red')
    plot_points(VIS_POINTS, ax0, False, 'black')
    
    if SHOW_LINE:
        line_points = pd.read_csv("lines/"+ROAD_NAME+"_raceline.csv", sep=";").values[:,1:3]
        ax0.plot(line_points[:,0],line_points[:,1], c='red', linewidth=3, label="trajectoire idéale", linestyle="--")

    ax0.plot(sol_points[:,0],sol_points[:,1], c='orange', linewidth=3, label="notre trajectoire")
    #ax0.scatter(*sol_points.T, c='r', marker='*', s=60, zorder=2)


    ax0.legend()

    plt.show()
