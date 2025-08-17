import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from numba import njit

from alive_progress import alive_bar

from src.road import *
from src.utils import *
from src.car import *



@njit(cache=True)
def points_from_state(state, points):
    controls = np.zeros((len(state),2))

    for i in range(len(state)):
        controls[i] = points[i][0]*(1-state[i])+points[i][1]*state[i]

    return controls
    #return np.array([POINTS[i][0]*(1-state[i])+POINTS[i][1]*state[i] for i in range(len(state))])

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
def dist_from_state(state, points):
    controls = points_from_state(state, points)

    d = 0

    for i in range(len(state)-1):
        d += np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)

    return d

@njit(cache=True)
def gradient_descent(state,scale,times,timef,points, dx):
    base_time = timef(state, points)

    times.append(base_time)

    gradient = np.zeros(len(state))


    for i in range(len(state)):
        state2 = np.zeros(len(state))

        for j in range(len(state)):
            state2[j] = state[j]

        state2[i] = state2[i]+dx

        new_time = timef(state2, points)

        gradient[i] = new_time-base_time
    
    for i in range(len(state)):
        state[i] = state[i]-(gradient[i]*scale)
        state[i] = min(max(state[i],0.0),1.0)



if __name__ == "__main__":

#-----------------------------------------------------------
#---------PARAMETRES----------------------------------------
#-----------------------------------------------------------

    g = 9.81

    # ROUTE

    WIDTH = 2.2 # largeur de la route
    N_POINTS = 800 # nombre de points

    # MODELISATION

    N_SECTORS = 230 # nombre de points de contrôle sur la courbe solution

    # DESCENTE DE GRADIENT

    SCALE = 30 # coefficient du gradient
    N_ITER = 2000 # nombre d'itérations

    # COSMETIQUE

    VERBOSE = False # affiche les infos dans la console
    SHOW_LINE = True # attention : Montréal et Shanhai n'ont pas de trajectoire idéale
    ROAD_NAME = "Nuerburgring"



#-----------------------------------------------------------
#---------TRAITEMENT ROUTE ---------------------------------
#-----------------------------------------------------------

    track_points = pd.read_csv("roads/" + ROAD_NAME + "_centerline.csv").values[:,:2]
    N = track_points.shape[0]

    spl = Road(N, track_points, WIDTH, True)

    POINTS = spl.compute_points2(N_SECTORS, 2)
    VIS_POINTS = spl.compute_points2(N_POINTS, 2)

#-----------------------------------------------------------
#-----------RESOLUTION--------------------------------------
#-----------------------------------------------------------

    dx = 0.0001
    curve_state = [0.5]*(N_SECTORS)

    TIMES = [0.0]

    min_state = curve_state
    min_time = float('inf')

    with alive_bar(N_ITER, bar="fish") as bar:
        for i in range(N_ITER):
            gradient_descent(curve_state, SCALE, TIMES, time_from_state, POINTS, dx)
            
            if TIMES[-1] < min_time:
                min_state = curve_state

            if VERBOSE and not i%50:
                print("Iter : ", i, " | Temps : ", TIMES[-1])

            bar()

    sol_points = points_from_state(min_state, POINTS)

#-----------------------------------------------------------
#-------------AFFICHAGE-------------------------------------
#-----------------------------------------------------------

    f0 = plt.figure()
    f1 = plt.figure()

    ax0 = f0.add_subplot()
    ax0.set_aspect('equal', 'datalim')

    plot_points(VIS_POINTS, ax0, False, 'black')

    if SHOW_LINE:
        line_points = pd.read_csv("lines/"+ROAD_NAME+"_raceline.csv", sep=";").values[:,1:3]
        ax0.plot(line_points[:,0],line_points[:,1], c='red', linewidth=3, label="trajectoire idéale", linestyle="--")

    ax0.plot(sol_points[:,0],sol_points[:,1], c='orange', linewidth=3, label="notre trajectoire")
    #ax0.scatter(*sol_points.T, c='r', marker='*', s=60, zorder=2)


    ax0.legend()

    ax1,ax2 = f1.subplots(1,2)


    ax1.set(xlabel='Itérations', ylabel='Temps')
    ax1.plot([i for i in range(N_ITER)], TIMES[1:])

    curvatures = []
    controls = sol_points


    for i in range(len(sol_points)-2):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        
        theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2))
        curvatures.append(np.tan(theta/2))

    ax2.plot(np.linspace(0,1,len(curvatures)), curvatures)


    plt.show()