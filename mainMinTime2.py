import numpy as np
import numpy.polynomial.legendre as poly
import matplotlib.pyplot as plt
import pandas as pd

import scipy.interpolate as interp
from scipy.integrate import simpson

import cma

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
def speeds_from_state(state, points):
    controls = points_from_state(state, points)

    speeds = np.zeros(len(state)-2)
    ds = np.zeros(len(state)-2)

    for i in range(len(state)-2):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        
        theta = np.arccos(np.dot(controls[i]-controls[i+1], controls[i+2]-controls[i+1])/(L1*L2)) 

        speeds[i] = np.sqrt(np.abs(np.tan(theta/2))+0.00001)
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



if __name__ == "__main__":

#-----------------------------------------------------------
#---------PARAMETRES----------------------------------------
#-----------------------------------------------------------
    g = 9.81
    # ROUTE

    WIDTH = 2.2 # largeur de la route
    N_POINTS = 800 # nombre de points
    mu = 1.3

    #VOITURE
    ACCEL = 1
    DECEL = 1.3

    # MODELISATION

    N_SECTORS = 130 # nombre de points de contrôle sur la courbe solution
    D = 3

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

#-----------------------------------------------------------
#-----------RESOLUTION--------------------------------------
#-----------------------------------------------------------    

    def obj_func(state):
        id_speed_prof, ds = speeds_from_state(state,POINTS)
        speed_prof = gen_speed_profile(id_speed_prof, ds, 5, ACCEL, DECEL)
        s = np.cumsum(ds)
        s = np.insert(s[:-1], 0, 0)
        speed_spl = interp.make_interp_spline(s, 1/speed_prof, k=4)
        
        #sample_pts, weights = poly.leggauss(D)
        
        # t = 0
        # for i in range(len(speed_prof)):
        #     t += 1/speed_prof[i]
        
        #return simpson(1/speed_prof, x=s)
        return float(speed_spl.integrate(0, s[-1]))
        #return t

    es = cma.CMAEvolutionStrategy([0.5]*N_SECTORS, 0.8, {'bounds' : [0,1], 'maxiter':10000})
    es.optimize(obj_func)
     # print(es.result)
    sol_state = es.result[0]
    sol_points = points_from_state(sol_state, POINTS)

    sol_id_speed_prof, ds = speeds_from_state(sol_state, POINTS)
    sol_speed_prof = gen_speed_profile(sol_id_speed_prof, ds, 0.1, ACCEL,DECEL)
    s = np.cumsum(ds)

    speed_spl = interp.make_interp_spline(s, 1/sol_speed_prof, k=4)
    # print("min time : ", obj_func(sol_state))
    # print(sol_id_speed_prof)

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
    
    ax1 = f1.add_subplot()
    #ax1,ax2 = f1.subplots(2,1)
    ax1.plot(s, sol_id_speed_prof, label="profil théorique")
    ax1.plot(s, sol_speed_prof, label="profil réel")
    
    ax1.legend()

    # ax2.plot(s, 1/sol_speed_prof, label= "profil réel")
    # ax2.plot(np.linspace(0, s[-1], 1500), speed_spl(np.linspace(0, s[-1], 1500)) + 1, label="profil interpolé")
    #
    # ax2.legend()
    # if VEL_PROFILE:
    #
    #     ax1,ax2 = f1.subplots(1,2)
    #
    #
    #     ax1.set(xlabel='Itérations', ylabel='Temps')
    #     ax1.plot([i for i in range(N_ITER)], TIMES[1:])
    #
    #     curvatures = []
    #     controls = sol_points
    #
    #     car = Car(10, -15, 1500, 10)
    #     speeds,s = car.compute_velocity_profile(sol_points, 1, 250)
    #
    #     #for i in range(len(sol_points)-2):
    #     #    L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
    #     #    L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
    #     #    
    #     #    theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2))
    #     #    curvatures.append(np.tan(theta/2))
    #
    #     ax2.plot(s,speeds)
    #
    # else:
    #     ax1 = f1.add_subplot()
    #     ax1.set(xlabel='Itérations', ylabel='Temps')
    #     ax1.plot([i for i in range(N_ITER)], TIMES[1:])

    plt.show()
