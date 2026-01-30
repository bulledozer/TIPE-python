import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import minimize

from numba import njit

from src.road import *
from src.utils import *

@njit(cache=True)
def points_from_state(state, points):
    controls = np.zeros((len(state),2))

    for i in range(len(state)):
        controls[i] = points[i][0]*(1-state[i])+points[i][1]*state[i]

    return controls

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

def solve(N_iter, points, accel, decel, start_speed, n_sectors, mu, g):
    def obj_func(state):
        state = np.clip(state, 0,1)
        id_speed_prof, ds = speeds_from_state(state,points,mu,g)
        speed_prof = gen_speed_profile(id_speed_prof, ds, start_speed,accel,decel)
        s = np.cumsum(ds)
        s = np.insert(s[:-1], 0, 0)

        interp = Akima1DInterpolator(s,1/speed_prof, method="makima")

        return float(interp.integrate(0, s[-1]))

    res = minimize(obj_func, [0.5]*n_sectors, method="BFGS", options={"maxiter":N_iter})
    print(res.success, ",", res.nit, ",",res.message)
    
    sol_state = np.clip(res.x,0,1)
    time = res.fun
    sol_points = points_from_state(sol_state, points)

    sol_id_speed_prof, ds = speeds_from_state(sol_state, points,mu,g)
    sol_speed_prof = gen_speed_profile(sol_id_speed_prof, ds, start_speed, accel, decel)
    s = np.cumsum(ds)
    
    return {"state":sol_state, "points":sol_points, "speed_prof":sol_speed_prof, "s":s, "time":time, "id_speed_prof":sol_id_speed_prof}

if __name__ == "__main__":

#-----------------------------------------------------------
#---------PARAMETRES----------------------------------------
#-----------------------------------------------------------
    d_params = read_json("opts.json")
    G = d_params["G"]
    MU = 1.3
    # ROUTE

    WIDTH = d_params["WIDTH"] # largeur de la route
    N_POINTS = d_params["N_POINTS"] # nombre de points

    #VOITURE
    ACCEL = d_params["ACCEL"]
    DECEL = d_params["DECEL"]

    N_ITER = d_params["N_ITER"]
    # MODELISATION

    N_SECTORS = d_params["N_SECTORS"] # nombre de points de contrôle sur la courbe solution

    # COSMETIQUE

    ROAD_NAME = d_params["ROAD_NAME"]

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

    data = solve(N_ITER, POINTS, ACCEL, DECEL,0.1, N_SECTORS, MU, G)
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

    plot_points(VIS_POINTS, ax0, False, 'black')

    ax0.plot(sol_points[:,0],sol_points[:,1], c='orange', linewidth=3, label="notre trajectoire")
    #ax0.scatter(*sol_points.T, c='r', marker='*', s=60, zorder=2)

    ax0.legend()
    
    ax1 = f1.add_subplot()
    #ax1,ax2 = f1.subplots(2,1)
    ax1.plot(s, sol_id_speed_prof, label="profil théorique")
    ax1.plot(s, sol_speed_prof, label="profil réel")
    
    ax1.legend()

    plt.show()
