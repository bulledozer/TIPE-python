import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

import casadi as ca

from src.road import *
from src.utils import *
from src.car import *




def points_from_state(state, points):
    controls = np.zeros((len(state),2))

    for i in range(len(state)):
        controls[i] = points[i][0]*(1-state[i])+points[i][1]*state[i]

    return controls

def speeds_from_state(state, points):
    controls = points_from_state(state, points)

    speeds,ds = [],[]

    for i in range(len(state)-2):
        L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        
        theta = np.arccos(np.dot(controls[i]-controls[i+1], controls[i+2]-controls[i+1])/(L1*L2)) 

        speeds.append(1/np.abs(np.tan(theta/2)))
        ds.append(L2)
    return speeds,ds


def gen_speed_profile(speeds, ds, start_speed, accel, decel):
    speed_prof = np.copy(speeds)
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
    ACCEL = 5
    DECEL = 7

    # MODELISATION

    N_SECTORS = 130 # nombre de points de contrôle sur la courbe solution

    # DESCENTE DE GRADIENT

    SCALE = 30 # coefficient du gradient
    N_ITER = 1000 # nombre d'itérations

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
   
    opti = ca.Opti()

    X = opti.variable(N_SECTORS)

    controls = np.zeros((N_SECTORS,2))

    for i in range(N_SECTORS):
        controls[i] = POINTS[i][0]*(1-X[i])+POINTS[i][1]*X[i]

    speeds, ds = speeds_from_state(X, POINTS)
    speed_prof = gen_speed_profile(speeds, ds, 0, ACCEL, DECEL)
    # for i in range(N_SECTORS):
    #     controls[i] = POINTS[i][0]*(1-X[i]) + POINTS[i][1]*X[i]

    T = opti.variable()
    for i in range(len(speed_prof)):
        T += ds[i]/speed_prof[i]

    opti.minimize(T)
    opti.subject_to(0<=X)
    opti.subject_to(1>=X)

    opti.solver('ipopt')
    sol = opti.solve()
    













