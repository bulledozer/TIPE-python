from mainMinTime2 import solve

from src.road import *
from src.utils import *

import pandas as pd
import numpy as np

#-----------------------------------------------------------
#---------PARAMETRES----------------------------------------
#-----------------------------------------------------------

d_params = read_json("opts.json")

G = d_params["G"]
MU = 0.5
# ROUTE

WIDTH = d_params["WIDTH"] # largeur de la route
N_POINTS = d_params["N_POINTS"] # nombre de points

#VOITURE
ACCEL = d_params["ACCEL"]
DECEL = d_params["DECEL"]

# MODELISATION

N_SECTORS = d_params["N_SECTORS"] # nombre de points de contrôle sur la courbe solution

# COSMETIQUE

ROAD_NAME = d_params["ROAD_NAME"]
START_SPEED = d_params["START_SPEED"]


N_ITER = d_params["N_ITER"]

# COSMETIQUE

ROAD_NAME = d_params["ROAD_NAME"]
SAVE_FILE = "minTime0_5"

#-----------------------------------------------------------
#---------TRAITEMENT ROUTE ---------------------------------
#-----------------------------------------------------------

track_points = pd.read_csv("roads/" + ROAD_NAME + "_centerline.csv").values[:,:2]
N = track_points.shape[0]

spl = Road(N, track_points, WIDTH, True)

POINTS = spl.compute_points2(N_SECTORS, 2)

#-----------------------------------------------------------
#-----------RESOLUTION--------------------------------------
#-----------------------------------------------------------    


data = solve(N_ITER, POINTS, ACCEL, DECEL, START_SPEED, N_SECTORS, MU, G)

DF_pts = pd.DataFrame(data["points"])
DF_sp = pd.DataFrame(np.vstack((data["s"],data["speed_prof"])).T)

DF_pts.to_csv("out/"+SAVE_FILE+"_pts.csv")
DF_sp.to_csv("out/"+SAVE_FILE+"_sp.csv")

print("Finished.")
