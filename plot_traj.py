import matplotlib.pyplot as plt

import pandas as pd

from src.road import *
from src.utils import *


FILES = ["out/minTime1_3_pts.csv", "out/minTime1_pts.csv", "out/minTime0_5_pts.csv"]
DESC = ["µ = 1.3", "µ = 1.0","µ = 0.5"]
COL = ["orange", "purple", "red"]
STYLES = ["solid", "dotted", "dashdot"]

ROAD_NAME = "Monza"
WIDTH = 2.2
N_POINTS = 800


track_points = pd.read_csv("roads/" + ROAD_NAME + "_centerline.csv").values[:,:2]
N = track_points.shape[0]

spl = Road(N, track_points, WIDTH, True)
VIS_POINTS = spl.compute_points2(N_POINTS, 2)

f0 = plt.figure()
ax0 = f0.add_subplot()
ax0.set_aspect('equal', 'datalim')
plot_points(VIS_POINTS, ax0, False, "black")

for f,d,c,s in zip(FILES,DESC,COL,STYLES):
    points = pd.read_csv(f).to_numpy()
    ax0.plot(points[:,1],points[:,2], c=c, linewidth="3", label=d, linestyle=s)

ax0.legend()

plt.show()

