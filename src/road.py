import numpy as np
from src.utils import *

import splines


class Spline:
    def __init__(self,N, M = []):
        self.N = N
        self.lengths = []
        #self.points = SX.sym("P",N,2) if M == SX(0) else M
        self.points = np.array(M)

class Road(Spline):
    def __init__(self,N, M, W, closed):
        super().__init__(N,M)
        self.W = W
        self.closed = closed
    
    def compute_points2(self,n,m):
        spl = splines.CatmullRom(self.points, alpha=1, endconditions='closed' if self.closed else 'natural')
        times = np.concatenate([np.linspace(spl.grid[0], spl.grid[-1],n),[0.001]]) if self.closed else np.linspace(spl.grid[0], spl.grid[-1],n+1)
        points = spl.evaluate(times)
        P = []
        for i in range(n):
            M = points[i]
            M2 = points[i+1]
            R = []
            for i in range(m):
                dir = (np.cross((M2-M)+[0], [0,0,1]))
                R.append(M+(dir/np.linalg.norm(dir))[:2]*self.W*((i/(m-1))*2-1))
            P.append(R)
        if self.closed:
            P[-1] = P[0]
        return np.array(P)

