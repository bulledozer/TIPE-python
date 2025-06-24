import numpy as np
from scipy.interpolate import *
from src.utils import *
from bisect import bisect_right
import splines


class Spline:
    def __init__(self,N, M = []):
        self.N = N
        self.lengths = []
        #self.points = SX.sym("P",N,2) if M == SX(0) else M
        self.points = np.array(M)
        

    def change_point(self, i, p):
        self.points[i,0] = p[0]
        self.points[i,1] = p[1]

    def compute_point(self, t):
        seg = int(t*(self.N-3))

        A = self.points[seg%self.N]  
        B = self.points[(seg+1)%self.N]
        C = self.points[(seg+2)%self.N]
        D = self.points[(seg+3)%self.N]
        
        nt = (t%(1/(self.N-3)))*(self.N-3)
        
        return 0.5*np.matmul(np.matmul(np.array([1,nt,nt**2,nt**3]),np.array([[0,2,0,0],[-1,0,1,0],[2,-5,4,-1],[-1,3,-3,1]])),np.array([A,B,C,D]))

    def compute_curvature(self, t):
        seg = int(t*(self.N-3))

        A = self.points[seg%self.N]  
        B = self.points[(seg+1)%self.N]
        C = self.points[(seg+2)%self.N]
        D = self.points[(seg+3)%self.N]
        
        nt = (t%(1/(self.N-3)))*(self.N-3)

        dP = 0.5*np.matmul(np.matmul(np.array([0,1,2*nt,3*nt**2]),np.array([[0,2,0,0],[-1,0,1,0],[2,-5,4,-1],[-1,3,-3,1]])),np.array([A,B,C,D]))
        d2P = 0.5*np.matmul(np.matmul(np.array([0,0,2,6*nt]),np.array([[0,2,0,0],[-1,0,1,0],[2,-5,4,-1],[-1,3,-3,1]])),np.array([A,B,C,D]))

        return ((dP[0]**2+dP[1]**2)**(1.5))/abs(dP[0]*d2P[1]-d2P[0]*dP[1])


class Road(Spline):
    def __init__(self,N, M, W):
        super().__init__(N,M)
        self.W = W

    def compute_width(self, t, R, seg = -1):
        P = []
        M,M2 = None,None
        if seg == -1:
            M = self.compute_point(t)
            M2 = self.compute_point(t+0.001)
        else:
            M = self.compute_point_seg(t, seg)
            M2 = self.compute_point_seg(t+0.001, seg)
        
        for i in range(R):
            dir = (np.cross((M2-M)+[0], [0,0,1]))
            P.append(M+(dir/np.linalg.norm(dir))[:2]*self.W*((i/R)*2-1))
        return np.array(P)
    
    def compute_points2(self,n,m):
        spl = splines.CatmullRom(self.points, alpha=1)
        times = np.linspace(spl.grid[0], spl.grid[-1],n+1)
        points = spl.evaluate(times)
        P = []
        for i in range(n):
            M = points[i]
            M2 = points[i+1]
            R = []
            for i in range(m):
                dir = (np.cross((M2-M)+[0], [0,0,1]))
                R.append(M+(dir/np.linalg.norm(dir))[:2]*self.W*((i/m)*2-1))
            P.append(R)
        return np.array(P)

