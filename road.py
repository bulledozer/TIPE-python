import pylab as pl
import numpy as np
from scipy.interpolate import *
from utils import *

class Spline:
    def __init__(self,N, M = []):
        self.N = N
        #self.points = SX.sym("P",N,2) if M == SX(0) else M
        self.points = np.array(M) if M != [] else np.zeros((N,2))
        self.lengths = [0]*(N-3)

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

    def compute_time(self, R, vmax):
        t = 0
        for i in range(R):
            p0 = self.compute_point(i/R)
            p1 = self.compute_point((i+1)/R)

            dst = pl.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

            t += dst/pl.sqrt(min(self.compute_curvature(i/R), vmax))
        return t
    
    def compute_lengthst(self, R):
        self.lengths = [0]*(self.N-3)
        for i in range(self.N-3):
            for j in range(R):
                p0 = self.compute_point(i/(self.N-3) + j/(R*(self.N-3)))
                p1 = self.compute_point(i/(self.N-3) + (j+1)/(R*(self.N-3)))

                dst = pl.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
                self.lengths[i] += dst

    def compute_point_eq(self, t):
        i = 0
        totlength = sum(self.lengths)
        partialsum = 0
        while i < self.N - 3:
            if t > (partialsum + self.lengths[i])/totlength:
                break
            partialsum += self.lengths[i]
            i += 1
        
        return self.compute_point(pl.interp(inv_lerp(partialsum/totlength, (partialsum + self.lengths[i])/totlength, t)))



class Road(Spline):
    def __init__(self,N, M, W):
        super().__init__(N,M)
        self.W = W

    def compute_width(self, t, R):
        P = []
        M = self.compute_point(t)
        M2 = self.compute_point(t+0.001)
        for i in range(R):
            dir = (np.cross((M2-M)+[0], [0,0,1]))
            P.append(M+(dir/np.linalg.norm(dir))[:2]*self.W*((i/R)*2-1))
        return np.array(P)
    
    def lerp_line(self, t, s):
        l1,l2 = self.compute_width(t,2)
        return (1-s)*l1+s*l2
    
    def compute_points(self, n,m):
        P = []
        for i in range(n):
            P.append(self.compute_width(i/(n-1), m))
        return np.array(P)
