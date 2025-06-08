import numpy as np
from scipy.interpolate import *
from utils import *
from bisect import bisect_right


class Spline:
    def __init__(self,N, M = []):
        self.N = N
        self.lengths = []
        #self.points = SX.sym("P",N,2) if M == SX(0) else M
        self.points = np.array(M) if M != [] else np.zeros((N,2))
        

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

    def compute_point_seg(self, t, seg):
        A = self.points[seg%self.N]  
        B = self.points[(seg+1)%self.N]
        C = self.points[(seg+2)%self.N]
        D = self.points[(seg+3)%self.N]

        return 0.5*np.matmul(np.matmul(np.array([1,t,t**2,t**3]),np.array([[0,2,0,0],[-1,0,1,0],[2,-5,4,-1],[-1,3,-3,1]])),np.array([A,B,C,D]))

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

            dst = np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

            t += dst/np.sqrt(min(self.compute_curvature(i/R), vmax))
        return t
    
    def compute_time_fast(self, R, vmax):
        times = np.linspace(.0,1.,R)
        curvature = np.maximum(np.array([self.compute_curvature(t) for t in times]), [vmax]*R)
        return sum(1/curvature)
    
    def compute_lengths(self, R):
        self.lengths = []
        for i in range(self.N-3):
            times = np.linspace(0.0,1.0,R)
            pos = np.array([self.compute_point_seg(t,i) for t in times])
            pos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            self.lengths.append(np.sum(pos))

        return self.lengths


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
    
    def lerp_line(self, t, s):
        l1,l2 = self.compute_width(t,2)
        return (1-s)*l1+s*l2
    
    def compute_points(self, n,m):
        if self.lengths == []:
            raise ValueError("self.lengths est vide")

        P = []
        #for i in range(n):
        #    P.append(self.compute_width(i/(n-1), m))

        for i in range(self.N-3):
            #dst = np.sqrt((self.points[i+1][0]-self.points[i+2][0])**2+(self.points[i+1][1]-self.points[i+2][1])**2)
            #times = np.linspace(i/(self.N-3), (i+1)/(self.N-3), int(dst*n), endpoint=False)
            times = np.linspace(0.0,1.0,int(self.lengths[i]*n), endpoint=False)
            for t in times:
                P.append(self.compute_width(t,m,i))

        return np.array(P)
