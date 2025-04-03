from casadi import *
import pylab as pl

from random import randint

from road import *
from utils import *


#---------COSMETIQUE (afficher la route) -----------

track_points = []
N = 0

with open("road.txt", 'r') as f:
    L = f.readlines()
    N = len(L)
    for l in L:
        l1 = l.removesuffix('\n').split(',')
        track_points.append((float(l1[0]), float(l1[1])))

#P = []
spl = Road(N, track_points, 8.5)

#points = np.array([spl.compute_point(i/R) for i in range(R+1)])
#curvature = np.array([1-1/(1+0.005*spl.compute_curvature(i/R)) for i in range(R+1)])

p2 = spl.compute_points(100,2)
plot_points(p2)


#----------------SOLVEUR------------------


SOL_RES = 24
opti = Opti()

X = opti.variable(SOL_RES)

#spl = Spline()

def F(x):
    R = 100
    controls = MX(SOL_RES,2)
    for i in range(SOL_RES):
        pts = spl.compute_width(i/SOL_RES, 2)
        P1,P2 = MX(2,1),MX(2,1)
        P1[0] = pts[0][0]
        P1[1] = pts[0][1]
        P2[0] = pts[1][0]
        P2[1] = pts[1][1]
        controls[i,:] = P1*(1-x[i])+P2*x[i]
    
    t = MX(0)
    for i in range(R):
        m = i/R
        seg = int(m*(SOL_RES-3))

        A = controls[seg%SOL_RES,:]  
        B = controls[(seg+1)%SOL_RES,:]
        C = controls[(seg+2)%SOL_RES,:]
        D = controls[(seg+3)%SOL_RES,:]

        nm = (m%(1/(SOL_RES-3)))*(SOL_RES-3)

        #P = 0.5*((2*B)+nm*(-A+C)+nm**2*(2*A-5*B+4*C-D)+t**3*(-A+3*B-3*C+D))
        dP = 0.5*((-A+C)+2*nm*(2*A-5*B+4*C-D)+3*nm**2*(-A+3*B-3*C+D))
        d2P = 0.5*(2*(2*A-5*B+4*C-D)+6*nm*(-A+3*B-3*C+D))

        t += ((dP[0]**2+dP[1]**2)**1.5)/fabs(d2P[0]*dP[1]-dP[0]*d2P[1])

    return t

opti.minimize(F(X))

opti.subject_to(X <= 1);
opti.subject_to(X >= 0);

opti.solver('ipopt')
sol = opti.solve()
print(sol.value(X))

weights = sol.value(X)
sol_spline = Spline(SOL_RES, [spl.lerp_line(i/N, weights[i]) for i in range(SOL_RES)])

points = np.array([sol_spline.compute_point(i/100) for i in range(100+1)])

for i in range(len(points)-1):
    pl.plot((points[i][0],points[i+1][0]), (points[i][1],points[i+1][1]), c="red", linewidth=3)

#pl.scatter(points[:,0],points[:,1], c="red")#, c=[(c,c,c) for c in curvature])
pl.show()