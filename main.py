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

"""SOL_RES = 100
opti = Opti()

T = opti.variable()
V = opti.variable(SOL_RES)
PHI = opti.variable(SOL_RES)
N = opti.variable(SOL_RES)

for i in range(SOL_RES):"""
    




SOL_RES = 30
MAX_CURVATURE = MX(100)
opti = Opti()

T = opti.variable()
X = opti.variable(SOL_RES)

T = MX(0)

#spl = Spline()

def time_CR(x):


    R = 50
    T2 = MX(0)

    controls = MX(SOL_RES,2)
    for i in range(SOL_RES):
        pts = spl.compute_width(i/SOL_RES, 2)
        P1,P2 = MX(2,1),MX(2,1)
        P1[0] = pts[0][0]
        P1[1] = pts[0][1]
        P2[0] = pts[1][0]
        P2[1] = pts[1][1]
        controls[i,:] = P1*(1-X[i])+P2*X[i]

    for i in range(R):
        m = i/R
        seg = int(m*(SOL_RES-3))

        A = controls[seg%SOL_RES,:]  
        B = controls[(seg+1)%SOL_RES,:]
        C = controls[(seg+2)%SOL_RES,:]
        D = controls[(seg+3)%SOL_RES,:]

        nm = (m%(1/(SOL_RES-3)))*(SOL_RES-3)

        P = 0.5*((2*B)+nm*(-A+C)+nm**2*(2*A-5*B+4*C-D)+nm**3*(-A+3*B-3*C+D))
        P2 = 0.5*((2*B)+(nm+1/R)*(-A+C)+(nm+1/R)**2*(2*A-5*B+4*C-D)+(nm+1/R)**3*(-A+3*B-3*C+D))
        dP = 0.5*((-A+C)+2*nm*(2*A-5*B+4*C-D)+3*nm**2*(-A+3*B-3*C+D))
        d2P = 0.5*(2*(2*A-5*B+4*C-D)+6*nm*(-A+3*B-3*C+D))

        C_radius = ((dP[0]**2+dP[1]**2)**1.5)/(fabs(d2P[0]*dP[1]-dP[0]*d2P[1])) 
        #t += fmax(C_radius, MAX_CURVATURE) #minimiser le temps et pas la courbure !!!!!
        #T += C_radius
        T2 += (P2[0]-P[0])**2 + (P2[1]-P[1])**2

    return T2
    #bonjour, comment ça slope tout ça?

def time_2(X):
    T2 = MX(0)

    controls = MX(SOL_RES,2)
    for i in range(SOL_RES):
        pts = spl.compute_width(i/SOL_RES, 2)
        P1,P2 = MX(2,1),MX(2,1)
        P1[0] = pts[0][0]
        P1[1] = pts[0][1]
        P2[0] = pts[1][0]
        P2[1] = pts[1][1]
        controls[i,:] = P1*(1-X[i])+P2*X[i]

    for i in range(SOL_RES-3):
        L1 = sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
        L2 = sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
        L3 = sqrt((controls[i+3,0]-controls[i+2,0])**2 + (controls[i+3,1]-controls[i+2,1])**2)
        
        theta = arccos(dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2)) 
        phi = arccos(dot(controls[i+1,:]-controls[i+2,:], controls[i+3,:]-controls[i+2,:])/(L1*L3))
        alpha = ((phi-theta))

        #T2 += (log(phi/theta))/(alpha)
        T2 += 2*(sqrt(phi)-sqrt(theta))/alpha
    return T2


T = time_2(X)


opti.subject_to(X <= 1)
opti.subject_to(X >= 0)

opti.solver('ipopt',{'expand': True},{'max_iter':40})
opti.minimize(T)

sol = opti.solve_limited()
debug = opti.debug.value(X)

weights = sol.value(X)

RENDER_RES = SOL_RES
RENDER_CONTROL = True

sol_spline = Spline(SOL_RES, [spl.lerp_line(i/RENDER_RES, weights[i]) for i in range(RENDER_RES)])

points = np.array([sol_spline.compute_point(i/RENDER_RES) for i in range(RENDER_RES+1)])
cols = [1-1/(1+0.03*spl.compute_curvature(i/RENDER_RES)) for i in range(RENDER_RES+1)]
FAST_COL = (0.75,1,0)
SLOW_COL = (1,0,0)

for i in range(len(points)-1):
    pl.plot((points[i][0],points[i+1][0]), (points[i][1],points[i+1][1]), c=[SLOW_COL[j]*(1-cols[i])+FAST_COL[j]*cols[i] for j in range(3)], linewidth=3)
    
if RENDER_CONTROL:
    for i in range(SOL_RES+1):
        pl.plot(sol_spline.compute_point(i/RENDER_RES)[0], sol_spline.compute_point(i/RENDER_RES)[1], 'ro')

#pl.scatter(points[:,0],points[:,1], c="red")#, c=[(c,c,c) for c in curvature])
pl.show()