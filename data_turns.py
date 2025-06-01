import pylab as pl

mu = 1.3 # coef de frottement statique
g = 9.81 # intensité du champ gravitationnel terrestre (m/s²)

accel = 3 # accélération de la voiture (m/s²)

R1,R2 = 10,15 # rayon de courbure intérieur (resp. extérieur) en m

N = 100

START = pl.pi/4
END = pl.pi-0.1

theta = pl.linspace(START, END, N)
phi = pl.linspace(0, END/2-0.1, N)

T,P = pl.meshgrid(theta,phi)

R = [[] for i in range(N)]
Vs = [[] for i in range(N)]
for i in range(N):
    for j in range(N):
        if T[i][j]/2 - P[i][j] <= pl.pi/6:
            R[i].append(0)
            Vs[i].append(0)
            continue
        
        phip = T[i][j]/2 - P[i][j]

        r = R1 + (R2-R1)/(1-pl.cos(phip))
        re = R1 + (R2-R1)/(1-pl.sin(phip))
        R[i].append(r)

        if P[i][j] >= 0:
            Vs[i].append(min(pl.sqrt(r*mu*g), pl.sqrt(2*accel*r*(phip) + re*mu*g)))
        else:
            Vs[i].append(pl.sqrt(re*mu*g))

ax = pl.figure().add_subplot(projection='3d')
ax.plot_surface(T,P,pl.array(Vs),edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,alpha=0.3)

ax.contour(T, P, pl.array(Vs), zdir='x', offset = END, cmap='coolwarm')
ax.contour(T, P, pl.array(Vs), zdir='y', offset = END/2, cmap='coolwarm')

ax.set(xlim=(START, END), ylim=(0, END/2),zlim=(0,50),xlabel='theta', ylabel='phi', zlabel='R')

pl.show()