import pylab as pl

mu = 1.3 # coef de frottement statique
g = 9.81 # intensité du champ gravitationnel terrestre (m/s²)

accel = 3 # accélération de la voiture (m/s²)

R1,R2 = 10,15 # rayon de courbure intérieur (resp. extérieur) en m

N = 100

START = pl.pi/4
END = pl.pi-0.1

theta = pl.linspace(START, END, N)
phi = pl.linspace(-END/2, END/2, N)

T,P = pl.meshgrid(theta,phi)

R = [[] for i in range(N)]
V = [[] for i in range(N)]
for i in range(N):
    for j in range(N):
        if theta[i]/2 - phi[j] <= 0:
            R[i].append(0)
            V[i].append(0)
            continue
        r = R1 + (R2-R1)/(1-pl.cos(theta[i]/2 - phi[j]))
        re = R1 + (R2-R1)/(1-pl.sin(theta[i]/2 - phi[j]))
        R[i].append(r)
        V[i].append(min(pl.sqrt(r*mu*g), pl.sqrt(2*accel*r*phi[j])))

ax = pl.figure().add_subplot(projection='3d')
ax.plot_surface(T,P,pl.array(V),edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,alpha=0.3)

ax.set(zlim=(-100,100),xlabel='theta', ylabel='phi', zlabel='R')

pl.show()