import pylab as pl


def plot_points(P):
    for i in range(len(P)-1):
        X = [P[i][0][0], P[i][-1][0],P[i+1][0][0],P[i+1][-1][0]]
        Y = [P[i][0][1], P[i][-1][1], P[i+1][0][1], P[i+1][-1][1]]
        pl.plot((X[0],X[1]),(Y[0],Y[1]), c='green')
        pl.plot((X[0],X[2]),(Y[0],Y[2]), c='green')
        pl.plot((X[3],X[1]),(Y[3],Y[1]), c='green')
        pl.plot((X[2],X[3]),(Y[2],Y[3]), c='green')
