import matplotlib.pyplot as plt


def plot_points(P, plot, show_quads, c):
    for i in range(len(P)-1):
        X = [P[i][0][0], P[i][-1][0],P[i+1][0][0],P[i+1][-1][0]]
        Y = [P[i][0][1], P[i][-1][1], P[i+1][0][1], P[i+1][-1][1]]
        if show_quads:
            plot.plot((X[0],X[1]),(Y[0],Y[1]), c=c)
            plot.plot((X[2],X[3]),(Y[2],Y[3]), c=c)
        plot.plot((X[0],X[2]),(Y[0],Y[2]), c=c)
        plot.plot((X[3],X[1]),(Y[3],Y[1]), c=c)
        

def inv_lerp(a: float, b: float, v: float) -> float:

    return (v - a) / (b - a)

def lerp(a,b,x):
    return (1-x)*a + x*b