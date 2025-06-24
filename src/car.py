import numpy as np
from src.road import Spline

import matplotlib.pyplot as plt

g = 9.81

class Car:
    def __init__(self, accel, brake, m):
        self.accel = accel
        self.brake = brake
        self.m = m

    def speed_from_curve(self, points, R, mu, dt):
        spl = Spline(len(points)+2, np.concatenate([[points[0]],points,[points[-1]]], axis=0))

        curvatures = np.array([spl.compute_curvature(i/R) for i in range(R)])
        
        speeds = np.sqrt(curvatures*mu*g) # type: ignore
        ds = np.gradient(speeds)
        d2s = np.gradient(ds)
        critics = np.concatenate(np.where(np.abs(ds) < 1e-3))
        minima = [i for i in critics if d2s[i] > 0]

        lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.array([spl.compute_point(i/R) for i in range(R)]), axis=0), axis=1))])

        t = 0
        x = 0

        curr_idx = 0

        new_speeds = [0]
        dist = [0]

        while x < lengths[-1]:
            t += dt
            dst = 0
            v = new_speeds[-1]

            while len(minima) > 0 and minima[0] < curr_idx:
                minima.pop(0)

            if len(minima) > 0:
                dst = (speeds[minima[0]]**2-v**2)/(2*(-self.brake))

            if len(minima) == 0 or lengths[minima[0]]-x > dst: #pas encore de freinage
                v += self.accel*dt
            else: #freinage
                v -= self.brake*dt
            
            v = min(v, speeds[curr_idx])
            new_speeds.append(v)
            dist.append(x)

            x += v*dt
            while lengths[curr_idx] < x and curr_idx < R-2:
                curr_idx += 1

        return new_speeds,dist
    
