import numpy as np
from scipy.signal import find_peaks

from src.road import Spline

import matplotlib.pyplot as plt

class Car:
    def __init__(self, accel, brake, m, g):
        self.accel = accel
        self.brake = brake
        self.m = m
        self.g = g

    def profile_prom_point(self, s, p, v, a_max, a_min):
        speeds = np.zeros(len(s))
        speeds[p] = v
        for i in range(p):
            speeds[p-1-i] = np.sqrt(speeds[p-i]**2-2*s[p-i-1]*a_min)
        for i in range(len(s)-p-1):
            speeds[p+i+1] = np.sqrt(speeds[p+i]**2+2*s[p+i]*a_max)

        return speeds

    def compute_velocity_profile(self, points, mu, R):
        spl = Spline(len(points)+2, np.concatenate([[points[0]],points,[points[-1]]], axis=0))

        curvatures = np.array([spl.compute_curvature(i/R) for i in range(R)])
        v_max_theo = np.sqrt(curvatures*mu*self.g)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.array([spl.compute_point(i/R) for i in range(R)]), axis=0), axis=1))])

        minima, prop = find_peaks(-v_max_theo, plateau_size=1)
        profiles = [self.profile_prom_point(s, i, v_max_theo[i], self.accel, self.brake) for i in np.concatenate([minima, prop['left_edges'], prop['right_edges']])]

        speeds = np.copy(v_max_theo)
        for p in profiles:
            speeds = np.minimum(speeds, p)

        return speeds
