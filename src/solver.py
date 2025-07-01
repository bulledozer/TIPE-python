import numpy as np
import copy

class Solver:
    def __init__(self, points, scale, n_sectors, vmax, dx):
        self.points = points
        self.scale = scale
        self.n_sectors = n_sectors
        self.vmax = vmax
        self.dx = dx

    def points_from_state(self, state):
        return np.array([self.points[i][0]*(1-state[i])+self.points[i][1]*state[i] for i in range(len(state))])

    def time_from_state(self, state):
        controls = self.points_from_state(state)

        t = 0

        for i in range(len(state)-2):
            L1 = np.sqrt((controls[i+1,0]-controls[i+2,0])**2 + (controls[i+1,1]-controls[i+2,1])**2)
            L2 = np.sqrt((controls[i+1,0]-controls[i,0])**2 + (controls[i+1,1]-controls[i,1])**2)
            
            theta = np.arccos(np.dot(controls[i,:]-controls[i+1,:], controls[i+2,:]-controls[i+1,:])/(L1*L2)) 

            t += 1/(min(np.abs(np.tan(theta/2)), self.vmax))
        return t

    def gradient_descent(self, state, scale, times, timef):
        base_time = timef(state)

        times.append(base_time)

        gradient = [0]*self.n_sectors


        for i in range(self.n_sectors):
            state2 = copy.deepcopy(state)
            state2[i] = np.clip(state2[i]+self.dx, 0,1)

            new_time = timef(state2)

            gradient[i] = new_time-base_time
        
        for i in range(self.n_sectors):
            state[i] -= gradient[i]*scale
            state[i] = np.clip(state[i],0,1)

    def solve(self, n_iter, times, verbose = True):
        curve_state = [0.5]*(self.n_sectors)

        min_state = curve_state
        min_time = float('inf')

        for i in range(n_iter):
            self.gradient_descent(curve_state, self.scale, times, self.time_from_state)
            if times[-1] < min_time:
                min_state = curve_state

            if verbose and not i%50:
                print("Iter : ", i, " | Temps : ", times[-1])

        return np.array(self.points_from_state(min_state))

