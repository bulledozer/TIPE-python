import numpy as np
import matplotlib.pyplot as plt

def gen_speed_profile(speeds, s, start_speed, accel, decel):
    speed_prof = np.copy(speeds)
    speed_prof[0] = start_speed 
    
    ds = np.diff(s)

    for i in range(len(speed_prof)-1):
        nsp = np.sqrt(speed_prof[i]**2 + 2*accel*ds[i])
        speed_prof[i+1] = min(speed_prof[i+1], nsp)

    for i in range(len(speed_prof)-1):
        nsp = np.sqrt(speed_prof[len(speed_prof)-i-1]**2 + 2*ds[len(speed_prof)-i-2]*decel)
        speed_prof[len(speed_prof)-i-2] = min(speed_prof[len(speed_prof)-i-2], nsp)

    return speed_prof, s

MAX_SPEED = 50

N = 100
speeds = np.ones(N)*MAX_SPEED
speeds[30:50] = MAX_SPEED/4
speeds[40] = 0.1
speeds[60] = 10e10

speed_prof,s = gen_speed_profile(speeds, np.linspace(0,100, N), 0, 5, 10)

plt.plot(s, speed_prof)
plt.plot(s, speeds)
plt.show()
