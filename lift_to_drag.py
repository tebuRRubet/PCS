import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


drag = []
lift = []

for i in range(4):
    data = np.loadtxt(f"drags{5 * (i + 1)}.csv", delimiter=",", dtype=float)
    drag.extend(data.tolist())
    data = np.loadtxt(f"lifts{5 * (i + 1)}.csv", delimiter=",", dtype=float)
    lift.extend(data.tolist())


def parabola(x, a, b, c):
    return a * (x - b)**2 + c


last = 5000
its = 2000
mean_drag = np.array([np.mean(drag[i][-last:-last + its]) for i in range(len(drag))])
mean_lift = np.array([np.mean(lift[i][-last:-last + its]) for i in range(len(lift))])
mean_ratio = mean_lift / mean_drag
angles = [i for i in range(20)]

start, end = 3, 12
fit, _ = curve_fit(parabola, angles[start:end], mean_ratio[start:end])

x_peak = fit[1]
print(x_peak)

plt.plot(angles[start:end], parabola(angles[start:end], *fit), linestyle='--', label=f"peak x: {x_peak}")
plt.plot(mean_ratio)
plt.xlabel("Angle of the airfoil in degrees")
plt.ylabel("Lift/Drag ratio")

plt.legend()
plt.show()
