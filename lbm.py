import taichi as ti
import taichi.math as tm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ti.init(arch=ti.gpu)
n = 150

grid = np.arange(n * n * 9).reshape(n, n, 9)
grid = np.random.randn(n, n, 9)
# grid = np.ones((n, n, 9))
# grid[:, :, 5] = 2

coords = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1),
          (1, -1)]
coords = [np.array(c) for c in coords]







n = 50

rho_init = 1.0
f_init = rho_init / 9
grid = np.full((n, n, 9), f_init)

grid[n//2, 0, 5] += 0.3
grid[n//2 + 10, -1, 3] += 0.2






def streaming(vals):
    for i, (x, y) in enumerate(coords):
        vals[:, :, i] = np.roll(vals[:, :, i], x, axis=1)
        vals[:, :, i] = np.roll(vals[:, :, i], -y, axis=0)
    return vals


def collision(vals, tau=5):
    rho = np.sum(vals, axis=2)
    w = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 9
    u = np.zeros((n, n, 2))
    for y in range(n):
        for x in range(n):
            for i in range(9):
                u[y, x] += vals[y, x, i] * coords[i]
            if rho[y, x] > 0:
                u[y, x] /= rho[y, x]
            else:
                u[y, x] = np.array([0, 0])
    feq = np.zeros_like(grid)
    for y in range(n):
        for x in range(n):
            uu = u[y, x]
            for i in range(9):
                feq[y, x][i] = w[i] * rho[y, x] * (1 + 3 * np.dot(coords[i], uu) + 9/2 * np.dot(coords[i], uu) ** 2 - 3/2 * (np.dot(uu, uu)))
    return vals + (feq - vals) / tau


fig, ax = plt.subplots(figsize=(8, 6))


def update(iteration):
    global grid
    grid = collision(streaming(grid))


    rho = np.sum(grid, axis=2)
    rho[rho == 0] = 1

    ax.clear()
    ax.imshow(rho, cmap='viridis', origin='lower')

    ax.set_title(f"Iteration {iteration}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


ani = FuncAnimation(fig, update, frames=100, interval=100)
plt.show()
