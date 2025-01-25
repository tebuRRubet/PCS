import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


ti.init(arch=ti.gpu)
fig, ax = plt.subplots(figsize=(8, 6))


class LBM:
    def __init__(self, n=150, rho_init=1.0):
        coords = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1),
                  (0, -1), (1, -1)]
        self.coords = [np.array(c) for c in coords]
        self.n = n
        f_init = rho_init / 9
        self.grid = np.full((n, n, 9), f_init)

        self.grid[n//2, 0, 5] += 1
        self.grid[n//2 + 10, -1, 3] += 0.2

    def streaming(self):
        for i, (x, y) in enumerate(self.coords):
            self.grid[:, :, i] = np.roll(self.grid[:, :, i], x, axis=1)
            self.grid[:, :, i] = np.roll(self.grid[:, :, i], -y, axis=0)

    def collision(self, tau=5):
        rho = np.sum(self.grid, axis=2)
        w = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 9
        u = np.tensordot(self.grid, self.coords, axes=([2], [0]))
        u = np.divide(u, rho[:, :, None], where=rho[:, :, None] != 0, out=np.zeros_like(u))
        # for y in range(n):
        #     for x in range(n):
        #         u[y, x] = np.dot(vals)
        #         for i in range(9):
        #             u[y, x] += vals[y, x, i] * self.coords[i]
        #         if rho[y, x] > 0:
        #             u[y, x] /= rho[y, x]
        #         else:
        #             u[y, x] = np.array([0, 0])
        uu = np.sum(u ** 2, axis=-1)
        cu = np.einsum('qd,abd->abq', self.coords, u)
        feq = w[None, None, :] * rho[:, :, None] * (
            1 + 3 * cu + 9/2 * cu ** 2 - 3/2 * uu[:, :, None]
        )
        # feq = np.zeros_like(grid)
        # for y in range(n):
        #     for x in range(n):
        #         uu = u[y, x]
        #         for i in range(9):
        #             feq[y, x][i] = w[i] * rho[y, x] * (1 + 3 * np.dot(self.coords[i], uu) + 9/2 * np.dot(self.coords[i], uu) ** 2 - 3/2 * (np.dot(uu, uu)))
        self.grid += (feq - self.grid) / tau

    def update(self, iteration):
        self.collision()
        self.streaming()

        rho = np.sum(self.grid, axis=2)
        rho[rho == 0] = 1

        ax.clear()
        ax.imshow(rho, cmap='viridis', origin='lower')

        ax.set_title(f"Iteration {iteration}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")


L = LBM()
ani = FuncAnimation(fig, L.update, frames=100, interval=10)
plt.show()
