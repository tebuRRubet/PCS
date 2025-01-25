import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


ti.init(arch=ti.gpu)
fig, ax = plt.subplots(figsize=(8, 6))


@ti.data_oriented
class LBM:
    def __init__(self, n=600, rho_init=1.0):
        coords = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1),
                  (0, -1), (1, -1)]
        self.coords = ti.Vector.field(n=2, dtype=ti.i32, shape=(9,))
        for i in range(9):
            self.coords[i] = coords[i]
        self.n = n
        f_init = rho_init / 9
        self.disp = ti.field(dtype=ti.f32, shape=(n, n))
        self.grid = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))
        self.update_grid = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))


        for i in range(-10, 11):
            for j in range(20):
                self.grid[j, n//2 + i][5] += 10


    @ti.kernel
    def streaming(self):
        self.update_grid.fill(0)
        for i, j in self.grid:
            for k in range(9):
                self.update_grid[(i + self.coords[k][0] + self.n) % self.n, (j + self.coords[k][1] + self.n) % self.n][k] = self.grid[i, j][k]

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
        self.streaming()
        exit()
        self.collision()

        rho = np.sum(self.grid, axis=2)
        rho[rho == 0] = 1

        ax.clear()
        ax.imshow(rho, cmap='viridis', origin='lower')

        ax.set_title(f"Iteration {iteration}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    @ti.kernel
    def get_2d(self):
        for i, j in self.grid:
            self.disp[i, j] = 0
            for k in range(9):
                self.disp[i, j] += self.grid[i, j][k]
            self.disp[i, j] /= 9

    def display(self):
        gui = ti.GUI('Hello World!', (self.n, self.n))
        while gui.running:
            self.get_2d()
            gui.set_image(self.disp)
            gui.show()
            self.streaming()
            # self.temp = self.grid
            # self.grid = self.update_grid
            # self.update_grid = self.temp
            self.grid, self.update_grid = self.update_grid, self.grid
            print(self.grid[1, self.n//2])


L = LBM()
L.display()

# ani = FuncAnimation(fig, L.update, frames=100, interval=10)
# plt.show()
