import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt


ti.init(arch=ti.gpu)


@ti.data_oriented
class LBM:
    def __init__(self, n=1000, rho_init=1.0, tau=5):
        tau = 10
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
        self.rho = ti.field(dtype=ti.f32, shape=(n, n))
        self.rho.fill(1)
        self.tau = tau
        self.tau_inv = 1 / tau
        self.u = ti.Vector.field(n=2, dtype=ti.f32, shape=(n, n))
        self.w = ti.field(dtype=ti.f32, shape=(9,))
        w = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 64
        self.w.from_numpy(w)


        self.grid.fill(f_init)
        self.grid[0, n//2][5] += 5
        for i in range(-10, 11):
            for j in range(20):
                if i**2 + j ** 2 < 200:
                    self.grid[j, n//2 + i] += 1
        # self.grid.fill(f_init)
        # self.grid[1, n//2][5] += 1


    # Race condition split
    @ti.kernel
    def stream_and_collide(self):
        self.rho.fill(0)

        self.u.fill(0)
        print(self.rho[0, 0])
        for i, j in self.grid:
            for k in ti.static(range(9)):
                self.update_grid[(i + self.coords[k].x + self.n) % self.n, (j + self.coords[k].y + self.n) % self.n][k] = self.grid[i, j][k]
                self.rho[i, j] += self.update_grid[i, j][k]
                self.u[i, j] += self.coords[k] * self.grid[i, j][k]
            ''''''''
            self.u[i, j] = (self.u[i, j] / self.rho[i, j]) if self.rho[i, j] > 1e-6 else tm.vec2([0, 0])
            # self.u[i, j] *= 1 / self.rho[i, j] * (self.rho[i, j] != 0)
        print(self.rho[0, 0])
        print()


    @ti.kernel
    def collide(self):
        for i, j in self.update_grid:
            for k in ti.static(range(9)):
                feq = self.w[k] * self.rho[i, j] * (1 + 3 * tm.dot(self.coords[k], self.u[i, j]) + 9 / 2 * tm.dot(self.coords[k], self.u[i, j]) ** 2 - 3/2 * tm.dot(self.u[i, j], self.u[i, j]))
                # print(self.tau_inv * (feq - self.update_grid[i, j][k]))
                self.update_grid[i, j][k] += self.tau_inv * (feq - self.update_grid[i, j][k])

    def update(self):
        self.grid.copy_from(self.update_grid)



    @ti.kernel
    def get_velocity_magnitude(self):
        for i, j in self.grid:
            self.disp[i, j] = self.u[i, j].norm()


    def display(self):
        gui = ti.GUI('LBM Simulation', (self.n, self.n))
        self.update_grid.copy_from(self.grid)

        while gui.running:
            self.get_velocity_magnitude()
            gui.set_image(self.disp.to_numpy())
            self.stream_and_collide()
            self.collide()
            self.update()
            gui.show()




L = LBM()
L.display()

