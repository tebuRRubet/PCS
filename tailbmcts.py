import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.cm as cm


ti.init(arch=ti.gpu)


def precompute_colormap():
    import matplotlib.cm as cm
    viridis = cm.get_cmap('flag', 256)
    # Extract RGB values
    colors = np.roll(viridis(np.linspace(0, 1, 256))[:, :3], 3)
    return colors.astype(np.float32)




@ti.data_oriented
class LBM:
    def __init__(self, n=1000, tau=5):
        # tau = 10
        coords = [(-1, 1), (0, 1), (1, 1),
                  (-1, 0), (0, 0), (1, 0),
                  (-1, -1), (0, -1), (1, -1)]
        self.coords = ti.Vector.field(n=2, dtype=ti.i32, shape=(9,))
        for i in range(9):
            self.coords[i] = coords[i]
        self.n = n
        self.disp = ti.field(dtype=ti.f32, shape=(n, n))
        self.grid = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))
        self.update_grid = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))
        self.tau = tau
        self.tau_inv = 1 / tau
        self.u = ti.Vector.field(n=2, dtype=ti.f32, shape=(n, n))
        self.w = ti.field(dtype=ti.f32, shape=(9,))
        w = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 36
        self.w.from_numpy(w)
        self.colormap = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
        colors = precompute_colormap()
        self.colormap.from_numpy(colors)
        self.rgb_image = ti.Vector.field(3, dtype=ti.u8, shape=(n, n))
        self.max_val = ti.field(ti.f32, shape=())
        self.max_val.fill(1e-8)
        self.boundary = ti.field(dtype=ti.i8, shape=(n, n))

        self.init_grid()

        # for i in (0, self.n - 1):
        #     for j in range(self.n):
        #         self.boundary[i, j] = 1
        #         self.grid[i, j].fill(0)

        # for j in (0, self.n - 1):
        #     for i in range(self.n):
        #         self.boundary[i, j] = 1
        #         self.grid[i, j].fill(0)

        # for i in range(450, 550):
        #     for j in range(450, 550):
        #         self.boundary[i, j] = 1
        #         self.grid[i, j].fill(0)


        # for i in range(-10, 11):
        #     for j in range(-10, 11):
        #         if i**2 + j ** 2 < 200:
        #             for k in range(9):
        #                 self.grid[n//2 + j, n//2 + i][k] += 5 * w[k]

        # for i in range(-10, 11):
        #     for j in range(20):
        #         if i**2 + j ** 2 < 200:
        #             self.grid[n//2 + j, n//2 + i + 200][1] += 1

        self.grid.fill(1/9)
        self.grid.fill(0)
        for i in range(-10, 11):
            for j in range(-10, 11):
                if i**2 + j ** 2 < 200:
                    for k in range(9):
                        self.grid[n//2 + j, n//2 + i][k] += w[k]
        print("Init done")
        self.init_grid()


    @ti.kernel
    def init_grid(self):
        for i, j in self.grid:
            for k in ti.static(range(9)):
                self.grid[i, j][k] = self.w[k]
                # print(self.w[k])

    @ti.kernel
    def stream(self):
        for i, j in ti.ndrange((1, self.n - 1), (1, self.n - 1)):
            for k in ti.static(range(9)):
                self.update_grid[i + self.coords[k].x, j + self.coords[k].y][k] = self.grid[i, j][k]

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange(ti.static((1, self.n - 1)), ti.static((1, self.n - 1))):
            rho = self.grid[i, j].sum()
            u = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                u += self.coords[k] * self.grid[i, j][k]

            u = (u / rho) if rho > 0 else tm.vec2([0, 0])
            self.u[i, j] = u
            for k in ti.static(range(9)):
                feq = self.w[k] * rho * (1 + 3 * tm.dot(self.coords[k], u) + 4.5 * tm.dot(self.coords[k], u) ** 2 - 1.5 * tm.dot(u, u))
                # if self.n > i + self.coords[k].x >= 0 and self.n > j + self.coords[k].y >= 0:
                self.update_grid[i + self.coords[k].x, j + self.coords[k].y][k] = (1 - self.tau_inv) * self.grid[i, j][k] + self.tau_inv * feq



    @ti.func
    def total_density(self):
        dens = 0.0
        for i, j in self.grid:
            ti.atomic_add(dens, self.grid[i, j].sum())
        return dens

    @ti.func
    def reverse_vector(self, x):
        return ti.Vector([x[i] for i in range(x.get_shape()[0])][::-1])

    @ti.kernel
    def bounce_boundary(self):
        for i, j in self.grid:
            if self.boundary[i, j]:
                # if i == 0:
                #     self.grid[i, j].fill(0)
                #     self.grid[i, j][5] = 1
                # else:
                for k in ti.static(range(9)):
                    self.grid[i + self.coords[k].x, j + self.coords[k].y][8-k] = self.update_grid[i, j][k]
                # self.grid[i, j] = self.reverse_vector(self.update_grid[i, j])
                # self.u[i, j] = tm.vec2([0, 0])
            else:
                self.grid[i, j] = self.update_grid[i, j]

    """Check performance"""
    @ti.kernel
    def update(self):
        # self.grid.copy_from(self.update_grid)
        for i, j in self.update_grid:
            self.grid[i, j] = self.update_grid[i, j]

    @ti.kernel
    def get_velocity_magnitude(self):
        for i, j in self.grid:
            self.disp[i, j] = self.u[i, j].norm()
            # self.disp[i, j] = tm.dot(ti.Vector([1] * 9), self.grid[i, j])

    def apply_colormap(self, data):
        norm_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        colormap = cm.viridis(norm_data)
        return (colormap[:, :, :3] * 255).astype(np.uint8)

    @ti.kernel
    def normalize_and_map(self):
        print(self.total_density())
        for i, j in self.disp:
            self.max_val[None] = ti.max(self.max_val[None], self.disp[i, j])
        # curr_max = 1e-8
        # for i, j in self.disp:
        #     curr_max = ti.max(curr_max, self.disp[i, j])
        # print(self.max_val[None], curr_max)

        for i, j in self.disp:
            norm_val = ti.cast(self.disp[i, j] / self.max_val[None] * (255), ti.i32)
            norm_val = ti.min(ti.max(norm_val, 0), (255))
            for c in ti.ndrange(3):
                self.rgb_image[i, j][c] = ti.u8(self.colormap[norm_val][c] * (255))
                if self.boundary[i, j]:
                    self.rgb_image[i, j][c] = (255)




    @ti.kernel
    def stream_new(self):
        for i, j in self.grid:
            for k in ti.static(range(9)):
                ni, nj = i + self.coords[k].x, j + self.coords[k].y
                if 0 <= ni < self.n and 0 <= nj < self.n:
                    self.update_grid[ni, nj][k] = self.grid[i, j][k]


    @ti.kernel
    def collide_new(self):
        for i, j in self.update_grid:
            rho = self.update_grid[i, j].sum()
            u = tm.vec2([0.0, 0.0])
            for k in range(9):
                u += self.coords[k] * self.update_grid[i, j][k]
            if rho != 0:
                if rho < 0:
                    print("WHAAAT THE FUUUCKKKK")
                u /= rho
            else:
                u = tm.vec2([0, 0])
            for k in range(9):
                feq = self.w[k] * rho * (1 + 3 * tm.dot(self.coords[k], u) + 4.5 * tm.dot(self.coords[k], u) ** 2 - 1.5 * tm.dot(u, u))
                self.update_grid[i, j][k] += self.tau_inv * (feq - self.update_grid[i, j][k])


    @ti.kernel
    def display_new(self):
        for i, j in self.grid:
            self.disp[i, j] = self.grid[i, j].sum() / 9


    def display(self):
        gui = ti.GUI('LBM Simulation', (self.n, self.n))
        self.update_grid.copy_from(self.grid)
        self.stream()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            self.stream_new()
            self.collide_new()
            self.update()
            self.display_new()
            gui.set_image(self.disp)
            gui.show()




            # self.get_velocity_magnitude()
            # self.normalize_and_map()
            # gui.set_image(self.rgb_image)
            # gui.show()
            # self.collide_and_stream()
            # self.bounce_boundary()


L = LBM()
L.display()
