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
        tau = 15
        self.dirs = ti.Matrix([(-1, 1), (0, 1), (1, 1),
                               (-1, 0), (0, 0), (1, 0),
                               (-1, -1), (0, -1), (1, -1)])
        self.n = n
        self.disp = ti.field(dtype=ti.f32, shape=(n, n))
        self.f1 = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))
        self.f2 = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))
        self.tau_inv = 1 / tau
        self.vel = ti.field(dtype=ti.f32, shape=(n, n))
        self.w = ti.field(dtype=ti.f32, shape=(9,))
        self.w.from_numpy(np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 36)

        self.colormap = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
        colors = precompute_colormap()
        self.colormap.from_numpy(colors)
        self.rgb_image = ti.Vector.field(3, dtype=ti.u8, shape=(n, n))
        self.max_val = ti.field(ti.f32, shape=())
        self.max_val.fill(1e-8)
        self.boundary = ti.field(dtype=ti.i8, shape=(n, n))

        self.init_grid()

        # for i in (0, self.n - 1):
        for i in (0,):
            for j in range(self.n):
                self.boundary[i, j] = 1
                self.f1[i, j].fill(0)

        for j in (0, self.n - 1):
            for i in range(self.n):
                self.boundary[i, j] = 1
                self.f1[i, j].fill(0)

        for i in range(450 - 300, 550-300):
            for j in range(450, 550):
                self.boundary[i, j] = 1
                self.f1[i, j].fill(0)

        # for i in range(-10, 11):
        #     for j in range(-10, 11):
        #         if i**2 + j ** 2 < 200:
        #             for k in range(9):
        #                 self.grid[n//2 + j, n//2 + i][k] += 5 * w[k]

        # for i in range(-10, 11):
        #     for j in range(20):
        #         if i**2 + j ** 2 < 200:
        #             self.grid[n//2 + j, n//2 + i + 200][1] += 1

        self.f1.fill(1/9)
        self.f1.fill(0)
        self.init_grid()
        for i in range(-10, 11):
            for j in range(-10, 11):
                if i**2 + j ** 2 < 200:
                    for k in range(1):
                        self.f1[-100 + n//2 + j, n//2 + i][5] += self.w[k]
        print("Init done")

    @ti.kernel
    def init_grid(self):
        for i, j in self.f1:
            for k in ti.static(range(9)):
                self.f1[i, j][k] = self.w[k]

    @ti.kernel
    def stream(self):
        for i, j in ti.ndrange((1, self.n - 1), (1, self.n - 1)):
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[k, 0], j + self.dirs[k, 1]][k] = self.f1[i, j][k]

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange(ti.static((1, self.n - 1)), ti.static((1, self.n - 1))):
            rho = self.f1[i, j].sum()
            # Calculates velocity vector in one step
            vel = (self.dirs.transpose() @ self.f1[i, j] / rho) if rho > 0 else tm.vec2([0, 0])
            self.vel[i, j] = vel.norm()
            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[k, 0] + vel[1] * self.dirs[k, 1]
                feq = self.w[k] * rho * (1 + 3 * cm + 4.5 * cm ** 2 - 1.5 * tm.dot(vel, vel))
                self.f2[i + self.dirs[k, 0], j + self.dirs[k, 1]][k] = (1 - self.tau_inv) * self.f1[i, j][k] + self.tau_inv * feq

    @ti.func
    def total_density(self):
        dens = 0.0
        for i, j in self.f1:
            ti.atomic_add(dens, self.f1[i, j].sum())
        return dens

    @ti.func
    def reverse_vector(self, x):
        return ti.Vector([x[8 - i] for i in ti.static(range(9))])

    @ti.kernel
    def bounce_boundary(self):
        for i, j in self.f1:
            if self.boundary[i, j]:
                if i == 0:
                    # self.grid[i, j].fill(0)
                    self.f2[i, j][5] = 1
                if i == self.n - 1:
                    # self.update_grid[i, j] = ti.Vector([0] * 9)
                    self.vel[i, j] = 0
                for k in ti.static(range(9)):
                    self.f1[i + self.dirs[k, 0], j + self.dirs[k, 1]] = self.reverse_vector(self.f2[i, j])
                # self.gird[i, j].fill(0)
                # self.grid[i, j] = self.reverse_vector(self.update_grid[i, j])
                # self.u[i, j] = tm.vec2([0, 0])
            else:
                self.f1[i, j] = self.f2[i, j]

    """Check performance"""
    @ti.kernel
    def update(self):
        # self.grid.copy_from(self.update_grid)
        for i, j in self.f2:
            self.f1[i, j] = self.f2[i, j]

    def apply_colormap(self, data):
        norm_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        colormap = cm.viridis(norm_data)
        return (colormap[:, :, :3] * 255).astype(np.uint8)

    @ti.kernel
    def normalize_and_map(self):
        # print(self.total_density())
        for i, j in self.vel:
            self.max_val[None] = ti.max(self.max_val[None], self.vel[i, j])
        curr_max = 1e-8
        for i, j in self.vel:
            curr_max = ti.max(curr_max, self.vel[i, j])
        print(self.max_val[None], curr_max)

        for i, j in self.vel:
            norm_val = ti.cast(self.vel[i, j] / self.max_val[None] * (255), ti.i32)
            # norm_val = ti.cast(self.u[i, j] / curr_max * (255), ti.i32)

            norm_val = ti.min(ti.max(norm_val, 0), (255))
            for c in ti.ndrange(3):
                self.rgb_image[i, j][c] = ti.u8(self.colormap[norm_val][c] * (255))
                if self.boundary[i, j]:
                    self.rgb_image[i, j][c] = (255)

    @ti.kernel
    def stream_basic(self):
        for i, j in self.f1:
            for k in ti.static(range(9)):
                ni, nj = i + self.dirs[k, 0], j + self.dirs[k, 1]
                if 0 <= ni < self.n and 0 <= nj < self.n:
                    self.f2[ni, nj][k] = self.f1[i, j][k]

    @ti.kernel
    def collide_basic(self):
        # What??..........
        for i, j in ti.ndrange((0, self.n), (0, self.n)):
            rho = self.f2[i, j].sum()
            u = tm.vec2([0.0, 0.0])
            for k in ti.static(range(9)):
                # Compiler throws error on self.dirs[k, 0] indexing. Not sure why.
                u += self.f2[i, j][k] * tm.vec2([self.dirs[k, 0], self.dirs[k, 1]])
            if rho != 0:
                if rho < 0:
                    print("WHAAAT THE FUUUCKKKK")
                u /= rho
            else:
                u = tm.vec2([0, 0])
            self.vel[i, j] = u
            for k in range(9):
                cm = u[0] * self.dirs[k, 0] + u[1] * self.dirs[k, 1]
                feq = self.w[k] * rho * (1 + 3 * cm + 4.5 * cm ** 2 - 1.5 * tm.dot(u, u))
                self.f2[i, j][k] += self.tau_inv * (feq - self.f2[i, j][k])

    @ti.kernel
    def display_new(self):
        for i, j in self.f1:
            self.disp[i, j] = self.f1[i, j].sum() / 9

    def display(self):
        gui = ti.GUI('LBM Simulation', (self.n, self.n))
        self.f2.copy_from(self.f1)
        self.stream()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # self.stream_basic()
            # self.collide_basic()
            # self.update()
            # self.bounce_boundary()
            # self.get_velocity_magnitude()
            # self.normalize_and_map()
            # gui.set_image(self.rgb_image)
            # gui.show()

            self.normalize_and_map()
            gui.set_image(self.rgb_image)
            gui.show()
            for _ in range(10):
                self.collide_and_stream()
                # self.stream()
                # self.update()
                # print(self.dirs[0])

                self.bounce_boundary()
                # exit()


L = LBM()
L.display()
