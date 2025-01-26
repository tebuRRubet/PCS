import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.cm as cm


ti.init(arch=ti.gpu)


def precompute_colormap():
    import matplotlib.cm as cm
    viridis = cm.get_cmap('flag', 256)
    colors = viridis(np.linspace(0, 1, 256))[:, :3]  # Extract RGB values
    return colors.astype(np.float32)


@ti.data_oriented
class LBM:
    def __init__(self, n=1000, rho_init=1.0, tau=5, wrap=True):
        self.wrap = wrap
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
        self.colormap = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
        colors = precompute_colormap()
        self.colormap.from_numpy(colors)
        self.rgb_image = ti.Vector.field(3, dtype=ti.u8, shape=(n, n))
        self.max_val = ti.field(ti.f32, shape=())
        self.max_val.fill(1e-8)

        self.grid.fill(f_init)
        # self.grid[0, n//2][5] += 10
        for i in range(-10, 11):
            for j in range(20):
                if i**2 + j ** 2 < 200:
                    self.grid[j, n//2 + i] += 2

        for i in range(-10, 11):
            for j in range(20):
                if i**2 + j ** 2 < 200:
                    self.grid[n//2 + j, n//2 + i + 150] += 2
        # self.grid.fill(f_init)
        # self.grid[1, n//2][5] += 1

    # Race condition split
    @ti.kernel
    def stream_and_collide(self):
        self.rho.fill(0)
        self.u.fill(0)
        # Static to allow for compile-time branch discarding.
        for i, j in ti.ndrange(
            (ti.static(0 if self.wrap else 1), self.n - ti.static(0 if self.wrap else 1)),
            (ti.static(0 if self.wrap else 1), self.n - ti.static(0 if self.wrap else 1))):
            rho = 0.0
            u = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                # Not sure if ternary operator allows for compile-time static branch discarding.
                ni = (i + self.coords[k].x + self.n) % self.n if ti.static(self.wrap) else (i + self.coords[k].x)
                nj = (j + self.coords[k].y + self.n) % self.n if ti.static(self.wrap) else (j + self.coords[k].y)
                self.update_grid[ni, nj][k] = self.grid[i, j][k]
                rho += self.update_grid[i, j][k]
                u += self.coords[k] * self.update_grid[i, j][k]
            self.rho[i, j] = rho
            ''''''''
            self.u[i, j] = (u / rho) if rho > 0 else tm.vec2([0, 0])
            # self.u[i, j] *= 1 / self.rho[i, j] * (self.rho[i, j] != 0)

    @ti.kernel
    def collide(self):
        for i, j in ti.ndrange(
            (ti.static(0 if self.wrap else 1), self.n - ti.static(0 if self.wrap else 1)),
            (ti.static(0 if self.wrap else 1), self.n - ti.static(0 if self.wrap else 1))):
            u = self.u[i, j]
            for k in ti.static(range(9)):
                feq = self.w[k] * self.rho[i, j] * (1 + 3 * tm.dot(self.coords[k], u) + 9 / 2 * tm.dot(self.coords[k], u) ** 2 - 3/2 * tm.dot(u, u))
                # print(self.tau_inv * (feq - self.update_grid[i, j][k]))
                self.update_grid[i, j][k] += self.tau_inv * (feq - self.update_grid[i, j][k])

    # @ti.kernel
    # def bounce_boundary(self):
    #     for i in ti.ndrange(self.n):
    #         self.

    def update(self):
        self.grid.copy_from(self.update_grid)

    @ti.kernel
    def get_velocity_magnitude(self):
        for i, j in self.grid:
            self.disp[i, j] = self.u[i, j].norm()

    def apply_colormap(self, data):
        norm_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        colormap = cm.viridis(norm_data)
        return (colormap[:, :, :3] * 255).astype(np.uint8)

    @ti.kernel
    def normalize_and_map(self):
        for i, j in self.disp:
            self.max_val[None] = ti.max(self.max_val[None], self.disp[i, j])
        for i, j in self.disp:
            norm_val = ti.cast(self.disp[i, j] / self.max_val[None] * 255, ti.i32)
            norm_val = ti.min(ti.max(norm_val, 0), 255)
            for c in ti.static(range(3)):
                self.rgb_image[i, j][c] = ti.u8(self.colormap[norm_val][c] * 255)

    def display(self):
        gui = ti.GUI('LBM Simulation', (self.n, self.n))
        self.update_grid.copy_from(self.grid)

        while gui.running:
            self.get_velocity_magnitude()
            self.normalize_and_map()
            gui.set_image(self.rgb_image)
            self.stream_and_collide()
            self.collide()
            self.update()

            gui.show()


L = LBM()
L.display()
