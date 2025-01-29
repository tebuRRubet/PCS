import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.cm as cm
from obstacles import translate_scale_rotate, is_in_obstacle


ti.init(arch=ti.gpu)

CYLINDER, EGG, AIRFOIL = 0, 1, 2


def precompute_colormap():
    import matplotlib.cm as cm
    viridis = cm.get_cmap('viridis', 256)
    # Extract RGB values
    colors = viridis(np.linspace(0, 1, 256))[:, :3]
    return colors.astype(np.float32)


@ti.data_oriented
class LBM:
    def __init__(self, n=512, tau=0.55, rho0=1.0, inlet_val=0.15):
        self.rho0 = rho0
        self.dirs = ti.Matrix([(-1, 1), (0, 1), (1, 1),
                               (-1, 0), (0, 0), (1, 0),
                               (-1, -1), (0, -1), (1, -1)]).transpose()
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
        self.inlet_val = inlet_val

        block = 128
        self.boundary_mask = ti.field(ti.i8)
        self.b_sparse_mask = ti.root.pointer(ti.ij, (n//block, n//block))
        self.b_sparse_mask.bitmasked(ti.ij, (block, block)).place(self.boundary_mask)
        self.obstacle = AIRFOIL
        self.cylinder_r = n // 20
        self.a = 0.041
        self.b = 0.272
        self.init_grid(rho0)

    @ti.func
    def feq(self, weight, rho, cm, vel):
        return weight * rho * (1 + 3 * cm + 4.5 * cm ** 2 - 1.5 * vel)

    @ti.kernel
    def init_grid(self, rho0: ti.types.f64):
        for i, j in self.f1:
            # Calculates velocity vector in one step
            vel = (self.dirs @ self.f1[i, j] / rho0) if rho0 > 0 else tm.vec2([0, 0])
            self.vel[i, j] = vel.norm()
            di, dj = translate_scale_rotate(i, j, self.n//2, self.n//2, 4, 5.0)

            if is_in_obstacle(di, dj, self.obstacle, self.cylinder_r, self.a, self.b):
                self.boundary_mask[i, j] = 1

            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f1[i, j][k] = self.feq(self.w[k], rho0, cm, tm.dot(vel, vel))

    def apply_colormap(self, data):
        norm_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        colormap = cm.viridis(norm_data)
        return (colormap[:, :, :3] * 255).astype(np.uint8)

    @ti.kernel
    def normalize_and_map(self):
        for i, j in self.vel:
            self.max_val[None] = ti.atomic_max(self.max_val[None], self.vel[i, j])
        for i, j in self.vel:
            norm_val = ti.cast(self.vel[i, j] / self.max_val[None] * (255), ti.i32)

            norm_val = ti.min(ti.max(norm_val, 0), (255))
            for c in ti.ndrange(3):
                self.rgb_image[i, j][c] = ti.u8(self.colormap[norm_val][c] * (255))
                if self.boundary_mask[i, j]:
                    self.rgb_image[i, j][c] = (255)

    @ti.kernel
    def stream(self):
        for i, j in ti.ndrange((1, self.n - 1), (1, self.n - 1)):
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = self.f1[i, j][k]

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange(ti.static((1, self.n - 1)), ti.static((1, self.n - 1))):
            rho = self.f1[i, j].sum()
            # Calculates velocity vector in one step
            vel = self.dirs @ self.f1[i, j] / rho
            self.vel[i, j] = vel.norm()
            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = (1 - self.tau_inv) * self.f1[i, j][k] + self.tau_inv * self.feq(self.w[k], rho, cm, tm.dot(vel, vel))

    @ti.kernel
    def apply_inlet(self):
        for i in ti.ndrange(self.n):
            self.f1[1, i][5] = 0.15

    @ti.kernel
    def boundary_condition(self):
        for i, j in self.boundary_mask:
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, 8-k], j + self.dirs[1, 8-k]][8-k] = self.f2[i, j][k]

    """Check performance"""
    @ti.kernel
    def update(self):
        # self.grid.copy_from(self.update_grid)
        for i, j in self.f2:
            self.f1[i, j] = self.f2[i, j]

    @ti.kernel
    def max_vel(self):
        for i, j in self.vel:
            ti.atomic_max(self.max_val[None], self.vel[i, j])

    def display(self):
        gui = ti.GUI('LBM Simulation', (self.n, self.n))
        self.f2.copy_from(self.f1)
        self.apply_inlet()
        self.stream()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            self.max_vel()
            self.normalize_and_map()
            gui.set_image(self.rgb_image)
            gui.show()

            for _ in range(100):
                self.max_vel()
                self.apply_inlet()
                self.collide_and_stream()
                self.boundary_condition()
                self.update()


L = LBM()
L.display()
