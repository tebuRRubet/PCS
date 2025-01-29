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
    def __init__(self, n=1024, tau=0.55, rho0=1.0, inlet_val=0.15):
        self.rho0 = rho0
        self.dirs = ti.Matrix([(-1, 1), (0, 1), (1, 1),
                               (-1, 0), (0, 0), (1, 0),
                               (-1, -1), (0, -1), (1, -1)]).transpose()
        print(self.dirs)
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
        self.boundary_mask = ti.field(dtype=ti.i8, shape=(n, n))
        self.inlet_val = inlet_val

        # block = 128
        # self.boundary_mask = ti.field(ti.i8)
        # self.b_sparse_mask = ti.root.pointer(ti.ij, (n//block, n//block))
        # self.b_sparse_mask.bitmasked(ti.ij, (block, block)).place(self.boundary_mask)
        self.obstacle = AIRFOIL
        self.cylinder_r = n // 20
        self.a = 0.041
        self.b = 0.272

        for i in range(n):
            self.boundary_mask[0, i] = 1

        # for i in range(n):
        #     self.boundary_mask[i, n - 1] = b_types[1]

        # for i in range(n):
        #     self.boundary_mask[n - 1, i] = b_types[2]

        # for i in range(n):
        #     self.boundary_mask[i, 0] = b_types[3]






        self.init_grid(rho0)

        # for i in range(11):
        #     for j in range(11):
        #         for k in range(9):
        #             self.f1[i + n//2 - 5, j + n//2 - 5][k] += self.w[k] * 0.15
        print("Init done")

    # @ti.kernel
    # def init_grid(self):
    #     for i, j in self.f1:
    #         for k in ti.static(range(9)):
    #             self.f1[i, j][k] = self.w[k]


    @ti.func
    def feq(self, weight, rho, cm, vel):
        return weight * rho * (1 + 3 * cm + 4.5 * cm ** 2 - 1.5 * vel)

    @ti.kernel
    def init_grid(self, rho0: ti.types.f64):
        for i, j in self.f1:
            # Calculates velocity vector in one step
            vel = (self.dirs @ self.f1[i, j] / rho0) if rho0 > 0 else tm.vec2([0, 0])
            self.vel[i, j] = vel.norm()
            di, dj = translate_scale_rotate(i, j, self.n//2, self.n//2, 4, 0.0)

            if is_in_obstacle(di, dj, self.obstacle, self.cylinder_r, self.a, self.b):
                self.boundary_mask[i, j] = 1
                # self.f1[i, j].fill(0)
            # else:
            #     for k in ti.static(range(9)):
            #         cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
            #         feq = self.w[k] * rho0 * (1 + 3 * cm + 4.5 * cm ** 2 - 1.5 * tm.dot(vel, vel))
            #         self.f1[i, j][k] = feq
            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f1[i, j][k] = self.feq(self.w[k], rho0, cm, tm.dot(vel, vel))

    def apply_colormap(self, data):
        norm_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        colormap = cm.viridis(norm_data)
        return (colormap[:, :, :3] * 255).astype(np.uint8)

    @ti.kernel
    def normalize_and_map(self):
        # print(self.total_density())
        for i, j in self.vel:
            self.max_val[None] = ti.atomic_max(self.max_val[None], self.vel[i, j])
        curr_max = 1e-8
        for i, j in self.vel:
            curr_max = ti.max(curr_max, self.vel[i, j])
        # print(self.max_val[None], curr_max)

        for i, j in self.vel:
            # norm_val = ti.cast(self.vel[i, j] / self.max_val[None] * (255), ti.i32)
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
    def boundary_condition(self, step: ti.types.i16, step_max: ti.types.i16):
        # Streamt nog niet!!!!
        # for i, j in self.boundary_mask:
        #     self.f1[i, j] += -self.f2[i, j] + self.inlet_val * self.boundary_mask[i, j] == 1 + self.reverse_vector(self.f2[i, j]) * self.boundary_mask[i, j] == 2



        # # CHECK VOOR BELANGRIJKE DEBUG.
        for i, j in self.f1:
            if self.boundary_mask[i, j]:
                if i == 0 and step < step_max:
                    self.f2[i, j][3] = 0.15
                # if i == self.n - 1:
                    # self.vel[i, j] = 0


                # Voormalig was self.f1[i + self.dirs[0, k], j + self.dirs[1, k]] = self.reverse_vector(self.f2[i, j]). Dat kan niet kloppen.
                rv = self.reverse_vector(self.f2[i, j])
                for k in ti.static(range(9)):
                    # self.f1[i + self.dirs[0, k], j + self.dirs[1, k]][8-k] = self.f2[i,j][k]
                    self.f1[i + self.dirs[0, 8-k], j + self.dirs[1, 8-k]][8-k] = self.f2[i, j][k]
            else:
                self.f1[i, j] = self.f2[i, j]

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

    @ti.kernel
    def min_max_dens(self):
        mini = 2.0
        maxi = 0.0
        for i, j in self.f1:
            ti.atomic_min(mini, self.f1[i, j].sum())
            ti.atomic_max(maxi, self.f1[i, j].sum())
            if i == self.n // 2 and j > self.n // 2 and self.f1[i, j].sum() < 1e-2:
                print("NUL DENSITYY!Y!Y!Y!Y!YY!Y!")
                print(i, j)
                print(self.f1[i,j])
                print()
        print(mini, maxi)

    @ti.kernel
    def get_disp(self):
        for i, j in self.f1:
            # self.rgb_image[i, j][0] = self.f1[i, j].sum() * 255 / 2
            self.rgb_image[i, j][1] = self.vel[i, j] * 255


    def display(self):
        gui = ti.GUI('LBM Simulation', (self.n, self.n))
        self.f2.copy_from(self.f1)
        self.stream()
        step = 0
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            self.max_vel()
            # self.get_disp()
            # gui.set_image(self.rgb_image)
            self.normalize_and_map()
            gui.set_image(self.rgb_image)
            gui.show()
            step += 1

            for _ in range(100):
                # self.max_vel()
                self.min_max_dens()
                # print(self.max_val[None])
                self.collide_and_stream()
                # self.stream()
                self.update()

                self.boundary_condition(step, 100)
                # exit()


L = LBM()
L.display()
