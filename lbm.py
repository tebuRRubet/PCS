import os
import subprocess
import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.cm as color_m
from obstacles import rotate, is_in_obstacle


ti.init(arch=ti.gpu)

CYLINDER, EGG, AIRFOIL = 0, 1, 2


def precompute_colormap():
    viridis = color_m.get_cmap('plasma', 256)
    # Extract RGB values
    colors = viridis(np.linspace(0, 1, 256))[:, :3]
    return colors.astype(np.float32)


@ti.data_oriented
class LBM:
    def __init__(self, width=1024, height=512, tau=0.55, rho0=1.0, inlet_val=0.15, block_size=128):
        if width % block_size or height % block_size:
            print(f"Error, block_size ({block_size}) must be a divisor of n ({width}) and m ({height})!")
            print(f"{width} = {width // block_size} * {block_size} + {width % block_size}.")
            print(f"{height} = {height // block_size} * {block_size} + {height % block_size}.")
            exit()
        if not isinstance(width, int) or not isinstance(height, int) or not isinstance(block_size, int):
            print("Width, height and blocksize must be integers.")
        if block_size == 1:
            print("Block size of 1 is not allowed.")
        if width > 2000 or height > 2000:
            print("Warning, simulation grid and window are very large.")
        if inlet_val > 1 / tm.sqrt(3):
            print("Warning, inlet velocity higher than system's speed of sound. This may cause instabilty.")
        self.rho0 = rho0
        self.dirs = ti.Matrix([(-1, 1), (0, 1), (1, 1),
                               (-1, 0), (0, 0), (1, 0),
                               (-1, -1), (0, -1), (1, -1)]).transpose()
        self.height, self.width = height, width
        self.disp = ti.field(dtype=ti.f32, shape=(width, height))
        self.f1 = ti.Vector.field(n=9, dtype=ti.f32, shape=(width, height))
        self.f2 = ti.Vector.field(n=9, dtype=ti.f32, shape=(width, height))
        self.tau_inv = 1 / tau
        self.vel = ti.field(dtype=ti.f32, shape=(width, height))
        self.w = ti.field(dtype=ti.f32, shape=(9,))
        self.w.from_numpy(np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 36)

        self.colormap = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
        colors = precompute_colormap()
        self.colormap.from_numpy(colors)
        self.rgb_image = ti.Vector.field(3, dtype=ti.u8, shape=(width, height))
        self.max_val = ti.field(ti.f32, shape=())
        self.max_val.fill(1e-8)
        self.inlet_val = inlet_val



        self.temp_max = ti.field(ti.f32, shape=())
        self.temp_max.fill(1e-8)
        self.cont = ti.field(ti.i8, shape=())
        self.cont.fill(1)





        self.boundary_mask = ti.field(ti.i8)
        self.b_sparse_mask = ti.root.pointer(ti.ij, (width // block_size, height // block_size))
        self.b_sparse_mask.bitmasked(ti.ij, (block_size, block_size)).place(self.boundary_mask)

        obstacle = CYLINDER
        center_x = self.width // 2
        center_y = self.height // 2
        scale = width // 8
        # a = 0.026
        # b = 0.077
        # r = 0.918
        scale = width//20
        a = 0
        b = 0
        r = 1
        theta = 0
        self.init_grid(rho0, obstacle, center_x - 300, center_y, scale, a, b, r, theta)

    @ti.func
    def feq(self, weight, rho, cm, vel):
        return weight * rho * (1 + 3 * cm + 4.5 * cm ** 2 - 1.5 * vel)

    @ti.kernel
    def init_grid(self, rho0: ti.types.f64, obstacle: ti.types.i8, center_x: ti.types.i16, center_y: ti.types.i16, scale: ti.types.i16, a: ti.types.f64, b: ti.types.f64, r: ti.types.f64, theta: ti.types.f64):
        for i, j in self.f1:
            # Calculates velocity vector in one step
            vel = (self.dirs @ self.f1[i, j] / rho0) if rho0 > 0 else tm.vec2([0, 0])
            self.vel[i, j] = vel.norm()
            di, dj = rotate(i, j, center_x, center_y, theta)

            if is_in_obstacle(di, dj, obstacle, center_x, center_y, scale, a, b, r):
                self.boundary_mask[i, j] = 1

            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f1[i, j][k] = self.feq(self.w[k], rho0, cm, tm.dot(vel, vel))

    @ti.kernel
    def normalize_and_map(self):
        for i, j in self.rgb_image:
            norm_val = ti.cast(self.vel[i, j] / self.max_val[None] * (255), ti.i32)

            norm_val = ti.min(ti.max(norm_val, 0), (255))
            for c in ti.static(range(3)):
                self.rgb_image[i, j][c] = ti.u8(self.colormap[norm_val][c] * (255))
        self.draw_boundary()

    @ti.func
    def draw_boundary(self):
        for i, j in self.boundary_mask:
            for c in ti.static(range(3)):
                self.rgb_image[i, j][c] = 255

    @ti.kernel
    def stream(self):
        for i, j in ti.ndrange((1, self.width - 1), (1, self.height - 1)):
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = self.f1[i, j][k]

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange(ti.static((1, self.width - 1)), ti.static((1, self.height - 1))):
            rho = self.f1[i, j].sum()
            # Calculates velocity vector in one step
            vel = self.dirs @ self.f1[i, j] / rho
            self.vel[i, j] = vel.norm()
            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = (1 - self.tau_inv) * self.f1[i, j][k] + self.tau_inv * self.feq(self.w[k], rho, cm, tm.dot(vel, vel))

    @ti.kernel
    def apply_inlet(self):
        xv = 0.2
        yv = 0.0
        for i in ti.ndrange(self.height):
            rho = self.f1[1, i].sum()
            vel = self.dirs @ self.f1[0, i] / rho
            for k in ti.static((2, 5, 8)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f1[1, i][k] = self.feq(self.w[k], rho / (1 - xv), cm, 0.01)
            # self.f1[1, i][2] = 0.3 * self.inlet_val
            # self.f1[1, i][5] = self.inlet_val
            # self.f1[1, i][8] = 0.3 * self.inlet_val

    @ti.kernel
    def apply_outlet(self):
        for i in ti.ndrange(self.height):
            x = self.width - 10
            if i == 300:
                vel = self.dirs @ self.f2[x, i] / self.f2[x, i].sum()
                ti.atomic_max(self.temp_max[None], vel[0])
                print(vel, self.temp_max[None])
            self.f2[self.width - 1, i] = self.f2[self.width - 2, i]
            if i == 300:
                vel = self.dirs @ self.f2[x, i] / self.f2[x, i].sum()
                ti.atomic_max(self.temp_max[None], vel[0])
                print(vel, self.temp_max[None])
                print()
            if self.temp_max[None] >= 0.035135:
                self.cont[None] = 0

    @ti.kernel
    def boundary_condition(self):
        for i, j in self.boundary_mask:
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, 8 - k], j + self.dirs[1, 8 - k]][8 - k] = self.f2[i, j][k]

    """Check performance"""
    @ti.kernel
    def update(self):
        # self.f1.copy_from(self.f2)
        for i, j in self.f2:
            self.f1[i, j] = self.f2[i, j]

    @ti.kernel
    def max_vel(self):
        curr_max = 1e-8
        for i, j in self.vel:
            ti.atomic_max(curr_max, self.vel[i, j])
        self.max_val[None] = self.max_val[None] * 0.9 + curr_max * 0.1

    def display(self):
        gui = ti.GUI('LBM Simulation', (self.width, self.height))

        # Create folder for saving frames
        output_folder = "lbm_frames"
        os.makedirs(output_folder, exist_ok=True)

        # frame_count = 0

        self.f2.copy_from(self.f1)
        self.apply_inlet()
        self.stream()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            self.max_vel()
            self.normalize_and_map()
            gui.set_image(self.rgb_image)

            # filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            # gui.show(filename)
            gui.show()
            # frame_count += 1
            # print(frame_count)

            for _ in range(100):
                self.apply_inlet()
                self.apply_outlet()
                self.collide_and_stream()
                self.boundary_condition()
                self.update()
                # if not self.cont[None]:
                #     sleep(0.3)
                #     self.max_vel()
                #     self.normalize_and_map()
                #     gui.set_image(self.rgb_image)
                #     gui.show()
                # while gui.running:
                #     gui.set_image(self.rgb_image)
                #     gui.show()

        # print("Simulation ended. Generating GIF...")
        # subprocess.run(["python", "generate_gif.py"])

from time import sleep
L = LBM()
L.display()
