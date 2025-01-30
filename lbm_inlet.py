import os
import subprocess
import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.cm as color_m
from obstacles import rotate, is_in_obstacle


ti.init(arch=ti.gpu)

CYLINDER, EGG, AIRFOIL = 0, 1, 2
UP_LEFT, UP, UP_RIGHT, LEFT, MID, RIGHT, DOWN_LEFT, DOWN, DOWN_RIGHT = range(9)


def precompute_colormap():
    viridis = color_m.get_cmap('plasma', 256)
    # Extract RGB values
    colors = viridis(np.linspace(0, 1, 256))[:, :3]
    return colors.astype(np.float32)


@ti.data_oriented
class LBM:
    def __init__(self, width=1024, height=512, tau=0.55, rho0=1.0, inlet_val=0.15, block_size=128, theta=0):
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
        self.min_val = ti.field(ti.f32, shape=())
        self.min_val.fill(1e-8)
        self.inlet_val = inlet_val

        self.temp_max = ti.field(ti.f32, shape=())
        self.temp_max.fill(1e-8)
        self.cont = ti.field(ti.i8, shape=())
        self.cont.fill(1)


        self.drag = ti.field(dtype=ti.f32, shape=())
        self.lift = ti.field(dtype=ti.f32, shape=())


        self.boundary_mask = ti.field(ti.i8)
        self.b_sparse_mask = ti.root.pointer(ti.ij, (width // block_size, height // block_size))
        self.b_sparse_mask.bitmasked(ti.ij, (block_size, block_size)).place(self.boundary_mask)

        obstacle = AIRFOIL
        center_x = self.width // 2
        center_y = self.height // 2
        scale = width // 8
        a = 0.026
        b = 0.077
        r = 0.918
        # scale = width//20
        # a = 0
        # b = 0
        # r = 1

        self.init_grid(rho0, obstacle, center_x, center_y, scale, a, b, r, theta)
        obstacle = CYLINDER
        scale = width//20
        a = 0
        b = 0
        r = 1
        theta = 0
        # self.init_grid(rho0, obstacle, center_x - 400, center_y, scale, a, b, r, theta)

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

            # if j == 0 or j == self.height - 1:
            #     if j == self.height - 1 and i == 300:
            #         print("Marked")
            #     self.boundary_mask[i, j] = 1

    @ti.kernel
    def normalize_and_map(self):
        for i, j in self.rgb_image:
            norm_val = ti.cast((self.vel[i, j] - self.min_val[None]) / (self.max_val[None] - self.min_val[None]) * (255), ti.i32)

            norm_val = ti.min(ti.max(norm_val, 0), (255))
            for c in ti.static(range(3)):
                self.rgb_image[i, j][c] = ti.u8(self.colormap[norm_val][c] * (255))
        self.draw_boundary()

    @ti.func
    def draw_boundary(self):
        for i, j in self.boundary_mask:
            # if i == 300 and j > self.width - 100:
            #     print(i, j)
            for c in ti.static(range(3)):
                self.rgb_image[i, j][c] = 255
        # for i in ti.ndrange(self.width):
        #     for c in ti.static(range(3)):
        #         self.rgb_image[i, 0][c] = 255
        #         self.rgb_image[i, self.height - 1][c] = 255


    @ti.kernel
    def stream(self):
        for i, j in ti.ndrange((1, self.width - 1), (1, self.height - 1)):
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = self.f1[i, j][k]

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange(ti.static((1, self.width - 1)), ti.static((1, self.height - 1))):
            rho = self.f1[i, j].sum()
            vel = self.dirs @ self.f1[i, j] / rho
            self.vel[i, j] = vel.norm()
            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = (1 - self.tau_inv) * self.f1[i, j][k] + self.tau_inv * self.feq(self.w[k], rho, cm, tm.dot(vel, vel))

    # @ti.kernel
    # def apply_inlet(self, t: ti.types.f32, T: ti.types.i32):
    #     xv = self.inlet_val * (1 - ti.exp(-t / T))
    #     for i in ti.ndrange((1, self.height - 1)):
    #         rho = self.f1[1, i].sum()
    #         vel = self.dirs @ self.f1[0, i] / rho
    #         for k in ti.static((2, 5, 8)):
    #             cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
    #             self.f1[1, i][k] = self.feq(self.w[k], rho / (1 - xv), cm, 0.01)


    @ti.kernel
    def apply_inlet(self, t: ti.types.f32, T: ti.types.i32):
        xv = self.inlet_val * (1 - ti.exp(-t / T))  # Inlet velocity
        for i in ti.ndrange((1, self.height - 1)):
            rho = self.f2[1, i].sum() / (1 - xv)
            # Compute density using Zou-He formulation
            # rho = (self.f1[1, i][0] + self.f1[1, i][2] + self.f1[1, i][4] +
                # 2 * (self.f1[1, i][3] + self.f1[1, i][6] + self.f1[1, i][7])) / (1 - xv)

            # Zou-He updates for missing right-moving distributions
            self.f2[1, i][RIGHT] = self.f2[1, i][LEFT] + (2 / 3) * rho * xv
            self.f2[1, i][UP_RIGHT] = self.f2[1, i][UP_LEFT] + (1 / 6) * rho * xv
            self.f2[1, i][DOWN_RIGHT] = self.f2[1, i][DOWN_LEFT] + (1 / 6) * rho * xv


    @ti.kernel
    def apply_outlet(self):
        for i in ti.ndrange(self.height):
            self.f2[self.width - 1, i] = self.f2[self.width - 2, i]


    @ti.kernel
    def boundary_condition(self):
        self.drag[None] = 0.0
        self.lift[None] = 0.0
        for i, j in self.boundary_mask:
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, 8 - k], j + self.dirs[1, 8 - k]][8 - k] = self.f2[i, j][k]
                self.drag[None] += 2 * self.f2[i, j][k] * self.dirs[0, k]
                self.lift[None] += 2 * self.f2[i, j][k] * self.dirs[1, k]

    @ti.kernel
    def upper_bound(self):
        for i in ti.ndrange((1, self.width - 1)):
            self.f2[i + self.dirs[0, DOWN_RIGHT], self.height - 1 + self.dirs[1, DOWN_RIGHT]][DOWN_RIGHT] = self.f2[i, self.height - 1][UP_LEFT]
            self.f2[i + self.dirs[0, DOWN], self.height - 1 + self.dirs[1, DOWN]][DOWN] = self.f2[i, self.height - 1][UP]
            self.f2[i + self.dirs[0, DOWN_LEFT], self.height - 1 + self.dirs[1, DOWN_LEFT]][DOWN_LEFT] = self.f2[i, self.height - 1][UP_RIGHT]

            self.f2[i + self.dirs[0, UP_RIGHT], self.dirs[1, UP_RIGHT]][UP_RIGHT] = self.f2[i, 0][DOWN_LEFT]
            self.f2[i + self.dirs[0, UP], self.dirs[1, UP]][UP] = self.f2[i, 0][DOWN]
            self.f2[i + self.dirs[0, UP_LEFT], self.dirs[1, UP_LEFT]][UP_LEFT] = self.f2[i, 0][DOWN_RIGHT]



    """Check performance"""
    @ti.kernel
    def update(self):
        # self.f1.copy_from(self.f2)
        for i, j in self.f2:
            self.f1[i, j] = self.f2[i, j]

    @ti.kernel
    def max_vel(self):
        curr_max = 1e-8
        curr_min = float('inf')
        for i, j in ti.ndrange((1, self.width - 1), (1, self.height - 1)):
            ti.atomic_max(curr_max, self.vel[i, j])
            ti.atomic_min(curr_min, self.vel[i, j])

        self.max_val[None] = self.max_val[None] * 0.9 + curr_max * 0.1
        self.min_val[None] = self.min_val[None] * 0.9 + curr_min * 0.1


    @ti.kernel
    def upper_sum(self):
        val = 0.0
        for i in ti.ndrange(self.width):
            ti.atomic_add(val, self.f2[i, self.height - 1].sum())
        print(f"Upper sum: {val}", end="\t")

    @ti.kernel
    def lower_sum(self):
        val = 0.0
        for i in ti.ndrange(self.width):
            ti.atomic_add(val, self.f2[i, 0].sum())
        print(f"Lower sum: {val}", end="\t")

    @ti.kernel
    def left_sum(self):
        val = 0.0
        for i in ti.ndrange(self.height):
            ti.atomic_add(val, self.f2[0, i].sum())
        print(f"Left sum: {val}", end="\t")

    @ti.kernel
    def right_sum(self):
        val = 0.0
        for i in ti.ndrange(self.height):
            ti.atomic_add(val, self.f2[self.width - 1, i].sum())
        print(f"Right sum: {val}")


    def display(self):
        gui = ti.GUI('LBM Simulation', (self.width, self.height))

        # Create folder for saving frames
        output_folder = "lbm_frames"
        os.makedirs(output_folder, exist_ok=True)

        frame_count = 0
        step = 0
        max_step = 5000
        self.f2.copy_from(self.f1)
        self.apply_inlet(step, max_step)
        self.stream()

        drag = []
        lift = []
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT) and step < 10000:
            self.max_vel()
            self.normalize_and_map()
            gui.set_image(self.rgb_image)

            # filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            # gui.show(filename)
            gui.show()
            frame_count += 1
            # print(frame_count)
            if step == max_step:
                print("Inlet max")

            for _ in range(10):
                step += 1
                self.apply_inlet(step, max_step)
                self.apply_outlet()
                self.collide_and_stream()
                self.boundary_condition()
                # self.upper_bound()
                self.update()

                drag.append(self.drag[None])
                lift.append(self.lift[None])
                print("ratio:", self.lift[None] / self.drag[None], "drag:", self.drag[None], "lift:", self.lift[None], step)

                # if not self.cont[None]:
                #     sleep(0.3)
                #     self.max_vel()
                #     self.normalize_and_map()
                #     gui.set_image(self.rgb_image)
                #     gui.show()
                # while gui.running:
                #     gui.set_image(self.rgb_image)
                #     gui.show()
            # self.left_sum()
            # self.upper_sum()
            # self.right_sum()
            # self.lower_sum()
        return (drag, lift)
        # print("Simulation ended. Generating GIF...")
        # subprocess.run(["python3", "generate_gif.py"])

import matplotlib.pyplot as plt
# final = 20
# amount = 5

# drags = []
# lifts = []
# for i in range(amount):
#     L = LBM(theta=final - amount)
#     drag, lift = L.display()
#     drags.append(drag)
#     lifts.append(lift)
#     print("DONE WITH", i)

#     np.savetxt(f"drags{final}.csv", drags, delimiter =", ", fmt ='% s')
#     np.savetxt(f"lifts{final}.csv", lifts, delimiter =", ", fmt ='% s')

drag = []
lift = []

for i in range(4):
    data = np.loadtxt(f"drags{5 * (i + 1)}.csv", delimiter=",", dtype=float)
    drag.extend(data.tolist())
    data = np.loadtxt(f"lifts{5 * (i + 1)}.csv", delimiter=",", dtype=float)
    lift.extend(data.tolist())


from scipy.optimize import curve_fit

def parabola(x, a, b, c):
    return a * (x - b)**2 + c

last = 5000
its = 2000
# print(drag[5])
mean_drag = np.array([np.mean(drag[i][-last:-last + its]) for i in range(len(drag))])
mean_lift = np.array([np.mean(lift[i][-last:-last + its]) for i in range(len(lift))])
mean_ratio = mean_lift / mean_drag
angles = [i for i in range(20)]

start, end = 3, 12
fit, _ = curve_fit(parabola, angles[start:end], mean_ratio[start:end])

x_peak = fit[1]
print(x_peak)

plt.plot(angles[start:end], parabola(angles[start:end], *fit), linestyle='--', label=f"peak x: {x_peak}")
plt.plot(mean_ratio)
plt.xlabel("Angle of the airfoil in degrees")
plt.ylabel("Lift/Drag ratio")
# plt.grid(1)
plt.legend()
plt.show()


# for i, d in enumerate(drag[::5]):
#     plt.plot(d, label=f"{i * 5}")
# plt.xlabel("step")
# plt.ylabel("drag")
# plt.legend()
# plt.show()

# for i, l in enumerate(lift[::5]):
#     plt.plot(l, label=f"{i * 5}")
# plt.xlabel("step")
# plt.ylabel("lift")
# plt.legend()
# plt.show()