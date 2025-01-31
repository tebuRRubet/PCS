import os
import subprocess
import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.cm as color_m
from obstacles import rotate, is_in_obstacle


CYLINDER, EGG, AIRFOIL = 0, 1, 2
UP_LEFT, UP, UP_RIGHT, LEFT, MID, RIGHT, DOWN_LEFT, DOWN, DOWN_RIGHT = range(9)


def precompute_colormap(colormap='plasma'):
    colormap = color_m.get_cmap(colormap, 256)
    colors = colormap(np.linspace(0, 1, 256))[:, :3]
    return colors.astype(np.float32)


@ti.data_oriented
class LBM:
    def __init__(self, obstacle=AIRFOIL, l_offset=0, b_offset=0, scale=100, theta=5.0, tau=0.8,
                 rho0=1.0, inlet_val=0.15, max_inlet_steps=5000, tunnel=False,
                 width=1024, height=512, block_size=128,
                 max_sim_step=float('inf'), steps_per_frame=10, calc_ltdr=False, make_gif=False,
                 colormap='plasma'):
        """
        obstacle (int): The type of obstacle in the simulation (default: AIRFOIL).
        l_offset (int): Horizontal offset of the obstacle from the center to the left (default: 0).
        b_offset (int): Vertical offset of the obstacle from the center to the bottom (default: 0).
        scale (int):    Scaling factor for the obstacle (default: 100).
        theta (float):  Rotation angle for the obstacle in degrees (counter-clockwise)
                        (default: 5.0).
        tau (float):    Relaxation time, affects viscosity and stability (default: 0.8).
        rho0 (float):   Initial density of the fluid (default: 1.0).
        inlet_val (float):  Velocity at the inlet (default: 0.15).
        max_inlet_steps (int):  Number of steps to increasing the inlet velocity from zero to
                                inlet_val (default: 5000).
        tunnel (bool):  Set walls on the top and bottom (default: False).
        width (int):    Width of the simulation (and pixel) grid (default: 1024).
        height (int):   Height of the simulation grid (and pixel) (default: 512).
        block_size (int):   Size of sparse object mask blocks (default: 128). Must divide width and
                            height.
        max_sim_step (int/float):   Maximum number of simulation steps (default: infinite).
        steps_per_frame (int):  Number of simulation steps per visualisation frame (default: 10).
        calc_ltdr (bool):   Whether to calculate the lift-to-drag ratio (default: False).
        make_gif (bool):    Whether to generate a GIF of the simulation (default: False).
        colormap (str):     The mpl colormap used for visualisation (default: 'plasma').
        """
        if width < 10 or height < 10:
            print("Warning, extremely small grids may lead to improper behaviour and " +
                  "are not properly supported.")
        if width % block_size or height % block_size:
            print(f"Error, block_size ({block_size}) must be a divisor of n ({width})" +
                  f"and m ({height})!")
            print(f"{width} = {width // block_size} * {block_size} + {width % block_size}.")
            print(f"{height} = {height // block_size} * {block_size} + {height % block_size}.")
            exit()
        if not isinstance(width, int) or not isinstance(height, int) or not isinstance(
                                                                                block_size, int):
            print("Width, height and blocksize must be integers.")
        if block_size == 1:
            print("Block size of 1 is not allowed.")
        if width > 2000 or height > 2000:
            print("Warning, simulation grid and window are very large.")
        if steps_per_frame > 100:
            print("Warning, over 100 simulation steps per frame is can cause a low frame rate.")
        if inlet_val > 1 / tm.sqrt(3):
            print("Warning, inlet velocity higher than system's speed of sound. " +
                  "This may cause instabilty.")

        self.height, self.width = height, width
        self.max_sim_step = max_sim_step
        self.ltd_ratio = calc_ltdr
        self.steps_per_frame = steps_per_frame
        self.max_inlet_steps = max_inlet_steps
        self.make_gif = make_gif

        self.rho0 = rho0
        self.dirs = ti.Matrix([(-1, 1), (0, 1), (1, 1),
                               (-1, 0), (0, 0), (1, 0),
                               (-1, -1), (0, -1), (1, -1)]).transpose()
        self.disp = ti.field(dtype=ti.f32, shape=(width, height))
        self.f1 = ti.Vector.field(n=9, dtype=ti.f32, shape=(width, height))
        self.f2 = ti.Vector.field(n=9, dtype=ti.f32, shape=(width, height))
        self.tau_inv = 1 / tau
        self.vel = ti.field(dtype=ti.f32, shape=(width, height))
        self.w = ti.field(dtype=ti.f32, shape=(9,))
        self.w.from_numpy(np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 36)

        self.colormap = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
        colors = precompute_colormap(colormap)
        self.colormap.from_numpy(colors)
        self.rgb_image = ti.Vector.field(3, dtype=ti.u8, shape=(width, height))
        self.max_val = ti.field(ti.f32, shape=())
        self.max_val.fill(1e-8)
        self.min_val = ti.field(ti.f32, shape=())
        self.min_val.fill(1e-8)
        self.inlet_val = inlet_val

        self.drag = ti.field(dtype=ti.f32, shape=())
        self.lift = ti.field(dtype=ti.f32, shape=())

        # Sparse datastructure to prevent warp divergence and simplify indexing.
        self.object_mask = ti.field(ti.i8)
        self.o_sparse_mask = ti.root.pointer(ti.ij, (width // block_size, height // block_size))
        self.o_sparse_mask.bitmasked(ti.ij, (block_size, block_size)).place(self.object_mask)

        self.tunnel = tunnel

        # Joukowski airfoil parameters.
        a = 0.026
        b = 0.077
        r = 0.918
        self.init_grid(rho0, obstacle, self.width//2 - l_offset, self.height//2 - b_offset, scale,
                       a, b, r, theta)

    @ti.func
    def feq(self, weight, rho, cm, vel):
        """
        Computes the equilibrium distribution function for the LBM.
        weight (ti float):  Lattice weight for the given direction.
        rho (ti float):     Local fluid density.
        cm (ti float):      Dot product of the lattice velocity direction and the macroscopic
                            velocity.
        vel (ti float):     Squared magnitude of the macroscopic velocity.

        Returns:
        ti float: Equilibrium distribution function value.
        """
        return weight * rho * (1 + 3 * cm + 4.5 * cm ** 2 - 1.5 * vel)

    @ti.kernel
    def init_grid(self, rho0: ti.types.f64, obstacle: ti.types.i8, center_x: ti.types.i16,
                  center_y: ti.types.i16, scale: ti.types.i16, a: ti.types.f64, b: ti.types.f64,
                  r: ti.types.f64, theta: ti.types.f64):
        """
        Initialises the simulation grid, setting velocity fields, density distributions,
        and obstacle placement for the Lattice Boltzmann Method (LBM).

        rho0 (ti float):    Initial density of the fluid.
        obstacle (int):     Obstacle type identifier.
        center_x (int):     X-coordinate of the obstacle's center.
        center_y (int):     Y-coordinate of the obstacle's center.
        scale (int):    Scaling factor for the obstacle size.
        a (ti float):   Parameter defining the Joukowski airfoil shape.
        b (ti float):   Parameter defining the Joukowski airfoil shape.
        r (ti float):   Parameter defining the Joukowski airfoil transformation.
        theta (ti float):   Rotation angle of the obstacle in degrees.
        """
        for i, j in self.f1:
            vel = (self.dirs @ self.f1[i, j] / rho0) if rho0 > 0 else tm.vec2([0, 0])
            self.vel[i, j] = vel.norm()
            di, dj = rotate(i, j, center_x, center_y, theta)
            if is_in_obstacle(di, dj, obstacle, center_x, center_y, scale, a, b, r):
                self.object_mask[i, j] = 1

            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f1[i, j][k] = self.feq(self.w[k], rho0, cm, tm.dot(vel, vel))

    @ti.kernel
    def normalize_and_map(self):
        """
        Normalises velocity values and maps them to RGB colors for visualisation.
        """
        for i, j in self.rgb_image:
            # Normalise to range of velocities
            norm_val = ti.cast((self.vel[i, j] - self.min_val[None]) /
                               (self.max_val[None] - self.min_val[None]) * (255), ti.i32)
            # Clips range.
            norm_val = ti.min(ti.max(norm_val, 0), (255))
            for c in ti.static(range(3)):
                self.rgb_image[i, j][c] = ti.u8(self.colormap[norm_val][c] * (255))
        self.draw_obstacles()

    @ti.func
    def draw_obstacles(self):
        """
        Draws obstacles as white.
        """
        for i, j in self.object_mask:
            for c in ti.static(range(3)):
                self.rgb_image[i, j][c] = 255
        # Branch pruning friendly implementation.
        if ti.static(self.tunnel):
            for i in ti.ndrange((1, self.width - 1)):
                for c in ti.static(range(3)):
                    self.rgb_image[i, 0][c] = 255
                    self.rgb_image[i, self.height - 1][c] = 255

    @ti.kernel
    def stream(self):
        """
        Performs the streaming step in the lattice D2Q9.
        """
        for i, j in ti.ndrange((1, self.width - 1), (1, self.height - 1)):
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = self.f1[i, j][k]

    @ti.kernel
    def collide_and_stream(self):
        """
        Collides and then performs the streaming step in the lattice. Collding-then-streaming
        eliminates some race conditions since collisions are a local operation.
        """
        for i, j in ti.ndrange(ti.static((1, self.width - 1)), ti.static((1, self.height - 1))):
            # Get total density.
            rho = self.f1[i, j].sum()
            # Desnity "should" never be zero. This product calculates the velocity vector in 1 step.
            vel = self.dirs @ self.f1[i, j] / rho
            self.vel[i, j] = vel.norm()
            for k in ti.static(range(9)):
                cm = vel[0] * self.dirs[0, k] + vel[1] * self.dirs[1, k]
                self.f2[i + self.dirs[0, k], j + self.dirs[1, k]][k] = (1 - self.tau_inv) *\
                    self.f1[i, j][k] + self.tau_inv * self.feq(self.w[k], rho, cm, tm.dot(vel, vel))

    @ti.func
    def apply_inlet(self, t, T):
        """
        Applies the inlet boundary condition using a time-dependent velocity ramp-up period.

        t (ti float): Current simulation time step.
        T (ti float): Time by which the inlet velocity will reach its maximum.
        """
        inlet_vel = self.inlet_val * (1 - ti.exp(-t / T))
        for i in ti.ndrange((1, self.height - 1)):
            rho = self.f2[1, i].sum() / (1 - inlet_vel)
            # Zou-He updates for missing right-moving distributions
            self.f2[1, i][RIGHT] = self.f2[1, i][LEFT] + (2 / 3) * rho * inlet_vel
            self.f2[1, i][UP_RIGHT] = self.f2[1, i][UP_LEFT] + (1 / 6) * rho * inlet_vel
            self.f2[1, i][DOWN_RIGHT] = self.f2[1, i][DOWN_LEFT] + (1 / 6) * rho * inlet_vel

    @ti.kernel
    def apply_outlet(self):
        """
        Applies the outlet zero-gradient boundary condition on the right and optionally bottom and
        top boundaries.
        """
        for i in ti.ndrange(self.height):
            self.f2[self.width - 1, i] = self.f2[self.width - 2, i]
        # Branch pruning implementaation.
        if ti.static(not self.tunnel):
            for i in ti.ndrange((1, self.width - 1)):
                self.f2[i, 0] = self.f2[i, 1]
                self.f2[i, self.height - 2] = self.f2[i, self.height - 2]

    @ti.kernel
    def object_collision(self):
        """
        Implements the bounce-back boundary condition for solid obstacles and computes drag/lift
        forces optionally using a branch pruning implementation.
        """
        if ti.static(self.ltd_ratio):
            self.drag[None] = 0.0
            self.lift[None] = 0.0
        for i, j in self.object_mask:
            for k in ti.static(range(9)):
                self.f2[i + self.dirs[0, 8 - k], j + self.dirs[1, 8 - k]][8 - k] = self.f2[i, j][k]
                if ti.static(self.ltd_ratio):
                    self.drag[None] += 2 * self.f2[i, j][k] * self.dirs[0, k]
                    self.lift[None] += 2 * self.f2[i, j][k] * self.dirs[1, k]

    @ti.func
    def apply_tunnel(self):
        """
        Implements the bounce-back boundary for top and bottom walls.
        """
        for i in ti.ndrange((1, self.width - 1)):
            self.f2[i + self.dirs[0, DOWN_RIGHT], self.height - 1 + self.dirs[1, DOWN_RIGHT]
                    ][DOWN_RIGHT] = self.f2[i, self.height - 1][UP_LEFT]
            self.f2[i + self.dirs[0, DOWN], self.height - 1 + self.dirs[1, DOWN]
                    ][DOWN] = self.f2[i, self.height - 1][UP]
            self.f2[i + self.dirs[0, DOWN_LEFT], self.height - 1 + self.dirs[1, DOWN_LEFT]
                    ][DOWN_LEFT] = self.f2[i, self.height - 1][UP_RIGHT]

            self.f2[i + self.dirs[0, UP_RIGHT], self.dirs[1, UP_RIGHT]
                    ][UP_RIGHT] = self.f2[i, 0][DOWN_LEFT]
            self.f2[i + self.dirs[0, UP], self.dirs[1, UP]][UP] = self.f2[i, 0][DOWN]
            self.f2[i + self.dirs[0, UP_LEFT], self.dirs[1, UP_LEFT]
                    ][UP_LEFT] = self.f2[i, 0][DOWN_RIGHT]

    @ti.kernel
    def boundary_condition(self, t: ti.types.f32, T: ti.types.i32):
        """
        Applies all relevant boundary conditions.

        t (ti float): Current simulation time step.
        T (ti float): Time by which the inlet velocity will reach its maximum.
        """
        self.apply_inlet(t, T)
        if ti.static(self.tunnel):
            self.apply_tunnel()

    @ti.kernel
    def max_vel(self):
        """
        Safely computes the maximum and minimum velocity magnitudes in the simulation grid.
        """
        curr_max = 1e-8
        curr_min = float('inf')
        for i, j in ti.ndrange((1, self.width - 1), (1, self.height - 1)):
            ti.atomic_max(curr_max, self.vel[i, j])
            ti.atomic_min(curr_min, self.vel[i, j])

        self.max_val[None] = self.max_val[None] * 0.95 + curr_max * 0.05
        self.min_val[None] = self.min_val[None] * 0.95 + curr_min * 0.05

    def display(self):
        """
        Runs the Lattice Boltzmann Method simulation and visualises it in a GUI.

        Returns:
            tuple: A tuple `(drag, lift)`, which contains lists tracking drag and lift forces over time.
        """
        gui = ti.GUI('LBM Simulation', (self.width, self.height))

        # Create folder for saving frames
        output_folder = "lbm_frames"
        os.makedirs(output_folder, exist_ok=True)

        frame_count = 0
        step = 0
        self.f2.copy_from(self.f1)
        self.boundary_condition(step, self.max_inlet_steps)
        self.stream()

        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT) and step < self.max_sim_step:
            self.max_vel()
            self.normalize_and_map()
            gui.set_image(self.rgb_image)

            filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            gui.show(filename if self.make_gif else None)
            frame_count += 1
            for _ in range(self.steps_per_frame):
                step += 1
                self.boundary_condition(step, self.max_inlet_steps)
                self.apply_outlet()
                self.collide_and_stream()
                self.object_collision()
                # Update grid f1 for the next iteration.
                self.f1.copy_from(self.f2)

    def simulate(self):
        self.display()
        if self.make_gif:
            print("Simulation ended. Generating GIF...")
            subprocess.run(["python3", "generate_gif.py"])
        if self.ltd_ratio:
            print("Simulation ended. Calculatin lift to drag ratio...")
            subprocess.run(["python3", "lift_to_drag.py"])


# Cuda is required (or the CPU) by the obstacle mask implementation.
ti.init(arch=ti.cuda)

# Karman Vortex Street
L = LBM(obstacle=CYLINDER, l_offset=400, tunnel=True, scale=40, theta=0,
        colormap="viridis")

# Airfoil
# L = LBM()

# Choose which to simulate or set your own parameters.
L.simulate()
