import taichi as ti
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ti.init(arch=ti.gpu)

# Simulation parameters
nx, ny = 400, 100  # Domain size
tau = 0.55         # Relaxation time
omega = 1.0 / tau  # Relaxation frequency
u_max = 0.1        # Maximum velocity
rho0 = 1.0         # Reference density
max_steps = 2000   # Reduced for reasonable gif size
ramp_steps = 100   # Number of steps to ramp up the velocity
show_every = 10

# Possible obstacles
CYLINDER, EGG, AIRFOIL = 0, 1, 2
obstacle = EGG

# Fields and constants
rho = ti.field(dtype=ti.f32, shape=(nx, ny))
u_x = ti.field(dtype=ti.f32, shape=(nx, ny))
u_y = ti.field(dtype=ti.f32, shape=(nx, ny))
f = ti.field(dtype=ti.f32, shape=(nx, ny, 9))
f_next = ti.field(dtype=ti.f32, shape=(nx, ny, 9))

# Lattice properties
c_x = ti.field(dtype=ti.f32, shape=9)
c_y = ti.field(dtype=ti.f32, shape=9)
w = ti.field(dtype=ti.f32, shape=9)
opposite = ti.field(dtype=ti.i32, shape=9)

# Obstacle parameters
cylinder_x, cylinder_y = nx // 3, ny // 2
cylinder_r = ny // 8
a, b = 0.041, 0.272
mask = ti.field(dtype=ti.i32, shape=(nx, ny))
MaskType = ti.types.ndarray(dtype=ti.i32, ndim=2)


@ti.kernel
def init_d2q9_constants():
    # Initialize D2Q9 lattice velocities and weights
    c_x[0], c_y[0], w[0], opposite[0] = 0, 0, 4.0 / 9.0, 0
    c_x[1], c_y[1], w[1], opposite[1] = 1, 0, 1.0 / 9.0, 3
    c_x[2], c_y[2], w[2], opposite[2] = 0, 1, 1.0 / 9.0, 4
    c_x[3], c_y[3], w[3], opposite[3] = -1, 0, 1.0 / 9.0, 1
    c_x[4], c_y[4], w[4], opposite[4] = 0, -1, 1.0 / 9.0, 2
    c_x[5], c_y[5], w[5], opposite[5] = 1, 1, 1.0 / 36.0, 7
    c_x[6], c_y[6], w[6], opposite[6] = -1, 1, 1.0 / 36.0, 8
    c_x[7], c_y[7], w[7], opposite[7] = -1, -1, 1.0 / 36.0, 5
    c_x[8], c_y[8], w[8], opposite[8] = 1, -1, 1.0 / 36.0, 6


@ti.func
def equilibrium(rho_local, ux_local, uy_local, k):
    cu = c_x[k] * ux_local + c_y[k] * uy_local
    usqr = ux_local**2 + uy_local**2
    return w[k] * rho_local * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr)


@ti.func
def distance(x1, y1, x2, y2):
    return ti.sqrt((x1 - x2)**2 + (y1 - y2)**2)


@ti.func
def glt(x, y):
    return ((x - cylinder_x) / cylinder_r) + a, \
           ((y - cylinder_y) / cylinder_r) + b


@ti.func
def inverse_glt(x, y):
    return cylinder_r * (x - a) + cylinder_x, cylinder_r * (y - b) + cylinder_y


@ti.func
def joukowski_transform(x, y):
    r = x**2 + y**2
    return x * (1 + 1 / r), y * (1 - 1 / r)


@ti.func
def inverse_joukowski_transform(alpha, beta):
    u = alpha**2 - beta**2 - 4
    v = 2 * alpha * beta
    r = ti.sqrt(u**2 + v**2)
    theta = ti.atan2(v, u)

    x1 = (alpha + ti.sqrt(r) * ti.cos(theta / 2)) / 2
    y1 = (beta + ti.sqrt(r) * ti.sin(theta / 2)) / 2
    x2 = (alpha - ti.sqrt(r) * ti.cos(theta / 2)) / 2
    y2 = (beta - ti.sqrt(r) * ti.sin(theta / 2)) / 2
    return x1, y1, x2, y2


@ti.func
def is_in_cylinder(x, y):
    dx, dy = x - cylinder_x, y - cylinder_y
    return ti.cast(distance(dx, dy, 0, 0) <= cylinder_r**2, ti.i32)


@ti.func
def is_in_egg(x, y):
    x_shifted, y_shifted = x - cylinder_x, y - cylinder_y
    r_squared = x_shifted**2 + y_shifted**2
    discriminant = ti.sqrt(ti.abs(r_squared - 4.0))
    zeta_x = (x_shifted - discriminant) * 0.5
    zeta_y = (y_shifted - discriminant) * 0.5
    return ti.cast(zeta_x**2 + zeta_y**2 <= cylinder_r**2, ti.i32)


@ti.func
def is_in_airfoil(alpha, beta):
    alpha2, beta2 = glt(alpha, beta)
    x1, y1, x2, y2 = inverse_joukowski_transform(alpha2, beta2)
    check1 = distance(x1, y1, a, b) <= 1
    check2 = distance(x2, y2, a, b) <= 1
    return ti.cast(not (check1 or check2), ti.i32)


@ti.func
def is_in_obstacle(x, y):
    result = 0
    if obstacle == CYLINDER:
        result = is_in_cylinder(x, y)
    elif obstacle == EGG:
        result = is_in_egg(x, y)
    elif obstacle == AIRFOIL:
        result = is_in_airfoil(x, y)
    return result


@ti.kernel
def initialize():
    for i, j in rho:
        # Initialize density to reference density everywhere
        rho[i, j] = rho0
        # Initialize all velocities to zero
        u_x[i, j], u_y[i, j] = 0.0, 0.0

        # Initialize distribution functions to equilibrium
        for k in range(9):
            f[i, j, k] = equilibrium(rho0, 0.0, 0.0, k)
            f_next[i, j, k] = f[i, j, k]

    for i, j in mask:
        # Assign 1 if inside, 0 if outside
        mask[i, j] = is_in_obstacle(i, j)


@ti.kernel
def apply_inlet_conditions(current_u_max: float):
    # Apply inlet conditions with current ramped velocity
    for j in range(ny):
        rho[0, j] = rho0
        u_x[0, j], u_y[0, j] = current_u_max, 0.0
        for k in range(9):
            f[0, j, k] = equilibrium(rho0, current_u_max, 0.0, k)


@ti.kernel
def collide(mask: MaskType):
    for i, j in rho:
        if not mask[i, j]:
            # Calculate macroscopic quantities
            r, u, v = 0.0, 0.0, 0.0
            for k in range(9):
                r += f[i, j, k]
                u += c_x[k] * f[i, j, k]
                v += c_y[k] * f[i, j, k]

            if r > 1e-10:
                u, v = u / r, v / r

            # Store macroscopic quantities
            rho[i, j], u_x[i, j], u_y[i, j] = r, u, v

            # Collision
            for k in range(9):
                f_eq = equilibrium(r, u, v, k)
                f_next[i, j, k] = f[i, j, k] - omega * (f[i, j, k] - f_eq)


@ti.kernel
def stream(mask: MaskType):
    # Streaming step
    for i, j, k in f:
        if not mask[i, j]:
            # Calculate destination
            ni, nj = i + int(c_x[k]), j + int(c_y[k])

            # Handle boundaries
            if 0 <= ni < nx and 0 <= nj < ny:
                if not mask[ni, nj]:
                    f[ni, nj, k] = f_next[i, j, k]
                else:
                    # Bounce-back on obstacle
                    f[i, j, opposite[k]] = f_next[i, j, k]


@ti.kernel
def apply_outlet_conditions():
    # Outlet (right boundary) - zero gradient outflow condition
    for j in range(ny):
        # Set zero gradient for density and velocity
        rho[nx - 1, j] = rho[nx - 2, j]
        u_x[nx - 1, j] = u_x[nx - 2, j]
        u_y[nx - 1, j] = u_y[nx - 2, j]

        # Calculate equilibrium distribution at outlet
        for k in range(9):
            f[nx - 1, j, k] = equilibrium(rho[nx - 1, j], u_x[nx - 1, j],
                                          u_y[nx - 1, j], k)


def create_frame(step, mask):
    # Create figure
    fig = Figure(figsize=(20, 5), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Get velocity magnitude
    ux = u_x.to_numpy()
    uy = u_y.to_numpy()
    vel_mag = np.sqrt(ux**2 + uy**2)

    # Plot velocity magnitude
    im = ax.imshow(vel_mag.T, extent=[0, nx, 0, ny],
                   cmap='viridis', aspect='equal', origin='lower',
                   vmin=0, vmax=u_max * 1.5)
    fig.colorbar(im, label='Velocity magnitude')

    # Add velocity vectors
    skip = 10
    x, y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip))
    ax.quiver(x, y, ux[::skip, ::skip].T, uy[::skip, ::skip].T,
              scale=5, color='white', alpha=0.3)

    # Overlay cylinder on the plot
    for i in range(nx):
        for j in range(ny):
            if mask[i, j] == 1:
                ax.plot(i, j, 'wo', markersize=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Flow field at step {step}')
    fig.tight_layout()

    # Convert to image
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape((height, width, 4))
    # Convert RGBA to RGB
    image = image[:, :, :3]

    plt.close(fig)
    return image


def simulate():
    # Initialize
    init_d2q9_constants()
    initialize()

    # List to store frames
    frames = []

    # Main loop
    for step in range(max_steps + show_every):
        # Calculate current inlet velocity during ramp-up
        if step < ramp_steps:
            current_u_max = u_max * (step / ramp_steps)
        else:
            current_u_max = u_max

        apply_inlet_conditions(current_u_max)
        collide(mask.to_numpy())
        stream(mask.to_numpy())
        apply_outlet_conditions()

        # Save frame every 10 steps
        if step % show_every == 0:
            print(f"Step {step}/{max_steps}")
            frames.append(create_frame(step, mask.to_numpy()))

    # Save as GIF
    print("Saving animation...")
    imageio.mimsave('flow_simulation.gif', frames, fps=15)
    print("Animation saved as 'flow_simulation.gif'")


if __name__ == "__main__":
    simulate()
