import taichi as ti
import imageio
import numpy as np
ti.init(arch=ti.gpu)

# Simulation parameters
nx, ny = 400, 100    # Domain size
tau = 0.55           # Relaxation time
omega = 1.0 / tau    # Relaxation frequency
u_max = 0.1          # Maximum velocity
rho0 = 1.0           # Reference density

# Obstacle
obstacle = "cylinder"
obstacle = "egg"
obstacle = "airfoil"

# Initialize grid fields
rho = ti.field(dtype=float, shape=(nx, ny))
u_x = ti.field(dtype=float, shape=(nx, ny))
u_y = ti.field(dtype=float, shape=(nx, ny))
f = ti.field(dtype=float, shape=(nx, ny, 9))
f_next = ti.field(dtype=float, shape=(nx, ny, 9))

# D2Q9 lattice constants
c_x = ti.field(dtype=float, shape=9)
c_y = ti.field(dtype=float, shape=9)
w = ti.field(dtype=float, shape=9)
opposite = ti.field(dtype=int, shape=9)

# Cylinder obstacle
cylinder_x = nx // 3    # Cylinder position x
cylinder_y = ny // 2    # Cylinder position y
cylinder_r = ny // 8    # Cylinder radius

a, b = 0.041, 0.272


@ti.kernel
def init_d2q9_constants():
    # D2Q9 velocities
    for i in ti.static(range(9)):
        # Velocity set
        if i == 0:      # Rest
            c_x[i], c_y[i] = 0, 0
            opposite[i] = 0
        elif i == 1:    # Right
            c_x[i], c_y[i] = 1, 0
            opposite[i] = 3
        elif i == 2:    # Top
            c_x[i], c_y[i] = 0, 1
            opposite[i] = 4
        elif i == 3:    # Left
            c_x[i], c_y[i] = -1, 0
            opposite[i] = 1
        elif i == 4:    # Bottom
            c_x[i], c_y[i] = 0, -1
            opposite[i] = 2
        elif i == 5:    # Top-right
            c_x[i], c_y[i] = 1, 1
            opposite[i] = 7
        elif i == 6:    # Top-left
            c_x[i], c_y[i] = -1, 1
            opposite[i] = 8
        elif i == 7:    # Bottom-left
            c_x[i], c_y[i] = -1, -1
            opposite[i] = 5
        else:          # Bottom-right
            c_x[i], c_y[i] = 1, -1
            opposite[i] = 6

    # D2Q9 weights
    w[0] = 4.0 / 9.0   # Rest
    for i in ti.static(range(1, 5)):  # Cardinals
        w[i] = 1.0 / 9.0
    for i in ti.static(range(5, 9)):  # Diagonals
        w[i] = 1.0 / 36.0


@ti.func
def equilibrium(rho_local, ux_local, uy_local, k):
    cu = c_x[k] * ux_local + c_y[k] * uy_local
    usqr = ux_local * ux_local + uy_local * uy_local
    return w[k] * rho_local * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)


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
    dx = x - cylinder_x
    dy = y - cylinder_y
    return ti.cast(dx * dx + dy * dy <= cylinder_r * cylinder_r, ti.i32)


@ti.func
def is_in_egg(x, y):
    x_shifted = float(x) - cylinder_x
    y_shifted = float(y) - cylinder_y

    r_squared = x_shifted * x_shifted + y_shifted * y_shifted
    z_squared = r_squared

    discriminant = ti.sqrt(ti.abs(z_squared - 4.0))

    zeta_x = (x_shifted - discriminant) * 0.5
    zeta_y = (y_shifted - discriminant) * 0.5

    zeta_r_squared = zeta_x * zeta_x + zeta_y * zeta_y

    return ti.cast(zeta_r_squared <= cylinder_r * cylinder_r, ti.i32)


@ti.func
def is_in_airfoil(alpha, beta):
    alpha2, beta2 = glt(alpha, beta)
    x1, y1, x2, y2 = inverse_joukowski_transform(alpha2, beta2)
    check1 = distance(a, b, x1, y1) <= 1
    check2 = distance(a, b, x2, y2) <= 1
    return ti.cast(not (check1 or check2), ti.i32)


@ti.func
def is_in_obstacle(x, y):
    result = 0
    if obstacle == "cylinder":
        result = is_in_cylinder(x, y)
    elif obstacle == "egg":
        result = is_in_egg(x, y)
    elif obstacle == "airfoil":
        result = is_in_airfoil(x, y)
    return result


@ti.kernel
def initialize():
    for i, j in rho:
        # Initialize density to reference density everywhere
        rho[i, j] = rho0

        # Initialize all velocities to zero
        u_x[i, j] = 0.0
        u_y[i, j] = 0.0

        # Set cylinder boundary conditions
        # if is_in_obstacle(i, j):
        #     u_x[i, j] = 0.0
        #     u_y[i, j] = 0.0

        # Initialize distribution functions to equilibrium
        for k in range(9):
            f[i, j, k] = equilibrium(rho[i, j], u_x[i, j], u_y[i, j], k)
            f_next[i, j, k] = f[i, j, k]


@ti.kernel
def apply_inlet_conditions(current_u_max: float):
    # Apply inlet conditions with current ramped velocity
    for j in range(ny):
        rho[0, j] = rho0
        u_x[0, j] = current_u_max
        u_y[0, j] = 0.0
        for k in range(9):
            f[0, j, k] = equilibrium(rho0, current_u_max, 0.0, k)


@ti.kernel
def collide():
    for i, j in rho:
        if not is_in_obstacle(i, j):
            # Calculate macroscopic quantities
            r = 0.0
            u = 0.0
            v = 0.0
            for k in range(9):
                r += f[i, j, k]
                u += c_x[k] * f[i, j, k]
                v += c_y[k] * f[i, j, k]

            if r > 1e-10:
                u /= r
                v /= r

            # Store macroscopic quantities
            rho[i, j] = r
            u_x[i, j] = u
            u_y[i, j] = v

            # Collision
            for k in range(9):
                f_eq = equilibrium(r, u, v, k)
                f_next[i, j, k] = f[i, j, k] - omega * (f[i, j, k] - f_eq)


@ti.kernel
def stream():
    # Streaming step
    for i, j, k in f:
        if not is_in_obstacle(i, j):
            # Calculate destination
            ni = i + int(c_x[k])
            nj = j + int(c_y[k])

            # Handle boundaries
            if 0 <= ni < nx and 0 <= nj < ny:
                if not is_in_obstacle(ni, nj):
                    f[ni, nj, k] = f_next[i, j, k]
                else:
                    # Bounce-back on cylinder
                    f[i, j, opposite[k]] = f_next[i, j, k]
            else:
                # Bounce-back on top/bottom walls
                if nj < 0 or nj >= ny:
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


@ti.kernel
def apply_wall_conditions():
    # Top and bottom walls
    for i in range(nx):
        # Bottom wall (j = 0)
        u_x[i, 0] = 0.0
        u_y[i, 0] = 0.0
        # Top wall (j = ny-1)
        u_x[i, ny - 1] = 0.0
        u_y[i, ny - 1] = 0.0


@ti.kernel
def generate_mask():
    for i, j in ti.ndrange(nx, ny):
        # Assign 1 if inside, 0 if outside
        mask[i, j] = is_in_obstacle(i, j)
        # x, y = airfoil_transform(i, j)
        # mask[x, y] = is_in_cylinder(i, j)


# Use integer type for cylinder mask
mask = ti.field(dtype=ti.i32, shape=(nx, ny))


def create_frame(step):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

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

    # Add velocity vectors (decimated for clarity)
    skip = 10
    x, y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip))
    ax.quiver(x, y, ux[::skip, ::skip].T, uy[::skip, ::skip].T,
              scale=5, color='white', alpha=0.3)

    # Generate cylinder mask and visualize it
    generate_mask()
    mask_np = mask.to_numpy()

    # Overlay cylinder on the plot
    for i in range(nx):
        for j in range(ny):
            if mask_np[i, j] == 1:
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

    # Simulation parameters
    max_steps = 2000  # Reduced for reasonable gif size
    ramp_steps = 100  # Number of steps to ramp up the velocity

    # List to store frames
    frames = []

    # Main loop
    for step in range(max_steps):
        # Calculate current inlet velocity during ramp-up
        if step < ramp_steps:
            current_u_max = u_max * (step / ramp_steps)
        else:
            current_u_max = u_max

        # Apply ramped inlet conditions
        apply_inlet_conditions(current_u_max)

        # Main simulation steps
        collide()
        stream()
        apply_outlet_conditions()
        apply_wall_conditions()

        # Save frame every 10 steps
        if step % 10 == 0:
            print(f"Step {step}/{max_steps}")
            frames.append(create_frame(step))

    # Save as GIF
    print("Saving animation...")
    imageio.mimsave('flow_simulation.gif', frames, fps=15)
    print("Animation saved as 'flow_simulation.gif'")


if __name__ == "__main__":
    simulate()
