import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


ti.init(arch=ti.gpu)



def precompute_colormap():
    import matplotlib.cm as cm
    viridis = cm.get_cmap('flag', 256)
    # Extract RGB values
    colors = np.roll(viridis(np.linspace(0, 1, 256))[:, :3], 3)
    return colors.astype(np.float32)



n = 1000
wrap = True
tau = 10
coords = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]
coords = ti.Vector.field(n=2, dtype=ti.i32, shape=(9,))
for i in range(9):
    coords[i] = coords[i]
f_init = 1 / 9
disp = ti.field(dtype=ti.f32, shape=(n, n))
grid = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))
update_grid = ti.Vector.field(n=9, dtype=ti.f32, shape=(n, n))
tau_inv = 1 / tau
u = ti.Vector.field(n=2, dtype=ti.f32, shape=(n, n))
w = ti.field(dtype=ti.f32, shape=(9,))
wp = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 36
w.from_numpy(wp)
colormap = ti.Vector.field(3, dtype=ti.f32, shape=(256,))
colors = precompute_colormap()
colormap.from_numpy(colors)
rgb_image = ti.Vector.field(3, dtype=ti.u8, shape=(n, n))
max_val = ti.field(ti.f32, shape=())
max_val.fill(1e-8)


grid.fill(f_init)
# grid[0, n//2][5] += 10
for i in range(-10, 11):
    for j in range(20):
        if i**2 + j ** 2 < 200:
            grid[j, n//2 + i] += 2

for i in range(-10, 11):
    for j in range(20):
        if i**2 + j ** 2 < 200:
            grid[n//2 + j, n//2 + i + 150] += 2
# grid.fill(f_init)
# grid[1, n//2][5] += 1


@ti.kernel
def stream():
    # Static to allow for compile-time branch discarding.
    for i, j in ti.ndrange(
        (ti.static(0 if wrap else 1), ti.static(n if wrap else n - 1)),
        (ti.static(0 if wrap else 1), ti.static(n if wrap else n - 1))):
        for k in ti.ndrange(9):
            # Not sure if ternary operator allows for compile-time static branch discarding.
            ni = (i + coords[k].x + n) % n if ti.static(wrap) else (i + coords[k].x)
            nj = (j + coords[k].y + n) % n if ti.static(wrap) else (j + coords[k].y)
            update_grid[ni, nj][k] = grid[i, j][k]
        ''''''''
        # u[i, j] *= 1 / rho[i, j] * (rho[i, j] != 0)

@ti.kernel
def collide_and_stream():
    # Static to allow for compile-time branch discarding.
    for i, j in ti.ndrange(
        (ti.static(0 if wrap else 1), ti.static(n if wrap else n - 1)),
        (ti.static(0 if wrap else 1), ti.static(n if wrap else n - 1))):

        rho = 0.0
        up = ti.Vector([0.0, 0.0])
        for k in ti.ndrange(9):
            # Not sure if ternary operator allows for compile-time static branch discarding.
            rho += grid[i, j][k]
            up += coords[k] * grid[i, j][k]

        up = (up / rho) if rho > 0 else tm.vec2([0, 0])
        # u /= rho
        # tm.nan
        u[i, j] = up

        for k in ti.ndrange(9):
            feq = w[k] * rho * (1 + 3 * tm.dot(coords[k], up) + 4.5 * tm.dot(coords[k], up) ** 2 - 1.5 * tm.dot(up, up))
            grid[i, j][k] += tau_inv * (feq - grid[i, j][k])

        for k in ti.ndrange(9):
            # Not sure if ternary operator allows for compile-time static branch discarding.
            ni = (i + coords[k].x + n) % n if ti.static(wrap) else (i + coords[k].x)
            nj = (j + coords[k].y + n) % n if ti.static(wrap) else (j + coords[k].y)
            update_grid[ni, nj][k] = grid[i, j][k]
        ''''''''
        # u[i, j] *= 1 / rho[i, j] * (rho[i, j] != 0)

# @ti.kernel
# def bounce_boundary():
#     for i in ti.ndrange(n):
#

def update():
    grid.copy_from(update_grid)

@ti.kernel
def get_velocity_magnitude():
    for i, j in grid:
        disp[i, j] = u[i, j].norm()

def apply_colormap(data):
    norm_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    colormap = cm.viridis(norm_data)
    return (colormap[:, :, :3] * 255).astype(np.uint8)




@ti.kernel
def normalize_and_map():
    for i, j in disp:
        max_val[None] = ti.max(max_val[None], disp[i, j])
    # curr_max = 1e-8
    # for i, j in disp:
        # curr_max = ti.max(max_val[None], disp[i, j])
    for i, j in disp:
        norm_val = ti.cast(disp[i, j] / max_val[None] * 255, ti.i32)
        norm_val = ti.min(ti.max(norm_val, 0), 255)
        for c in ti.ndrange(3):
            rgb_image[i, j][c] = ti.u8(colormap[norm_val][c] * 255)


def display():
    gui = ti.GUI('LBM Simulation', (n, n))
    update_grid.copy_from(grid)
    stream()
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        update()

        # collide()

        get_velocity_magnitude()
        normalize_and_map()
        print(grid[0, n//2])
        gui.set_image(rgb_image)
        gui.show()
        # stream()
        collide_and_stream()




display()