import taichi as ti
import taichi.math as tm
import numpy as np
import time

ti.init(arch=ti.cpu)


m, n = 40, 40
tau = 15
tau_inv = 1 / tau

f = ti.Vector.field(n=9, dtype=ti.f32, shape=(m, n))
f_update = ti.Vector.field(n=9, dtype=ti.f32, shape=(m, n))





dirs = ti.Vector.field(n=2, dtype=ti.i8, shape=(9,))

for i in range(1):
    for j in range(2):
        f[i + 10, j + 20] += 1 / 9

# for i in range(-5, 60):
#     for j in range(-5, 60):
#         f[i + 500, j + 400] += 1 / 9

for i, d in enumerate([(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]):
    dirs[i] = d
w = ti.field(dtype=ti.f32, shape=(9,))
w.from_numpy(np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 36)
u_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(m, n))

disp = ti.field(dtype=ti.f32, shape=(m, n))











@ti.kernel
def stream():
    for i, j in ti.ndrange((1, m - 1), (1, n - 1)):
        for k in ti.static(range(9)):
            ni, nj = i + dirs[k].x, j + dirs[k].y
            f_update[ni, nj][k] = f[i, j][k]


@ti.kernel
def collide():
    for i, j in ti.ndrange((1, m - 1), (1, n - 1)):
        rho = f_update[i, j].sum()
        u = tm.vec2([0.0, 0.0])
        for k in ti.static(range(9)):
            u += f_update[i, j][k] * dirs[k]
        if rho != 0:
            u /= rho
        else:
            u = tm.vec2([0.0, 0.0])
        u_field[i, j] = u
        for k in ti.static(range(9)):
            feq = w[k] * rho * (1 + 3 * tm.dot(u, dirs[k]) + 4.5 * tm.dot(u, dirs[k]) ** 2 - 1.5 * tm.dot(u, u))
            f_update[i, j][k] += tau_inv * (feq - f_update[i, j][k])


@ti.kernel
def update_grid():
    for i, j in f_update:
        f[i, j] = f_update[i, j]


@ti.kernel
def get_density_disp():
    for i, j in f:
        # disp[i, j] = u_field[i, j].norm()
        disp[i, j] = f[i, j].sum()


if m == 10:
    scale = 64
else:
    scale = 1
scale = 20
gui = ti.GUI("LBM no-OOP", (m * scale, n * scale))

while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    get_density_disp()
    
    up_disp = disp.to_numpy().repeat(scale, axis=0).repeat(scale, axis=1)
    # up_disp = disp
    gui.set_image(up_disp)
    gui.show()    

    stream()
    collide()
    update_grid()
    time.sleep(0.02)
