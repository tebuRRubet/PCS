import taichi as ti
import taichi.math as tm


ti.init(arch=ti.cpu)


m, n = 1000, 1000
f = ti.Vector.field(n=9, dtype=ti.f32, shape=(m, n))
f_update = ti.Vector.field(n=9, dtype=ti.f32, shape=(m, n))
disp = ti.field(dtype=ti.f32, shape=(m, n))



@ti.kernel
def get_density_disp(f: ti.Vector.field(n=9, dtype=ti.f32, shape=(m, n))):
    for i, j in f:
        disp[i, j] = f[i, j].sum()
    return disp





gui = ti.GUI("LBM no-OOP", (m, n))
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    gui.set_image(get_density_disp(f))
    gui.show()