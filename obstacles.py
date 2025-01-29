import taichi as ti

CYLINDER, EGG, AIRFOIL = 0, 1, 2


@ti.func
def translate_scale_rotate(i, j, di, dj, scale, theta):
    x, y = (i - di) / scale, (j - dj) / scale

    theta = ti.cast(theta * ti.math.pi / 180.0, ti.f32)
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    return cos_theta * x - sin_theta * y, sin_theta * x + cos_theta * y


@ti.func
def distance(x1, y1, x2, y2):
    return ti.sqrt((x1 - x2)**2 + (y1 - y2)**2)


@ti.func
def glt(x, y, cylinder_r, a, b):
    return (x / cylinder_r) + a, \
        (y / cylinder_r) + b


@ti.func
def inverse_glt(x, y, cylinder_r, a, b):
    return cylinder_r * (x - a), cylinder_r * (y - b)


@ti.func
def joukowski_transform(self, x, y):
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
def is_in_cylinder(x, y, cylinder_r):
    return ti.cast(distance(x, y, 0, 0) <= cylinder_r, ti.i32)


@ti.func
def is_in_egg(x, y, cylinder_r):
    x_shifted, y_shifted = x, y
    r_squared = x_shifted**2 + y_shifted**2
    discriminant = ti.sqrt(ti.abs(r_squared - 4.0))
    zeta_x = (x_shifted - discriminant) * 0.5
    zeta_y = (y_shifted - discriminant) * 0.5
    return ti.cast(zeta_x**2 + zeta_y**2 <= cylinder_r**2, ti.i32)


@ti.func
def is_in_airfoil(alpha, beta, cylinder_r, a, b):
    alpha2, beta2 = glt(alpha, beta, cylinder_r, a, b)
    x1, y1, x2, y2 = inverse_joukowski_transform(alpha2, beta2)
    check1 = distance(x1, y1, a, b) <= 1
    check2 = distance(x2, y2, a, b) <= 1
    return ti.cast(not (check1 or check2), ti.i32)


@ti.func
def is_in_obstacle(x, y, obstacle, cylinder_r, a, b):
    result = 0
    if obstacle == CYLINDER:
        result = is_in_cylinder(x, y, cylinder_r)
    elif obstacle == EGG:
        result = is_in_egg(x, y, cylinder_r)
    elif obstacle == AIRFOIL:
        result = is_in_airfoil(x, y, cylinder_r, a, b)
    return result
