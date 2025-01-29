import taichi as ti

CYLINDER, EGG, AIRFOIL = 0, 1, 2


@ti.func
def rotate(x, y, theta):
    theta = ti.cast(theta * ti.math.pi / 180.0, ti.f32)
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    return cos_theta * x - sin_theta * y, sin_theta * x + cos_theta * y


@ti.func
def distance(x1, y1, x2, y2):
    return ti.sqrt((x1 - x2)**2 + (y1 - y2)**2)


@ti.func
def glt(x, y, horizontal_shift, vertical_shift, scale, new_x, new_y, new_r):
    return ((x - horizontal_shift) / scale) * new_r + new_x, \
        ((y - vertical_shift) / scale) * new_r + new_y


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
def is_in_cylinder(x, y, scale):
    return ti.cast(distance(x, y, 0, 0) <= scale, ti.i32)


@ti.func
def is_in_egg(x, y, scale):
    x_shifted, y_shifted = x, y
    r_squared = x_shifted**2 + y_shifted**2
    discriminant = ti.sqrt(ti.abs(r_squared - 4.0))
    zeta_x = (x_shifted - discriminant) * 0.5
    zeta_y = (y_shifted - discriminant) * 0.5
    return ti.cast(zeta_x**2 + zeta_y**2 <= scale**2, ti.i32)


@ti.func
def is_in_airfoil(alpha, beta, horizontal_shift, vertical_shift, scale, new_x, new_y, new_r):
    alpha2, beta2 = glt(alpha, beta, horizontal_shift, vertical_shift, scale, new_x, new_y, new_r)
    x1, y1, x2, y2 = inverse_joukowski_transform(alpha2, beta2)
    check1 = distance(x1, y1, new_x, new_y) <= new_r
    check2 = distance(x2, y2, new_x, new_y) <= new_r
    return ti.cast(not (check1 or check2), ti.i32)


@ti.func
def is_in_obstacle(x, y, obstacle, horizontal_shift, vertical_shift, scale, new_x, new_y, new_r):
    result = 0
    if obstacle == CYLINDER:
        result = is_in_cylinder(x, y, scale)
    elif obstacle == EGG:
        result = is_in_egg(x, y, scale)
    elif obstacle == AIRFOIL:
        result = is_in_airfoil(x, y, horizontal_shift, vertical_shift, scale, new_x, new_y, new_r)
    return result
