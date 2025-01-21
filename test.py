import matplotlib.pyplot as plt
import numpy as np

nx, ny = 400, 100    # Domain size
cylinder_x = nx // 3    # Cylinder position x
cylinder_y = ny // 2    # Cylinder position y
cylinder_r = ny // 8    # Cylinder radius
a, b = 0.041, 0.272


def test(a, b):
    u = a**2 - b**2 - 4
    v = 2 * a * b
    r = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)

    x = (a + np.sqrt(r) * np.cos(theta / 2)) / 2
    y = (b + np.sqrt(r) * np.sin(theta / 2)) / 2
    return x, y


def test2(a, b):
    u = a**2 - b**2 - 4
    v = 2 * a * b
    r = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)

    x = (a - np.sqrt(r) * np.cos(theta / 2)) / 2
    y = (b - np.sqrt(r) * np.sin(theta / 2)) / 2
    return x, y


def dcircle(x, y, x1, y1):
    return np.sqrt((x - x1)**2 + (y - y1)**2)


inside_x = []
inside_y = []
outside_x = []
outside_y = []

puntjes = 40000
r = 1

c = (0.041, 0.272)
for i in range(puntjes):
    x = (np.random.rand() - 0.5) * 5
    y = (np.random.rand() - 0.5) * 5

    x1, y1 = test(x, y)
    x2, y2 = test2(x, y)
    if dcircle(c[0], c[1], x1, y1) < r or dcircle(c[0], c[1], x2, y2) < r:
        inside_x.append(x)
        inside_y.append(y)
    else:
        outside_x.append(x)
        outside_y.append(y)

plt.scatter(inside_x, inside_y, c="b")
plt.scatter(outside_x, outside_y, c="r")
# print(outside_x)
# y_test = 0.5
# x_test = np.linspace(0, 1, 100)
# plt.plot(x_test, test(x_test, y_test))
plt.show()
