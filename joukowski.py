import numpy as np
import matplotlib.pyplot as plt

nx, ny = 400, 100    # Domain size
cylinder_x = nx // 3    # Cylinder position x
cylinder_y = ny // 2    # Cylinder position y
cylinder_r = ny // 8    # Cylinder radius
a, b = 0.041, 0.272


def joukowski_transformc(z):
    return z + 1 / z


def gltc(z, center, radius):
    a = 0.041
    b = 0.272
    c = complex(center[0], center[1])
    d = complex(a, b)
    return (z - c) / radius + d


def igltc(z):
    a = 0.041
    b = 0.272
    c = complex(400 // 3, 100 // 2)
    d = complex(a, b)
    return (100 // 8 * (z - d)) + c


def glt(x, y):
    return ((x - cylinder_x) / cylinder_r) + a, ((y - cylinder_y) / cylinder_r) + b


def iglt(x, y):
    return cylinder_r * (x - a) + cylinder_x, cylinder_r * (y - b) + cylinder_y


def joukowski_transform(x, y):
    r = x**2 + y**2
    return x * (1 + 1 / r), y * (1 - 1 / r)


def airfoil_transform(c):
    x, y = c[0], c[1]
    t1 = glt(x, y)
    t2 = joukowski_transform(t1[0], t1[1])
    t3 = iglt(t2[0], t2[1])
    return t3[0], t3[1]


def plot_joukowski(center, radius, ax1, ax2):
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = (center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta))
    center2 = (0.041, 0.272)
    circle2 = center2[0] + np.cos(theta) + 1j * (center2[1] + np.sin(theta))

    # Transform the circle
    # circle3 = glt(circle, center, radius)
    # transformed = joukowski_transform(circle)
    # transformed = iglt(transformed)
    # transformed = iglt(circle)
    # transformed = iglt(joukowski_transform(glt(circle, center, radius)))
    transformed = airfoil_transform(circle)
    # transformed1 = glt(circle, center, radius)
    # transformed2 = joukowski_transform(transformed1)
    # transformed3 = iglt(transformed2)

    # Plot the original circle
    ax1.clear()
    ax1.plot(circle[0], circle[1], 'b-', label="Original Circle")
    ax1.scatter([1, -1], [0, 0], color='r', label="Critical Points")
    ax1.set_title("Original Circle")
    ax1.axis("equal")
    ax1.legend()

    # Plot the transformed airfoil
    ax2.clear()
    ax2.plot(transformed[0], transformed[1], 'r-', label="Transformed Airfoil")
    ax2.set_title("Joukowski Transformed Airfoil")
    ax2.axis("equal")
    ax2.legend()


# Initial parameters
initial_center = (0.041, 0.272)
initial_radius = 1.0
initial_center = (400 // 3, 100 // 2)
initial_radius = 100 // 8
# initial_center = (0, 0)

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot initial setup
plot_joukowski(initial_center, initial_radius, ax1, ax2)

# Add sliders for interactive control
# ax_slider_x = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# ax_slider_y = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# slider_x = Slider(ax_slider_x, 'Center X', -2.0, 2.0, valinit=initial_center[0])
# slider_y = Slider(ax_slider_y, 'Center Y', -2.0, 2.0, valinit=initial_center[1])

plt.show()
