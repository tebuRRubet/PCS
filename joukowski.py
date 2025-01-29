import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def joukowski_transform(z):
    return z + 1 / z


def plot_joukowski(center, radius, ax1, ax2):
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = center[0] + radius * np.cos(theta) + 1j * (center[1] + radius * np.sin(theta))

    # Transform the circle
    transformed = joukowski_transform(circle)

    # Plot the original circle
    ax1.clear()
    ax1.plot(circle.real, circle.imag, 'b-', label="Original Circle")
    ax1.set_title("Original Circle")
    ax1.axis("equal")
    ax1.legend()

    # Plot the transformed airfoil
    ax2.clear()
    ax2.plot(transformed.real, transformed.imag, 'r-', label="Transformed Airfoil")
    ax2.set_title("Joukowski Transformed Airfoil")
    ax2.axis("equal")
    ax2.legend()


# Initial parameters
initial_center = (0.026, 0.077)
initial_radius = 0.918

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

# Plot initial setup
plot_joukowski(initial_center, initial_radius, ax1, ax2)

# Add sliders for interactive control
ax_slider_x = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_y = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_r = plt.axes([0.2, 0.09, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_x = Slider(ax_slider_x, 'Center X', -2.0, 2.0, valinit=initial_center[0])
slider_y = Slider(ax_slider_y, 'Center Y', -2.0, 2.0, valinit=initial_center[1])
slider_r = Slider(ax_slider_r, 'r', -2.0, 2.0, valinit=initial_radius)


def update(val):
    center = (slider_x.val, slider_y.val)
    plot_joukowski(center, slider_r.val, ax1, ax2)
    fig.canvas.draw_idle()


slider_x.on_changed(update)
slider_y.on_changed(update)
slider_r.on_changed(update)


plt.show()
