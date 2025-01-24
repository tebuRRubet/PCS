import taichi as ti
import taichi.math as tm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ti.init(arch=ti.gpu)


# grid = np.arange(50 * 50 * 9).reshape(50, 50, 9)
grid = np.random.randn(150, 150, 9)
# grid = np.ones((5, 5, 9))

coords = [(-1, 1), (0, 1), (1, 1), (-1, 0), (0, 0), (1, 0), (-1, -1), (0, -1),
          (1, -1)]


def streaming(vals):
    for i, (x, y) in enumerate(coords):
        vals[:, :, i] = np.roll(vals[:, :, i], x, axis=1)
        vals[:, :, i] = np.roll(vals[:, :, i], -y, axis=0)
    return vals



def collision(vals, tau=5):
    # i = 0
    rho = np.sum(vals, axis=2)
    w = np.array([1, 4, 1, 4, 16, 4, 1, 4, 1]) / 9
    u = np.array([np.array([sum([vals[i, j][k] * np.array([x, y]) for k, (x, y) in enumerate(coords)]) for j in range(vals.shape[1])]) for i in range(vals.shape[0])]) / rho[:, :, None]
    feq = np.zeros_like(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            for k in range(9):
                val = w[k] * rho[i, j] * (1 + 3 * np.array(coords[k]).dot(u[i, j]) + 9 / 2 * np.array(coords[k]).dot(u[i, j]) ** 2 - 3 / 2 * u[i, j].dot(u[i, j]))
                feq[i, j][k] = val
    print(feq)
    return vals + (feq - vals) / tau




fig, ax = plt.subplots(figsize=(8, 6))

def update(iteration):
    global grid  # Use the global grid variable
    grid = collision(streaming(grid))  # Perform one iteration of streaming + collision

    # Compute density and velocity
    rho = np.sum(grid, axis=2)
    rho[rho == 0] = 1  # Prevent division by zero
    coords_array = np.array(coords)
    u = np.tensordot(grid, coords_array, axes=(2, 0)) / rho[:, :, None]

    # Clear and redraw the plot
    ax.clear()
    ax.imshow(rho, cmap='viridis', origin='lower')
    # ax.quiver(
    #     np.arange(grid.shape[1]),
    #     np.arange(grid.shape[0]),
    #     u[:, :, 0],
    #     u[:, :, 1],
    #     scale=1,
    #     scale_units='xy',
    #     color='white',
    # )
    ax.set_title(f"Iteration {iteration}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

# Create animation
ani = FuncAnimation(fig, update, frames=100, interval=100)  # 100 frames, 100ms delay
plt.show()