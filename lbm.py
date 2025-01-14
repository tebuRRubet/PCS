import os
import numpy as np
import matplotlib.pyplot as plt


output_dir = "frames2"
os.makedirs(output_dir, exist_ok=True)


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    Nx = 400
    Ny = 100
    tau = 0.53
    Nt = 5000
    plot_every = 100
    save = False

    # Lattice speeds and weights.
    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36,
                        1 / 9, 1 / 36])

    # Initial conditions.
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3

    cylinder = np.full((Ny, Nx), False)

    for y in range(Ny):
        for x in range(Nx):
            if distance(Nx // 4, Ny // 2, x, y) < 13:
                r = x**2 + y**2
                a = int(x * (1 + 1 / r))
                b = int(y * (1 - 1 / r))
                cylinder[b][a] = True

    # Main loop.
    for it in range(Nt):

        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Fluid variables.
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        F[cylinder, :] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            uu = ux**2 + uy**2
            vu = cx * ux + cy * uy
            Feq[:, :, i] = rho * w * (
                1 + 3 * vu + (9 / 2) * vu**2 - (3 / 2) * uu
            )

        F -= (F - Feq) / tau

        if it % plot_every == 0:
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy

            plt.imshow(curl, cmap="bwr")
            # plt.imshow(np.sqrt(uu))

            if save:
                frame_path = os.path.join(output_dir, f"frame_{it:04d}.png")
                plt.savefig(frame_path)

            plt.pause(0.01)
            plt.cla()


if __name__ == "__main__":
    main()
