import numpy as np
import matplotlib.pyplot as plt


def ESW_maker2(E, S, HQC):
    colors = np.array([[0.3, 0.1, 0.5], [0.9, 0, 0.52], [0.9, 0.8, 0.8]])

    E_vis = np.zeros((E.shape[0], E.shape[1], 3))
    S_vis = np.zeros((E.shape[0], E.shape[1], 3))

    E_vis[:, :, 0] = colors[0, 0] * E
    E_vis[:, :, 1] = colors[0, 1] * E
    E_vis[:, :, 2] = colors[0, 2] * E

    S_vis[:, :, 0] = colors[1, 0] * S
    S_vis[:, :, 1] = colors[1, 1] * S
    S_vis[:, :, 2] = colors[1, 2] * S

    y = E_vis + S_vis

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if HQC[i, j] == 0 or E[i, j] + S[i, j] == 0:
                y[i, j, 0] = colors[2, 0]
                y[i, j, 1] = colors[2, 1]
                y[i, j, 2] = colors[2, 2]

    return y
