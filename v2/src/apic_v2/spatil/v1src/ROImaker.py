import numpy as np


def ROImaker(I, MMMM):
    y1 = (
        I * MMMM[:, :, np.newaxis]
    )  # Expand dimensions to match the third dimension of I
    colors = [0.9, 0.8, 0.8]

    y2 = np.zeros_like(I)

    y2[:, :, 0] = colors[0] * (1 - MMMM)
    y2[:, :, 1] = colors[1] * (1 - MMMM)
    y2[:, :, 2] = colors[2] * (1 - MMMM)

    y = y1 + y2

    return y
