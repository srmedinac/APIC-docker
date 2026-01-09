import numpy as np
import matplotlib.pyplot as plt


def putbnd(x, y=None, buffer=None, nocall=None):
    if y is None:
        y = []
    if buffer is None:
        buffer = []
    if nocall is None:
        nocall = []

    if not y or np.isscalar(y):
        nocall = buffer
        buffer = y

        if x.shape[1] >= 2:
            y = x[:, 1]
            x = x[:, 0]
        else:
            raise ValueError("PUTBND: invalid point coordinates")

    x = x.flatten()
    y = y.flatten()

    if len(x) != len(y):
        raise ValueError("PUTBND: lengths of coordinate vectors are incompatible.")

    if buffer is None:
        buffer = 0.05

    if nocall is None:
        nocall = 0

    # Remove NaN's
    indx = np.isfinite(x) & np.isfinite(y)
    x = x[indx]
    y = y[indx]

    v = np.zeros(4)
    v[0] = min(x) - buffer * np.ptp(x)
    v[1] = max(x) + buffer * np.ptp(x)
    v[2] = min(y) - buffer * np.ptp(y)
    v[3] = max(y) + buffer * np.ptp(y)

    if v[1] - v[0] < np.finfo(float).eps:  # No variation in x
        v[0] = x[0] - 1
        v[1] = x[0] + 1

    if v[3] - v[2] < np.finfo(float).eps:  # No variation in y
        v[2] = y[0] - 1
        v[3] = y[0] + 1

    if not nocall:
        plt.axis(v)

    return v
