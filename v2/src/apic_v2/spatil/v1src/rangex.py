import numpy as np


def rangex(x):
    # Calculates the range of x
    y = np.abs(np.max(x) - np.min(x))
    return y
