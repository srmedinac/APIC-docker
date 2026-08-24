import numpy as np


def calc_frac(M):
    # M is an arbitrary binary mask in uint format where values are either 0 or 255
    frac = np.sum(np.sum(M.astype(float))) / (M.shape[0] * M.shape[1])
    return frac
