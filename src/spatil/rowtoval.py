import numpy as np
from rangex import rangex


def rowtoval(x, base=None):
    if base is None:
        base = []

    x = np.array(x, dtype=float)
    r, c = x.shape
    xmin = np.min(x)
    x = x - xmin

    if not base:
        base = int(np.max(x) + 1)
        # print(base)
    vals = np.zeros(r)

    for i in range(c):
        vals = vals + x[:, i] * base ** (c - i)

    return vals, base
