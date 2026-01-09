import numpy as np
import matplotlib.pyplot as plt


def construct_nodesCluster_new(bounds, alpha, r):
    X = np.array([bounds["centroid_r"], bounds["centroid_c"]])

    # distance matrix
    D = np.linalg.norm(X[:, np.newaxis, :] - X[:, :, np.newaxis], axis=0)

    # Suppress "divide by zero" warning during power operation
    with np.errstate(divide="ignore", invalid="ignore"):
        # probability matrix
        P = np.power(D, -alpha)

    # Replace inf values with a small positive number
    P[np.isinf(P)] = np.finfo(float).eps

    VX = []
    VY = []
    x = []
    y = []
    edges = np.zeros_like(D, dtype=int)
    z = 0
    t = 0
    convH = []
    tt = 0
    xloc = []

    for i in range(len(D) - 1):
        count = 0
        for j in range(i + 1, len(D)):
            if r < P[i, j]:
                edges[i, j] = 1
                VX.append([bounds["centroid_r"][i], bounds["centroid_r"][j]])
                xloc.append([i, j])
                VY.append([bounds["centroid_c"][i], bounds["centroid_c"][j]])

                x.append(bounds["centroid_r"][i])
                y.append(bounds["centroid_c"][i])
                t += 1

                x.append(bounds["centroid_r"][j])
                y.append(bounds["centroid_c"][j])
                t += 1

                z += 1
                count += 1
        # print(edges)
    return (
        np.array(VX),
        np.array(VY),
        np.array(x),
        np.array(y),
        edges,
    )  # , np.array(xloc), convH
