import numpy as np


def eucl(crds1, crds2=None):
    if crds2 is None:
        if crds1.size == 0:
            print("EUCL: need at least two points")
            return np.nan

        N, P = crds1.shape
        if N < 2:
            print("EUCL: need at least two points")
            return np.nan

        crds1 = crds1.T
        dists = np.zeros((N, N))

        for i in range(N - 1):
            c1 = crds1[:, i][:, np.newaxis] * np.ones((1, N - i - 1))

            if P > 1:
                # print("c1:", c1,"\n")
                # print("crds1[:, (i):]:", crds1[:, (i):],"\n")
                d = np.sqrt(np.sum((c1 - crds1[:, (i + 1) :]) ** 2, axis=0))
            else:
                d = np.abs(c1 - crds1[:, (i + 1) :])

            dists[i, (i + 1) :] = d
            dists[(i + 1) :, i] = d.T

        if N == 2:
            dists = dists[0, 1]

    else:
        N1, P1 = crds1.shape
        N2, P2 = crds2.shape

        if P1 != P2:
            print("EUCL: sets of coordinates must be of the same dimension")
            return np.nan

        P = P1
        crds1 = crds1.T
        crds2 = crds2.T

        if N1 > 1 and N2 > 1:
            dists = np.zeros((N1, N2))
            for i in range(N1):
                c1 = np.tile(crds1[:, i][:, np.newaxis], (1, N2)).T
                if P > 1:
                    d = np.sqrt(np.sum((c1 - crds2) ** 2, axis=0))
                else:
                    d = np.abs(c1 - crds2)
                dists[i, :] = d

        elif N1 == 1 and N2 == 1:
            dists = np.sqrt(np.sum((crds1 - crds2) ** 2))

        elif N1 > 1 and N2 == 1:
            crds1 = crds1 - np.tile(crds2.T, (N1, 1))
            if P > 1:
                dists = np.sqrt(np.sum(crds1**2, axis=0))
            else:
                dists = np.abs(crds1)

        elif N1 == 1 and N2 > 1:
            crds2 = crds2 - np.tile(crds1.T, (N2, 1))
            if P > 1:
                dists = np.sqrt(np.sum(crds2**2, axis=0))
            else:
                dists = np.abs(crds2)

    return dists
