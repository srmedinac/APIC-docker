import numpy as np
import matplotlib.pyplot as plt
from rowtoval import rowtoval
from eucl import eucl
from sortmat import sortmat
from putbnd import putbnd


def mstree(crds, labels, doplot):
    if labels is None:
        labels = []
    if doplot is None:
        doplot = False

    # print(crds)

    n, p = crds.shape

    if doplot or p < 2:
        doplot = False

    if labels:
        if not isinstance(labels, str):
            labels = list(map(str, labels))

    totlen = 0
    edges = np.zeros((n - 1, 2), dtype=int)
    edgelen = np.zeros(n - 1)

    dist = eucl(crds)

    highval = np.max(dist) + 1

    e1 = np.ones(n - 1, dtype=int)
    e2 = np.arange(2, n + 1).T
    ed = dist[1:, 0]

    for edge in range(n - 1):
        mindist, i = np.min(ed), np.argmin(ed)
        t, u = e1[i], e2[i]
        totlen += mindist

        if t < u:
            edges[edge, :] = [t, u]
        else:
            edges[edge, :] = [u, t]

        edgelen[edge] = mindist

        if edge < n - 1:

            i = np.where(e2 == u)
            e1[i] = 0
            e2[i] = 0
            ed[i] = highval

            indx = np.where(e1 > 0)[0]

            for k in range(len(indx)):
                j = indx[k]
                t = e1[j] - 1
                v = e2[j] - 1
                u = u - 1

                if dist[u, v] < dist[t, v]:
                    e1[j] = u
                    ed[j] = dist[u, v]

    v, base = rowtoval(edges)
    # print("edges:" ,edges)
    v = v.T
    (
        v,
        outmat1,
        outmat2,
        outmat3,
        outmat4,
        outmat5,
        outmat6,
        outmat7,
        outmat8,
        outmat9,
    ) = sortmat(v, edges, edgelen, None, None, None, None, None, None, None)

    if doplot:
        plt.figure()
        plt.plot(crds[:, 0], crds[:, 1], "ok")
        putbnd(crds[:, 0], crds[:, 1])

        deltax = 0.018 * np.ptp(crds[:, 0])
        deltay = 0.02 * np.ptp(crds[:, 1])

        for i in range(n):
            lab = labels[i] if labels else str(i + 1)
            plt.text(crds[i, 0] + deltax, crds[i, 1] + deltay, lab)

        plt.hold(True)

        for i in range(n - 1):
            t, u = edges[i, 0], edges[i, 1]
            x = np.array([crds[t, 0], crds[u, 0]])
            y = np.array([crds[t, 1], crds[u, 1]])
            plt.plot(x, y, "-k")

        plt.hold(False)
        plt.show()

    return edges, edgelen, totlen
