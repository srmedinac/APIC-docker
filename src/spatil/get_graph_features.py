import numpy as np
from scipy.spatial import Delaunay, Voronoi, distance
from scipy.io import loadmat
from mstree import mstree


def get_graph_features(x, y):
    # Load data from GraphFeatureDescription.mat
    data = loadmat("src/spatil/GraphFeatureDescription.mat")

    # Calculate the Voronoi diagram
    points = np.column_stack((x, y))
    vor = Voronoi(points)

    # Access Voronoi vertices
    V = vor.vertices

    # Access Voronoi regions
    C = vor.regions

    # Get the Delaunay triangulation
    tri = Delaunay(points)

    # Get the Minimum Spanning Tree (MST)
    _, mst_edgelen, _ = mstree(points, [], 0)

    # Record indices of inf and extreme values to skip these cells later
    Vnew = vor.vertices[1:]

    # Find the data points that lie far outside the range of the data
    Vsorted = np.sort(np.concatenate((Vnew[:, 0], Vnew[:, 1])))
    N = len(Vsorted)
    Q1 = round(0.25 * (N + 1))
    Q3 = round(0.75 * (N + 1))
    IQR = Q3 - Q1
    highrange = Q3 + 1.5 * IQR
    lowrange = Q1 - 1.5 * IQR
    Vextreme = np.concatenate((Vnew[Vnew > highrange], Vnew[Vnew < lowrange]))

    banned = []
    for i, cell in enumerate(vor.regions):
        if cell and (
            np.any(np.isinf(vor.vertices[cell, :]))
            or np.any(np.isin(vor.vertices[cell, :], Vextreme))
        ):
            banned.append(i)

    # If you've eliminated the whole thing (or most of it), then only ban
    # indices that are infinity (leave the outliers)
    if len(banned) > len(vor.regions) - 2:
        banned = [
            i
            for i, cell in enumerate(vor.regions)
            if np.any(np.isinf(vor.vertices[cell, :]))
        ]

    # Voronoi Diagram Features
    # Area
    c = 1
    d = 1
    e = d
    chorddist = []
    perimdist = []
    area = []
    for i, cell in enumerate(vor.regions):
        if i not in banned and cell:
            X = vor.vertices[cell, :]
            chord = distance.pdist(X)
            chorddist.extend(chord)

            perim = np.sqrt(np.sum(np.diff(X, axis=0) ** 2, axis=1))
            perimdist.extend(perim)

            area.append(np.abs(np.sum(np.cross(X, np.roll(X, 1, axis=0))) / 2))

    vfeature = np.zeros(51)

    if not area:
        print("No valid Voronoi regions.")
    else:
        vfeature[0] = np.std(area)
        vfeature[1] = np.mean(area)
        vfeature[2] = np.min(area) / np.max(area)
        vfeature[3] = 1 - (1 / (1 + (vfeature[0] / vfeature[1])))

        vfeature[4] = np.std(perimdist)
        vfeature[5] = np.mean(perimdist)
        vfeature[6] = np.min(perimdist) / np.max(perimdist)
        vfeature[7] = 1 - (1 / (1 + (vfeature[4] / vfeature[5])))

        vfeature[8] = np.std(chorddist)
        vfeature[9] = np.mean(chorddist)
        vfeature[10] = np.min(chorddist) / np.max(chorddist)
        vfeature[11] = 1 - (1 / (1 + (vfeature[8] / vfeature[9])))

    # Delaunay
    # Edge length and area
    c = 0
    d = 0
    sidelen = np.zeros(3 * len(tri.simplices))
    dis = np.zeros((len(tri.simplices), 3))
    triarea = np.zeros(len(tri.simplices))

    for i in range(len(tri.simplices)):
        t = points[tri.simplices[i]]

        sidelen[c : c + 3] = [
            np.sqrt(np.sum((t[0] - t[1]) ** 2)),
            np.sqrt(np.sum((t[0] - t[2]) ** 2)),
            np.sqrt(np.sum((t[1] - t[2]) ** 2)),
        ]

        dis[i, :] = np.sum(sidelen[c : c + 3])
        c += 3

        triarea[d] = np.abs(np.cross(t[0] - t[2], t[1] - t[2])) / 2
        d += 1

    vfeature[12] = np.min(sidelen) / np.max(sidelen)
    vfeature[13] = np.std(sidelen)
    vfeature[14] = np.mean(sidelen)
    vfeature[15] = 1 - (1 / (1 + (vfeature[13] / vfeature[14])))

    vfeature[16] = np.min(triarea) / np.max(triarea)
    vfeature[17] = np.std(triarea)
    vfeature[18] = np.mean(triarea)
    vfeature[19] = 1 - (1 / (1 + (vfeature[17] / vfeature[18])))

    # MST: Average MST Edge Length
    # The MST is a tree that spans the entire population in such a way that the
    # sum of the Euclidian edge length is minimal.
    vfeature[20] = np.mean(mst_edgelen)
    vfeature[21] = np.std(mst_edgelen)
    vfeature[22] = np.min(mst_edgelen) / np.max(mst_edgelen)
    vfeature[23] = 1 - (1 / (1 + (vfeature[21] / vfeature[20])))

    # Nuclear Features
    # Density
    vfeature[24] = np.sum(area)
    vfeature[25] = len(vor.regions)
    vfeature[26] = vfeature[25] / vfeature[24]

    # Average Distance to K-NN
    # Construct N x N distance matrix:
    distmat = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        for j in range(len(x)):
            distmat[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

    DKNN = np.zeros((3, len(distmat)))
    kcount = 0

    for K in [3, 5, 7]:
        # Calculate the summed distance of each point to its K nearest neighbors
        for i in range(len(distmat)):
            tmp = np.sort(distmat[i])

            # NOTE: when finding the summed distance, throw out the first result,
            # since it's the zero value at distmat(x,x). Add 1 to K to compensate.
            try:
                DKNN[kcount, i] = np.sum(tmp[1 : K + 1])
            except IndexError:
                DKNN[kcount, i] = 0

        kcount += 1

    # Average Distance to K-NN
    vfeature[27] = np.mean(DKNN[0, :])
    vfeature[28] = np.mean(DKNN[1, :])
    vfeature[29] = np.mean(DKNN[2, :])

    # Standard Deviation of Distance to K-NN
    vfeature[30] = np.std(DKNN[0, :])
    vfeature[31] = np.std(DKNN[1, :])
    vfeature[32] = np.std(DKNN[2, :])

    eps = np.finfo(float).eps

    # Relative Standard Deviation of Distance to K-NN
    vfeature[33] = 1 - (1 / (1 + (vfeature[30] / (vfeature[27] + eps))))
    vfeature[34] = 1 - (1 / (1 + (vfeature[31] / (vfeature[28] + eps))))
    vfeature[35] = 1 - (1 / (1 + (vfeature[32] / (vfeature[29] + eps))))

    # NNRR_av: Average Number of Neighbors in a Restricted Radius
    # Set the number of pixels within which to search
    rcount = 0
    NNRR_av = np.zeros((5,))
    NNRR_sd = np.zeros((5,))
    NNRR_dis = np.zeros((5,))

    for R in range(10, 60, 10):
        rcount += 1

        # For each point, find the number of neighbors within R pixels
        NNRR = np.array(
            [len(np.where(distmat[i] <= R)[0]) - 1 for i in range(len(distmat))]
        )

        if np.sum(NNRR) == 0:
            NNRR_av[rcount - 1] = 0
            NNRR_sd[rcount - 1] = 0
            NNRR_dis[rcount - 1] = 0
        else:
            NNRR_av[rcount - 1] = np.mean(NNRR)
            NNRR_sd[rcount - 1] = np.std(NNRR)
            NNRR_dis[rcount - 1] = 1 - (
                1 / (1 + (NNRR_sd[rcount - 1] / NNRR_av[rcount - 1]))
            )

    # Assign to vfeature
    vfeature[36:51] = np.concatenate([NNRR_av, NNRR_sd, NNRR_dis])

    # print(vfeature)

    return vfeature
