import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.stats import mode


def fillConvexHull_Group(members, nodes, ccc):
    numMembers = len(members)
    areas = []

    for i in range(numMembers):
        member = members[i]
        col = nodes["centroid_c"][member]
        row = nodes["centroid_r"][member]

        if len(col) > 2:
            points = np.column_stack((col, row))
            hull = ConvexHull(points)

            plt.plot(
                points[hull.vertices, 0],
                points[hull.vertices, 1],
                color=ccc,
                linewidth=4,
            )
            plt.fill(
                points[hull.vertices, 0], points[hull.vertices, 1], ccc, facealpha=0.5
            )

            areas.append(hull.volume)

    totalArea = sum(areas)
    avgArea = np.mean(areas)
    medianArea = np.median(areas)
    stdArea = np.std(areas)
    modeArea = mode(areas)[0][0]

    return totalArea, avgArea, medianArea, stdArea, modeArea
