import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from scipy.ndimage import label
from construct_nodesCluster_new import construct_nodesCluster_new
from networkComponents import networkComponents

# from fillboundary_Group import fillboundary_Group

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def fillboundary_Group(groupMembers, nodes, color):
    for i in range(len(groupMembers)):
        member = groupMembers[i]

        try:
            # Check if 'centroid_c' and 'centroid_r' keys exist in nodes
            if "centroid_c" in nodes and "centroid_r" in nodes:
                col = nodes["centroid_c"][member].T
                row = nodes["centroid_r"][member].T

                # Ensure there are at least three points to form a convex hull
                if len(col) > 2:
                    try:
                        hull = ConvexHull(np.column_stack((row.T, col.T)))
                    except QhullError as e:
                        print(
                            f"Warning: QhullError encountered for cluster {i}. Skipping ConvexHull calculation."
                        )
                        print(f"Error details: {e}")
                        continue

                    # Calculate centroid of the convex hull
                    hull_centroid = np.mean(
                        np.column_stack((row[hull.vertices], col[hull.vertices])),
                        axis=0,
                    )

                    # Calculate polar angles of vertices with respect to the centroid
                    angles = np.arctan2(
                        col[hull.vertices] - hull_centroid[1],
                        row[hull.vertices] - hull_centroid[0],
                    )

                    # Sort vertices based on polar angles
                    sorted_indices = np.argsort(angles)
                    vertices = hull.vertices[sorted_indices]

                    # Add the first vertex to the end to close the loop
                    vertices = np.append(vertices, vertices[0])

                    plt.plot(row[vertices], col[vertices], color=color, linewidth=2)
                    plt.fill(row[vertices], col[vertices], color=color, alpha=0.7)
            else:
                print(
                    "Error: 'centroid_c' and/or 'centroid_r' keys not found in nodes."
                )

        except Exception as ex:
            print(
                f"Error: An unexpected error occurred for cluster {i}. Skipping this cluster."
            )
            print(f"Error details: {ex}")
            continue


def drawGraph_boundary_standard(I, coords, colors, a, r, lineWidth, markerSize):
    numGroups = len(coords)
    MM = [None] * numGroups
    plt.imshow(I, cmap="gray", origin="upper")
    for i in range(numGroups):
        alpha = a[i]
        # xx = np.array(coords[i]).T
        xx = (coords[i]).T

        Nodes = {"centroid_r": xx[1], "centroid_c": xx[0]}

        _, _, _, _, edges = construct_nodesCluster_new(
            {"centroid_r": coords[i][:, 1], "centroid_c": coords[i][:, 0]}, alpha, r
        )

        groupMatrix = edges
        MM[i] = groupMatrix

        _, _, groupMembers = networkComponents(groupMatrix)

        fillboundary_Group(groupMembers, Nodes, colors[i])
    # Don't show plot - will be saved by caller


import numpy as np
import matplotlib.pyplot as plt


def drawGraph_standard(
    coords, M, colors, lineWidth=3, markerSize=20, transpLine=1, transpMarker=0.8
):
    numGroups = len(coords)

    for k in range(numGroups):
        matrix = M[k]
        if len(matrix) > 0:
            centroids = coords[k]
            # print(centroids)
            numCent = len(centroids)

            for i in range(numCent):
                for j in range(i + 1, numCent):
                    if matrix[i, j] > 0:
                        # print("lineWidth:", lineWidth)
                        plt.plot(
                            [centroids[i, 1], centroids[j, 1]],
                            [centroids[i, 0], centroids[j, 0]],
                            color=colors[k],
                            linewidth=lineWidth,
                        )
                        plt.plot(
                            [centroids[i, 1], centroids[j, 1]],
                            [centroids[i, 0], centroids[j, 0]],
                            color="k",
                            linewidth=1,
                            linestyle="--",
                        )

            if markerSize > 0:
                scatter1 = plt.scatter(
                    centroids[:, 1],
                    centroids[:, 0],
                    s=markerSize,
                    facecolors=colors[k],
                    edgecolors="k",
                )
                scatter1.set_alpha(transpMarker)
    # Don't show plot - will be saved by caller


def SA_drawGraphsAndConvexHull_all(I, V30, V41, coords, colors, r, a):

    numGroups = len(coords)

    MM = []
    for i in range(numGroups):
        alpha = a[i]
        _, _, _, _, groupMatrix = construct_nodesCluster_new(
            {"centroid_r": coords[i][:, 1], "centroid_c": coords[i][:, 0]}, alpha, r
        )
        MM.append(groupMatrix)

    fig, ax = plt.subplots()
    ax.imshow(V41, cmap="gray", origin="upper")

    drawGraph_standard(coords, MM, colors)
    drawGraph_boundary_standard(I, coords, colors, a, r, 3, 3)

    # plt.savefig(visFile + '_4.png')

    # plt.show()
