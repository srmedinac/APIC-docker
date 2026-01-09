import numpy as np
from scipy.spatial import ConvexHull, QhullError


def getClusterProperties(clusters, nodes):
    num_clusters = len(clusters)

    centroids = np.zeros((num_clusters, 2))
    polygons = [None] * num_clusters
    areas = np.zeros(num_clusters)
    densities = np.zeros(num_clusters)

    for i in range(num_clusters):
        member = clusters[i]
        col = nodes[member, 0]
        row = nodes[member, 1]

        # Check if all x-coordinates are the same
        if np.all(col == col[0]):
            print(
                f"Warning: All x-coordinates are the same for cluster {i}. Skipping ConvexHull calculation."
            )
            centroids[i, :] = [np.mean(col), np.mean(row)]
            continue

        try:
            hull = ConvexHull(np.column_stack((col, row)), qhull_options="QJ")
        except QhullError:
            print(f"Error: ConvexHull calculation failed for cluster {i}. Skipping.")
            continue

        areas[i] = hull.volume
        densities[i] = len(col) / areas[i]

        cx = np.mean(col[hull.vertices])
        cy = np.mean(row[hull.vertices])
        centroids[i, :] = [cx, cy]
        polygons[i] = np.column_stack((col[hull.vertices], row[hull.vertices]))

    return centroids, polygons, areas, densities


# Example usage:
# centroids, polygons, areas, densities = get_cluster_properties(clusters, nodes)
