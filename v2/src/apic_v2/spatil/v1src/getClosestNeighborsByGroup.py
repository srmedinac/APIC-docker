import numpy as np


def getClosestNeighborsByGroup(groups, group_id):
    num_groups = len(groups)
    neighbors = [{} for _ in range(num_groups)]

    for j in range(num_groups):
        if group_id == j:
            dist = np.square(
                np.linalg.norm(
                    groups[group_id]["clusterCentroids"], axis=1, keepdims=True
                )
                - np.linalg.norm(
                    groups[group_id]["clusterCentroids"], axis=1, keepdims=True
                ).T
            )
            np.fill_diagonal(dist, np.inf)
        else:
            # Check if centroids array is not empty
            if (
                not groups[group_id]["clusterCentroids"].any()
                or not groups[j]["clusterCentroids"].any()
            ):
                continue  # Skip empty centroids array

            # Use broadcasting to calculate distances
            dist = np.linalg.norm(
                groups[group_id]["clusterCentroids"][:, np.newaxis, :]
                - groups[j]["clusterCentroids"][np.newaxis, :, :],
                axis=2,
            )

        if len(groups[j]["clusters"]) == 1:
            num_neigh = len(dist)
            if group_id == j:
                num_neigh = num_neigh - 1
            neighbors[j]["closest"] = np.ones(num_neigh, dtype=int)
            neighbors[j]["sortedDist"] = dist[:num_neigh]
        else:
            num_neigh = dist.shape[0]
            if group_id == j:
                num_neigh = num_neigh - 1
            idx = np.argsort(dist)
            neighbors[j]["closest"] = idx[:, :num_neigh].T
            neighbors[j]["sortedDist"] = np.sort(dist, axis=1)[:, :num_neigh]

    return neighbors
