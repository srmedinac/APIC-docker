import numpy as np


def get_absolute_closest_neighbors(groups, group_id, max_clust):
    num_groups = len(groups)
    num_clusters = len(groups[group_id]["clusters"])
    absolute_closest = []

    for j in range(num_clusters):
        dist_mat = np.full((num_groups, max_clust), np.inf)

        for k in range(num_groups):
            neighbor_data = groups[group_id]["neighborsPerGroup"][k]
            if "sortedDist" in neighbor_data:
                sorted_dist = neighbor_data["sortedDist"]

                if isinstance(sorted_dist, np.ndarray) and sorted_dist.size > 0:
                    if len(sorted_dist) == 1:
                        dist = sorted_dist[j]
                        dist_mat[k, 0] = dist
                    else:
                        if np.all(sorted_dist):
                            dist = sorted_dist[j, :]
                            dist_mat[k, : len(dist)] = dist
                else:
                    dist = []

        idx = np.argsort(dist_mat.flatten())
        x, y = np.unravel_index(idx, dist_mat.shape)

        num_idx = np.sum(~np.isinf(dist_mat))
        # close = np.zeros((num_idx, 2), dtype=int)
        close = []
        # closest_data = groups[x[k]]['neighborsPerGroup'][group_id]['closest']

        for k in range(num_idx):
            group_x = x[k]
            closest_data = groups[group_x]["neighborsPerGroup"][group_id]["closest"]
            # print(closest_data.shape[0])
            # print(closest_data.shape[1])
            if closest_data.shape == (0,):
                pass
            elif closest_data.shape[0] == 1:
                close.append((group_x, closest_data[0]))
            elif closest_data.size == 1:
                close.append((group_x, closest_data[0, 0]))
            elif closest_data.shape[0] > j and (
                closest_data.ndim > 1 and closest_data.shape[1] > y[k]
            ):
                close.append((group_x, closest_data[j, y[k]]))
            elif len(closest_data) == 0:
                pass

        absolute_closest.append({"idx": close})

    return absolute_closest
