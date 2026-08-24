import numpy as np
from getFeatureStats import getFeatureStats


def get_neighborhood_measures(groups, neighborhood_size, max_num_clusters):
    features = []
    feature_names = []

    stats = ["Total", "Mean", "Std", "Median", "Max", "Min", "Kurtosis", "Skewness"]
    num_stats = len(stats)

    num_groups = len(groups)

    for i in range(num_groups):
        num_clust = len(groups[i]["clusters"])

        if num_clust == 0:
            feat_stats = np.zeros((1, num_groups * neighborhood_size * num_stats))
        else:
            M = np.zeros((num_clust, num_groups * neighborhood_size))
            for j in range(num_clust):
                idx = 0
                num_rows = len(groups[i]["absoluteClosest"][j]["idx"])
                if num_rows > 0:
                    for k in range(1, neighborhood_size + 1):
                        if k <= max_num_clusters and k <= num_rows:
                            # print(groups[i]['absoluteClosest'][j]['idx'])
                            val = np.array(
                                [
                                    item[0]
                                    for item in groups[i]["absoluteClosest"][j]["idx"]
                                ]
                            )[:k]

                            for kk in range(num_groups):
                                idx += 1
                                M[j, idx - 1] = np.sum(val == kk) / k

                        else:
                            for kk in range(num_groups):
                                idx += 1
                                M[j, idx - 1] = M[j, idx - num_groups]

            if num_clust == 1:
                st_arr = [getFeatureStats(x) for x in M.T]
                feat_stats = np.zeros((num_stats, num_groups * neighborhood_size))
                num_val = len(st_arr)
                for xx in range(num_val):
                    val = st_arr[xx]
                    for yy in range(num_stats):
                        feat_stats[yy, xx] = val[yy]

            else:
                feat_stats = getFeatureStats(M)

        flat_feat_stats = [item for sublist in feat_stats for item in sublist]

        # Extend the features list with flat_feat_stats
        features.extend(flat_feat_stats)

        for st in range(num_stats):
            for k in range(neighborhood_size):
                for kk in range(num_groups):
                    feature_names.append(
                        f"{stats[st]}PercentageClusters_G{kk + 1}_Surrounding_G{i + 1}_Neighborhood{k + 1}"
                    )

    return features, feature_names
