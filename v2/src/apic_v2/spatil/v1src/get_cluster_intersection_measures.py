import itertools
import numpy as np
from shapely.geometry import Polygon
from itertools import combinations
from getFeatureStats import getFeatureStats


def get_cluster_intersection_measures(groups, n):
    features = []
    feature_names = []

    num_groups = len(groups)

    comb = list(combinations(range(num_groups), 2))
    num_comb = len(comb)

    for i in range(num_comb):
        group1 = groups[comb[i][0]]
        group2 = groups[comb[i][1]]

        int_areas = []
        int_rel1 = []
        int_rel2 = []
        int_rel3 = []

        num_clust1 = len(group1["clusters"])
        # print(num_clust1)
        for j in range(num_clust1 - 1):
            pol1 = group1["clusterPolygons"][j]
            area1 = group1["areas"][j]

            # Check if 'neighborsPerGroup' and 'comb[i][1]' exist
            if "neighborsPerGroup" in group1 and comb[i][1] < len(
                group1["neighborsPerGroup"]
            ):
                neighbor_data = group1["neighborsPerGroup"][comb[i][0]]

                # Check if 'closest' key exists in neighbor_data
                if "closest" in neighbor_data:
                    # print(j)
                    closest_indices = neighbor_data["closest"][j]
                    # print(closest_indices.shape)
                    if (closest_indices.shape) == 1:
                        num_closest = min(n, closest_indices)
                    else:
                        num_closest = min(n, len(closest_indices))

                        num_clust2 = len(group2["clusters"])
                        num_closest = min(num_closest, num_clust2)

                        for k in range(num_closest):
                            closest_index = closest_indices[k]

                            # Check if closest_index is a valid index for 'clusterPolygons' in group2
                            if 0 <= closest_index < len(group2["clusterPolygons"]):
                                pol2 = group2["clusterPolygons"][closest_index]
                                area2 = group2["areas"][closest_index]

                                polyarray1 = Polygon(pol1)
                                polyarray2 = Polygon(pol2)
                                polyout = polyarray1.intersection(polyarray2)

                                if not polyout.is_empty:
                                    int_area = polyout.area

                                    int_areas.append(int_area)
                                    int_rel1.append(int_area / area1)
                                    int_rel2.append(int_area / area2)
                                    int_rel3.append(2 * int_area / (area1 + area2))

        if not int_areas:
            features.extend([0] * (8 * 4))
        else:
            features.extend(
                getFeatureStats(int_areas)
                + getFeatureStats(int_rel1)
                + getFeatureStats(int_rel2)
                + getFeatureStats(int_rel3)
            )

        gr1 = str(comb[i][0])
        gr2 = str(comb[i][1])
        stats = ["Total", "Mean", "Std", "Median", "Max", "Min", "Kurtosis", "Skewness"]
        meas = [
            f"IntersectedAreaClusters_G{gr1}&{gr2}",
            f"RatioIntersectedAreaClusters_G{gr1}&{gr2}_ToArea_G{gr1}",
            f"RatioIntersectedAreaClusters_G{gr1}&{gr2}_ToArea_G{gr2}",
            f"RatioIntersectedAreaClusters_G{gr1}&{gr2}_ToAvgArea_G{gr1}&{gr2}",
        ]

        for ms in meas:
            feature_names.extend([f"{stat}{ms}" for stat in stats])

    return features, feature_names
