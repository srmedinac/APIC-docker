import numpy as np
from scipy.stats import kurtosis, skew
from getSumNodeWeightsThreshold import getSumNodeWeightsThreshold


def getGroupingFactorByGroup(groups, grouping_thres):
    num_groups = len(groups)
    stats = ["Total", "Mean", "Std", "Median", "Max", "Min", "Kurtosis", "Skewness"]
    num_stats = len(stats)

    features = []
    feature_names = []

    for i in range(num_groups):
        feat_vector = getSumNodeWeightsThreshold(
            groups[i]["clusterCentroids"], "euclidean", grouping_thres
        )

        features.extend(
            [
                np.nansum(feat_vector),
                np.nanmean(feat_vector),
                np.nanstd(feat_vector),
                np.nanmedian(feat_vector),
                np.nanmax(feat_vector),
                np.nanmin(feat_vector),
                kurtosis(feat_vector, fisher=False),
                skew(feat_vector),
            ]
        )

        for st in range(num_stats):
            feature_names.append(stats[st] + "GroupingFactor_G" + str(i + 1))

    return features, feature_names
