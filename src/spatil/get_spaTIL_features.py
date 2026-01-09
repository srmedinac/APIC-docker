import numpy as np
from construct_nodesCluster_new import construct_nodesCluster_new
from networkComponents import networkComponents
from getClusterProperties import getClusterProperties
from getClosestNeighborsByGroup import getClosestNeighborsByGroup
from get_absolute_closest_neighbors import get_absolute_closest_neighbors
from getClusterDensityMeasures_v2 import getClusterDensityMeasures_v2
from get_cluster_intersection_measures import get_cluster_intersection_measures
from get_neighborhood_measures import get_neighborhood_measures
from getGraphFeatures import getGraphFeatures
from getIntersectionGroups import getIntersectionGroups
from getGroupingFactorByGroup import getGroupingFactorByGroup


def get_spaTIL_features(
    coords, alpha=None, r=0.185, neigborhoodSize=5, groupingThreshold=0.005
):
    neigborhoodSize = 5
    groupingThreshold = 0.005
    if alpha is None:
        alpha = [0.42] * len(coords)
    maxNeigborsInters = 5
    numGroups = len(coords)

    featureNames = []
    clustPerGroup = np.zeros(numGroups, dtype=int)

    groups = []

    for i in range(numGroups):
        nodes = coords[i]
        # print(len(coords[i]))
        _, _, _, _, groupMatrix = construct_nodesCluster_new(
            {"centroid_r": coords[i][:, 1], "centroid_c": coords[i][:, 0]}, alpha[i], r
        )
        _, _, clusters = networkComponents(groupMatrix)

        clusters = [cluster for cluster in clusters if len(cluster) > 2]

        centroids, polygons, areas, densities = getClusterProperties(clusters, nodes)

        clustPerGroup[i] = len(clusters)
        featureNames.append(f"NumClusters_G{i}")

        groups.append(
            {
                "nodes": nodes,
                "clusters": clusters,
                "clusterCentroids": centroids,
                "clusterPolygons": polygons,
                "areas": areas,
                "densities": densities,
            }
        )

    # Identifying closest neighbors per group
    for i in range(numGroups):
        groups[i]["neighborsPerGroup"] = getClosestNeighborsByGroup(groups, i)

    maxNumClust = max(clustPerGroup)
    for i in range(numGroups):
        groups[i]["absoluteClosest"] = get_absolute_closest_neighbors(
            groups, i, maxNumClust
        )

    # Extracting cluster-related features
    densFeat, densFeatNames = getClusterDensityMeasures_v2(groups)
    intersClustFeat, intersClustFeatNames = get_cluster_intersection_measures(
        groups, maxNeigborsInters
    )
    richFeat, richFeatNames = get_neighborhood_measures(
        groups, neigborhoodSize, maxNumClust
    )

    # Extracting group-related features
    graphFeat, graphFeatNames = getGraphFeatures(groups)
    intersGroupFeat, intersGroupFeatNames = getIntersectionGroups(groups)
    groupingFeat, groupingFeatNames = getGroupingFactorByGroup(
        groups, groupingThreshold
    )

    # Define feature lists
    feature_lists = [
        clustPerGroup,
        densFeat,
        intersClustFeat,
        richFeat,
        graphFeat,
        intersGroupFeat,
        groupingFeat,
    ]

    # Concatenate lists
    features = np.concatenate([np.array(lst).reshape(-1) for lst in feature_lists])

    featureNames += (
        densFeatNames
        + intersClustFeatNames
        + richFeatNames
        + graphFeatNames
        + intersGroupFeatNames
        + groupingFeatNames
    )

    return features, featureNames
