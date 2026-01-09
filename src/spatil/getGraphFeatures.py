import numpy as np
from scipy.io import loadmat
from get_graph_features import (
    get_graph_features,
)  # Assuming you have a function for graph features


def getGraphFeatures(groups):
    features = []
    feature_names = []

    num_groups = len(groups)

    for i in range(num_groups):
        centroids = groups[i]["clusterCentroids"]

        # Assuming you have a file 'GraphFeatureDescription.mat' with 'GraphFeatureDescription' variable
        name_data = loadmat("src/spatil/GraphFeatureDescription.mat")
        names = [
            f"Graph{str(desc).replace(' ', '')}_G{i + 1}"
            for desc in name_data["GraphFeatureDescription"][0]
        ]
        # print(len(names))

        if centroids.shape[0] > 2:
            feat = get_graph_features(centroids[:, 0], centroids[:, 1])
            # feat = get_graph_features(centroids[:, 0], centroids[:, 1])  # Call your Python function here
        else:
            feat = np.zeros(len(names))

        features.extend(feat)
        feature_names.extend(names)

    return features, feature_names
