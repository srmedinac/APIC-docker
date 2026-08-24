import numpy as np
from itertools import combinations
from shapely.geometry import Polygon, Point


def getIntersectionGroups(groups):
    features = []
    featureNames = []

    numGroups = len(groups)
    comb = list(combinations(range(1, numGroups + 1), 2))
    numComb = len(comb)

    polygons = {}

    for i in range(numGroups):
        col = groups[i]["clusterCentroids"][:, 0]
        row = groups[i]["clusterCentroids"][:, 1]

        if len(col) > 2:
            hull = Polygon(np.column_stack([col, row])).convex_hull
            polygons[i + 1] = {
                "coords": np.column_stack(hull.exterior.xy),
                "area": hull.area,
            }
        else:
            polygons[i + 1] = {"coords": np.array([]), "area": 0}

        # plt.plot(hull.exterior.xy[0], hull.exterior.xy[1], 'yellow', linewidth=5)

    for i in range(numComb):
        gr1, gr2 = comb[i]

        pol1 = polygons[gr1]["coords"]
        pol2 = polygons[gr2]["coords"]

        if len(pol1) > 0 and len(pol2) > 0:
            poly1 = Polygon(pol1)
            poly2 = Polygon(pol2)
            polyout = poly1.intersection(poly2)
            int_area = polyout.area

            in1 = np.zeros(len(groups[gr1 - 1]["clusterCentroids"]), dtype=bool)
            for centroid in groups[gr1 - 1]["clusterCentroids"]:
                point = Point(centroid[0], centroid[1])
                in1 |= poly2.contains(point)

            in2 = np.zeros(len(groups[gr2 - 1]["clusterCentroids"]), dtype=bool)
            for centroid in groups[gr2 - 1]["clusterCentroids"]:
                point = Point(centroid[0], centroid[1])
                in2 |= poly1.contains(point)

            avg_area = (polygons[gr1]["area"] + polygons[gr2]["area"]) / 2

            features.extend(
                [
                    int_area,
                    int_area / polygons[gr1]["area"],
                    int_area / polygons[gr2]["area"],
                    int_area / avg_area,
                    sum(in1),
                    sum(in2),
                ]
            )
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        strGr1 = str(gr1)
        strGr2 = str(gr2)

        featureNames.extend(
            [
                f"IntersectionArea_G{strGr1}&{strGr2}",
                f"RatioIntersectedArea_G{strGr1}&{strGr2}_ToArea_G{strGr1}",
                f"RatioIntersectedArea_G{strGr1}&{strGr2}_ToArea_G{strGr2}",
                f"RatioIntersectedArea_G{strGr1}&{strGr2}_ToAvgArea_G{strGr1}&{strGr2}",
                f"NumCentroidsClusters_G{strGr1}_InConvHull_G{strGr2}",
                f"NumCentroidsClusters_G{strGr2}_InConvHull_G{strGr1}",
            ]
        )

    return features, featureNames
