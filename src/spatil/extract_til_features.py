import numpy as np
from get_nuclei_features import get_nuclei_features
from get_spaTIL_features import get_spaTIL_features
from ESW_maker2 import ESW_maker2
from ROImaker import ROImaker
from SA_drawGraphsAndConvexHull_all import SA_drawGraphsAndConvexHull_all
from drawNucContoursByClass_SA2 import drawNucContoursByClass_SA2
from construct_nodesCluster_new import construct_nodesCluster_new
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import scipy.io


def extract_til_features(
    image,
    nuclei_mask,
    lymphocyte_mask,
    epi_mask,
    stroma_mask,
    histoqc_mask,
    draw_option,
    filename,
    alpha,
    r,
    output_viz_path=None,
):
    image_2 = image.copy()
    nuclei_centroids, n_1, n_2 = get_nuclei_features(image, nuclei_mask)

    nuclei_centroids_rounded = np.round(nuclei_centroids)

    epi_nuclei = np.zeros(len(nuclei_centroids_rounded), dtype=bool)
    tam_nuclei = np.zeros(len(nuclei_centroids_rounded), dtype=bool)
    features = np.array([])
    tam_nuclei_count = 0
    flag = 1

    for c in range(len(nuclei_centroids_rounded)):
        epi_nuclei[c] = epi_mask[
            int(nuclei_centroids_rounded[c, 0]), int(nuclei_centroids_rounded[c, 1])
        ]
        tam_nuclei[c] = lymphocyte_mask[
            int(nuclei_centroids_rounded[c, 0]), int(nuclei_centroids_rounded[c, 1])
        ]
        if (
            lymphocyte_mask[
                int(nuclei_centroids_rounded[c, 1]), int(nuclei_centroids_rounded[c, 0])
            ]
            == 1
        ):
            tam_nuclei_count += 1

    if len(nuclei_centroids_rounded) < 3:
        features = np.zeros(350)
    else:
        for c in range(len(nuclei_centroids_rounded)):
            epi_nuclei[c] = epi_mask[
                int(nuclei_centroids_rounded[c, 0]), int(nuclei_centroids_rounded[c, 1])
            ]
            tam_nuclei[c] = lymphocyte_mask[
                int(nuclei_centroids_rounded[c, 0]), int(nuclei_centroids_rounded[c, 1])
            ]

        coords = [nuclei_centroids[~tam_nuclei, :], nuclei_centroids[tam_nuclei, :]]
        features, feat_names = get_spaTIL_features(coords, alpha, r)
        # print(feat_names)
        if draw_option == 1 and output_viz_path:
            # Create clean visualization
            from construct_nodesCluster_new import construct_nodesCluster_new

            # Get spatial graph connections for both groups
            MM = []
            for i in range(len(coords)):
                alpha_val = alpha[i]
                _, _, _, _, groupMatrix = construct_nodesCluster_new(
                    {"centroid_r": coords[i][:, 1], "centroid_c": coords[i][:, 0]}, alpha_val, r
                )
                MM.append(groupMatrix)

            # Create figure with no axes
            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            ax.imshow(image, origin="upper")
            ax.axis('off')

            # Define colors: other nuclei (blue), lymphocytes (green)
            colors = [(0.2, 0.4, 0.8, 0.8), (0.2, 0.8, 0.3, 0.8)]  # Blue for other, Green for lymphocytes

            # Draw spatial graphs and nuclei centroids
            for k in range(len(coords)):
                matrix = MM[k]
                if len(matrix) > 0 and len(coords[k]) > 0:
                    centroids = coords[k]

                    # Draw connections
                    for i in range(len(centroids)):
                        for j in range(i + 1, len(centroids)):
                            if matrix[i, j] > 0:
                                ax.plot(
                                    [centroids[i, 1], centroids[j, 1]],
                                    [centroids[i, 0], centroids[j, 0]],
                                    color=colors[k], linewidth=1.5, alpha=0.6
                                )

                    # Draw nuclei centroids
                    ax.scatter(
                        centroids[:, 1], centroids[:, 0],
                        s=30, c=[colors[k]], edgecolors='white',
                        linewidths=0.5, alpha=0.9
                    )

            # Save with tight layout, no whitespace
            plt.savefig(output_viz_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    return features, flag


def extract_til_features_for_visualization(image, nuclei_mask, lymphocyte_mask, patch_coords):
   nuclei_centroids = get_nuclei_features(image, nuclei_mask)[0]
   x, y, w, h = patch_coords

   tam_nuclei = np.zeros(len(nuclei_centroids), dtype=bool)
   valid_indices = (nuclei_centroids[:,0].astype(int) < lymphocyte_mask.shape[0]) & \
                  (nuclei_centroids[:,1].astype(int) < lymphocyte_mask.shape[1])
   
   tam_nuclei[valid_indices] = lymphocyte_mask[
       nuclei_centroids[valid_indices,0].astype(int),
       nuclei_centroids[valid_indices,1].astype(int)
   ]
   
   lymphocyte_coords = nuclei_centroids[tam_nuclei] + np.array([x, y])
   other_nuclei_coords = nuclei_centroids[~tam_nuclei] + np.array([x, y])
   
   # Get connections for lymphocytes
   lymph_centroids = {"centroid_r": nuclei_centroids[tam_nuclei,0], 
                     "centroid_c": nuclei_centroids[tam_nuclei,1]}
   lymph_VX, lymph_VY, _, _, _ = construct_nodesCluster_new(lymph_centroids, alpha=0.6, r=0.07)
   
   # Get connections for other nuclei
   other_centroids = {"centroid_r": nuclei_centroids[~tam_nuclei,0],
                     "centroid_c": nuclei_centroids[~tam_nuclei,1]} 
   other_VX, other_VY, _, _, _ = construct_nodesCluster_new(other_centroids, alpha=0.6, r=0.07)
   
   # Convert to global coordinates
   lymph_connections = []
   for vx, vy in zip(lymph_VX, lymph_VY):
       start = np.array([vx[0], vy[0]]) + np.array([x, y])
       end = np.array([vx[1], vy[1]]) + np.array([x, y])
       lymph_connections.append([start.tolist(), end.tolist()])
       
   other_connections = []
   for vx, vy in zip(other_VX, other_VY):
       start = np.array([vx[0], vy[0]]) + np.array([x, y])
       end = np.array([vx[1], vy[1]]) + np.array([x, y])
       other_connections.append([start.tolist(), end.tolist()])

   return {
       'patch_coords': patch_coords,
       'lymphocytes': lymphocyte_coords.tolist(),
       'other_nuclei': other_nuclei_coords.tolist(),
       'lymph_connections': lymph_connections,
       'other_connections': other_connections
   }