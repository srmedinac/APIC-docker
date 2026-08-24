import numpy as np
from skimage import io, morphology
from skimage.morphology import binary_dilation, disk
from skimage.measure import label
from normalize_staining import normalize_staining
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from get_nuclei_entropy import get_nuclei_entropy


def compute_pixel_list(image, mask, region_label):
    region_mask = label(mask) == region_label
    y, x = np.where(region_mask)
    pixel_list = np.column_stack([y, x])
    return pixel_list


def compute_centroid(pixel_list):
    if not pixel_list.size:
        return None

    centroid = np.mean(pixel_list, axis=0)
    return centroid


def get_nuclei_features(image, mask):

    # Convert mask to logical and open it
    mask = mask[:, :]
    mask = mask > 0  # Assuming the mask is not binary; adjust this based on your data
    mask = binary_dilation(mask, disk(1))
    image = normalize_staining(image)

    # Label connected components
    label_image = label(mask)

    # Get centroids and areas manually
    centroids = []
    areas = []

    for region_label in range(1, np.max(label_image) + 1):
        pixel_list = compute_pixel_list(image, mask, region_label)
        centroid = compute_centroid(pixel_list)
        area = np.sum(label_image == region_label)

        centroids.append(centroid)
        areas.append(area)

    centroids = np.array(centroids)
    areas = np.array(areas)

    # Continue with the rest of your code for feature extraction...
    gray_image = rgb2gray(image)

    # Initialize empty lists for features
    med_red = []
    entropy_red = []
    min_intensity = []
    max_intensity = []
    eccentricity = []
    ratio_axes = []

    # Extract features for each nucleus
    nuclei_num = len(centroids)
    for i in range(nuclei_num):
        centroid = centroids[i]
        y, x = int(centroid[0]), int(centroid[1])
        bbox = (
            max(0, np.min(y) - 15),
            max(0, np.min(x) - 15),
            min(image.shape[0], np.max(y) + 15),
            min(image.shape[1], np.max(x) + 15),
        )

        roi = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :]

        # Extract red channel
        R = roi[:, :, 0]

        # Get median intensity of red channel
        med_red.append(np.median(R))

        # Calculate entropy of red channel
        entropy_red.append(get_nuclei_entropy(R))

        # Get other features
        min_intensity.append(np.min(R))
        max_intensity.append(np.max(R))

        # Additional features
        eccentricity.append(1.0)  # Placeholder for eccentricity
        ratio_axes.append(1.0)  # Placeholder for ratio_axes

    # Combine features into a single array
    features = np.column_stack(
        [
            areas,
            eccentricity,
            ratio_axes,
            med_red,
            entropy_red,
            min_intensity,
            max_intensity,
        ]
    )

    # Create feature names list
    feature_names = [
        "Area",
        "Eccentricity",
        "RatioAxes",
        "MedianRed",
        "EntropyRed",
        "MinIntensity",
        "MaxIntensity",
    ]

    # Print the number of nuclei
    # print(f"Number of nuclei: {len(centroids)}")

    return centroids, features, feature_names
