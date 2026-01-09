"""

- Processes HoverNet nuclei segmentation outputs
- Extracts 8 morphological features per nucleus
- Computes spatial connectivity graphs 
- Calculates 24 Haralick texture features from co-occurrence matrices
- Aggregates features using 17 statistical measures
- Outputs 3,264 features total per slide (8 x 24 x 17)

"""

import os
import sys
import logging
import subprocess
import numpy as np
import cv2
from scipy import stats, sparse
from scipy.spatial.distance import pdist, squareform
from skimage import measure
import glob
import warnings
from itertools import combinations
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)


def regionprops_custom(binary_image):
    """
    Extract region properties matching MATLAB's regionprops
    
    Parameters:
    binary_image: Binary image with nuclei
    
    Returns:
    List of dictionaries with region properties
    """
    # Label connected components
    labeled_image = measure.label(binary_image, connectivity=2)
    
    # Get region properties
    props = measure.regionprops(labeled_image)
    if not props:
        return []
    
    centroids_png = np.array([p.centroid for p in props])
    idx_png = np.lexsort((centroids_png[:, 1], centroids_png[:, 0]))
    
    # Sort properties by centroid (top to bottom, then left to right)
    props = [props[i] for i in idx_png]
    custom_props = []
    
    for prop in props:
        # Skip very small regions
        if prop.area <= 5:
            continue
            
        # Calculate circularity (4*pi*area/perimeter^2)
        if prop.perimeter > 0:
            circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2)
        else:
            circularity = 0
            
        custom_props.append({
            'Centroid': prop.centroid,
            'Area': prop.area,
            'MajorAxisLength': prop.major_axis_length,
            'MinorAxisLength': prop.minor_axis_length,
            'Eccentricity': prop.eccentricity,
            'EquivDiameter': prop.equivalent_diameter,
            'Solidity': prop.solidity,
            'Perimeter': prop.perimeter,
            'Circularity': circularity
        })
    
    return custom_props


def get_stats_all(roi_data):
    """
    Calculate statistical measures for ROI data
    Matches MATLAB getStats_all.m exactly
    
    Parameters:
    roi_data: Array of values
    
    Returns:
    Array of 17 statistical measures
    """
    roi_padded = np.pad(roi_data, pad_width=1, mode='constant', constant_values=np.nan)
    
    # Flatten and remove NaN values
    roi_flat = roi_padded.flatten()
    valid_data = roi_flat[~np.isnan(roi_flat)]
    if len(valid_data) == 0:
        return np.full(17, np.nan)
    
    # Calculate statistics
    stats_vec = np.zeros(17)
    
    # 1. Mean
    stats_vec[0] = np.mean(valid_data)
    
    # 2. Variance (using N normalization to match MATLAB)
    stats_vec[1] = np.var(valid_data, ddof=0)
    
    # 3. Skewness
    stats_vec[2] = stats.skew(valid_data, nan_policy='omit')
    
    # 4. Kurtosis (subtract 3 to match MATLAB's convention)
    stats_vec[3] = stats.kurtosis(valid_data, nan_policy='omit', fisher=True)
    
    # 5. Median
    stats_vec[4] = np.median(valid_data)
    
    # 6. Minimum
    stats_vec[5] = np.min(valid_data)
    
    # 7. 10th percentile
    stats_vec[6] = np.percentile(valid_data, 10)
    
    # 8. 90th percentile
    stats_vec[7] = np.percentile(valid_data, 90)
    
    # 9. Maximum
    stats_vec[8] = np.max(valid_data)
    
    # 10. Interquartile range
    stats_vec[9] = np.percentile(valid_data, 75) - np.percentile(valid_data, 25)
    
    # 11. Range
    stats_vec[10] = stats_vec[8] - stats_vec[5]
    
    # 12. Mean absolute deviation
    stats_vec[11] = np.mean(np.abs(valid_data - stats_vec[0]))
    
    # 13. Robust mean absolute deviation
    robust_set = valid_data[(valid_data >= stats_vec[6]) & (valid_data <= stats_vec[7])]
    if len(robust_set) > 0:
        mean_robust = np.mean(robust_set)
        stats_vec[12] = np.mean(np.abs(robust_set - mean_robust))
    else:
        stats_vec[12] = np.nan
    
    # 14. Median absolute deviation
    stats_vec[13] = np.mean(np.abs(valid_data - stats_vec[4]))
    
    # 15. Coefficient of variation
    if stats_vec[0] != 0:
        stats_vec[14] = np.sqrt(stats_vec[1]) / stats_vec[0]
    else:
        stats_vec[14] = np.nan
    
    # 16. Energy
    stats_vec[15] = np.sum(valid_data ** 2)
    
    # 17. Root mean square
    stats_vec[16] = np.sqrt(np.mean(valid_data ** 2))
    
    return stats_vec


def construct_ccgs(bounds, alpha=0.5, r=0.2):
    """
    Construct CCGs (Connected Component Graphs)
    
    Parameters:
    -----------
    bounds : dict
        Dictionary with centroid_r and centroid_c lists
    alpha : float
        Alpha parameter for probability calculation
    r : float
        Threshold parameter for edge definition
        
    Returns:
    --------
    VX, VY : ndarray
        Edge coordinates
    edges : ndarray
        Adjacency matrix
    """
    # Get centroids
    centroid_r = np.array(bounds['centroid_r'])
    centroid_c = np.array(bounds['centroid_c'])
    n = len(centroid_r)
    
    if n < 2:
        return np.array([]), np.array([]), np.zeros((n, n))
    
    # Create coordinate matrix
    X = np.vstack([centroid_r, centroid_c]).T
    
    # Calculate pairwise distances
    D = squareform(pdist(X, metric='euclidean'))
    
    # Convert to probability matrix
    with np.errstate(divide='ignore'):
        P = np.where(D > 0, D ** (-alpha), 0)
    
    # Create adjacency matrix
    edges = (P > r)
    np.fill_diagonal(edges, 0)  # Remove self-loops
    
    # Get edge coordinates
    xx, yy = np.where(edges)
    
    # Create edge coordinate arrays
    if len(xx) > 0:
        VX = np.column_stack([centroid_r[xx], centroid_r[yy]])
        VY = np.column_stack([centroid_c[xx], centroid_c[yy]])
    else:
        VX = np.array([])
        VY = np.array([])
    
    return VX, VY, edges


def conncomp(adj_matrix):
    """
    Find connected components in graph (equivalent to MATLAB's graphconncomp)
    
    Parameters:
    adj_matrix: Adjacency matrix
    
    Returns:
    n_components: Number of connected components
    labels: Component labels for each node
    """
    # Convert to sparse matrix if not already
    if not sparse.issparse(adj_matrix):
        adj_matrix = sparse.csr_matrix(adj_matrix)
    
    # Find connected components
    n_components, labels = sparse.csgraph.connected_components(
        adj_matrix, directed=False, return_labels=True
    )
    
    labels = labels + 1
    
    return n_components, labels


def calc_glcm_features(glcm):
    """
    Calculate Haralick texture features from GLCM
    
    Parameters:
    glcm: Gray Level Co-occurrence Matrix
    
    Returns:
    Array of 24 Haralick features
    """
    # Normalize GLCM
    if np.sum(glcm) > 0:
        glcm = glcm / np.sum(glcm)
    
    n_gray = glcm.shape[0]
    features = np.zeros(24)
    
    # Compute marginal probabilities
    px = np.sum(glcm, axis=1)
    py = np.sum(glcm, axis=0)
    
    # Create coordinate matrices
    i, j = np.meshgrid(np.arange(1, n_gray + 1), np.arange(1, n_gray + 1), indexing='ij')
    
    # Feature 1: Maximum probability
    features[0] = np.max(glcm)
    
    # Feature 2: Joint average
    features[1] = np.sum(glcm * i)
    
    # Feature 3: Joint variance
    features[2] = np.sum(glcm * (i - features[1]) ** 2)
    
    # Feature 4: Entropy
    mask = glcm > 0
    if np.any(mask):
        features[3] = -np.sum(glcm[mask] * np.log2(glcm[mask]))
    
    # Feature 5-7: Difference statistics
    diff_prob = np.zeros(n_gray)
    for k in range(n_gray):
        if k == 0:
            diff_prob[k] = np.sum(np.diag(glcm))
        else:
            diff_prob[k] = np.sum(np.diag(glcm, k)) + np.sum(np.diag(glcm, -k))
    
    k_vals = np.arange(n_gray)
    features[4] = np.sum(k_vals * diff_prob)  # Difference average
    features[5] = np.sum((k_vals - features[4]) ** 2 * diff_prob)  # Difference variance
    mask = diff_prob > 0
    if np.any(mask):
        features[6] = -np.sum(diff_prob[mask] * np.log2(diff_prob[mask]))  # Difference entropy
    
    # Feature 8-10: Sum statistics
    sum_prob = np.zeros(2 * n_gray - 1)
    for k in range(2, 2 * n_gray + 1):
        sum_prob[k - 2] = np.sum(glcm[i + j == k])
    
    k_vals = np.arange(2, 2 * n_gray + 1)
    features[7] = np.sum(k_vals * sum_prob)  # Sum average
    features[8] = np.sum((k_vals - features[7]) ** 2 * sum_prob)  # Sum variance
    mask = sum_prob > 0
    if np.any(mask):
        features[9] = -np.sum(sum_prob[mask] * np.log2(sum_prob[mask]))  # Sum entropy
    
    # Feature 11: Energy
    features[10] = np.sum(glcm ** 2)
    
    # Feature 12: Contrast
    features[11] = np.sum((i - j) ** 2 * glcm)
    
    # Feature 13: Dissimilarity
    features[12] = np.sum(np.abs(i - j) * glcm)
    
    # Feature 14: Inverse difference
    features[13] = np.sum(glcm / (1 + np.abs(i - j)))
    
    # Feature 15: Inverse difference normalized
    features[14] = np.sum(glcm / (1 + np.abs(i - j) / n_gray))
    
    # Feature 16: Inverse difference moment
    features[15] = np.sum(glcm / (1 + (i - j) ** 2))
    
    # Feature 17: Inverse difference moment normalized
    features[16] = np.sum(glcm / (1 + (i - j) ** 2 / n_gray ** 2))
    
    # Feature 18: Inverse variance
    mask = (i != j)
    if np.any(mask):
        features[17] = 2 * np.sum(np.triu(glcm * mask / (i - j) ** 2, k=1))
    
    # Feature 19: Correlation
    mu_i = np.sum(np.arange(1, n_gray + 1) * px)
    mu_j = np.sum(np.arange(1, n_gray + 1) * py)
    sigma_i = np.sqrt(np.sum(((np.arange(1, n_gray + 1) - mu_i) ** 2) * px))
    
    if sigma_i > 0:
        features[18] = np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i ** 2)
    
    # Feature 20: Autocorrelation
    features[19] = np.sum(i * j * glcm)
    
    # Feature 21: Cluster tendency
    features[20] = np.sum((i + j - 2 * mu_i) ** 2 * glcm)
    
    # Feature 22: Cluster shade
    features[21] = np.sum((i + j - 2 * mu_i) ** 3 * glcm)
    
    # Feature 23: Cluster prominence
    features[22] = np.sum((i + j - 2 * mu_i) ** 4 * glcm)
    
    # Feature 24: Information correlation 1
    hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    
    pxy = np.outer(px, py)
    mask = (glcm > 0) & (pxy > 0)
    if np.any(mask):
        hxy1 = -np.sum(glcm[mask] * np.log2(pxy[mask]))
        
        if max(hx, hy) > 0:
            features[23] = (features[3] - hxy1) / max(hx, hy)
    
    # Handle NaN values
    features[np.isnan(features)] = 0
    
    return features


def nuclei_diversity_per_network(bounds, properties, alpha, r, feature_name, 
                                feature_max, feature_min):
    """
    Calculate nuclei diversity features per network
    """
    if len(properties) < 2:
        return []
    
    # Build cell cluster graph
    VX, VY, edges = construct_ccgs(bounds, alpha, r)
    
    # Make edges symmetric
    edges = np.maximum(edges, edges.T)
    
    # Find connected components
    n_components, group = conncomp(edges)
    
    # Remove single-node networks
    network_sizes = np.bincount(group)[1:]  # Skip 0 index
    valid_networks = np.where(network_sizes > 1)[0] + 1
    
    try:
        # Extract feature values safely
        feature_values = np.array([prop[feature_name] for prop in properties])
        
        if len(feature_values) == 0:
            return []
            
        # Feature discretization parameters
        num_levels = 6
        bin_size = (feature_max - feature_min) / num_levels
        
        # Clip values to expected range
        feature_values = np.clip(feature_values, feature_min, feature_max)
        
        # Discretize features
        feature_discrete = np.floor((feature_values - feature_min) / bin_size).astype(int) + 1
        feature_discrete = np.clip(feature_discrete, 1, num_levels)
        
        # Calculate features for each valid network
        features = []
        
        for network_id in valid_networks:
            # Get features in this network
            network_mask = (group == network_id)
            network_features = feature_discrete[network_mask]
            
            # Skip if too few unique values
            unique_vals = np.unique(network_features[~np.isnan(network_features)])
            if len(unique_vals) <= 2:
                continue
            
            # Build co-occurrence matrix
            cooc_matrix = np.zeros((num_levels, num_levels))
            
            # Get all pairs
            pairs = list(combinations(unique_vals, 2))
            for v1, v2 in pairs:
                count1 = np.sum(network_features == v1)
                count2 = np.sum(network_features == v2)
                cooc_matrix[v1-1, v2-1] = count1 * count2
            
            # Make symmetric
            cooc_matrix = cooc_matrix + cooc_matrix.T
            
            # Add diagonal (self-pairs)
            for val in unique_vals:
                count = np.sum(network_features == val)
                cooc_matrix[val-1, val-1] = count
            
            # Normalize
            if np.sum(cooc_matrix) > 0:
                cooc_matrix = cooc_matrix / np.sum(cooc_matrix)
            
            # Calculate GLCM features
            glcm_features = calc_glcm_features(cooc_matrix)
            features.append(glcm_features)
        
        return features
        
    except Exception as e:
        logger.warning(f"Error processing feature {feature_name}: {str(e)}")
        return []


def process_slide_nucdiv(nuclei_seg_dir, slide_name, config):
    """
    Process all nuclei segmentation masks for a single slide
    
    Parameters:
    nuclei_seg_dir: Directory containing nuclei segmentation masks
    slide_name: Name of the slide
    config: Nuclear diversity configuration
    
    Returns:
    Feature matrix for the slide (1 x 3264)
    """
    logger.info(f'Processing nuclear diversity features for slide: {slide_name}')
    
    # Configuration 
    num_shape_features = 8
    num_haralick_features = 24
    num_stats = 17
    
    feature_names = [
        'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'EquivDiameter', 'Solidity', 'Perimeter', 'Circularity'
    ]
    
    feature_max_values = [2000, 80, 15, 1, 60, 1, 180, 1]
    feature_min_values = [50, 5, 2, 0.4, 6, 0.8, 10, 0.85]
    
    # Parameters for CCG
    alpha_min = config.get('alpha_min', 0.44)
    alpha_max = config.get('alpha_max', 0.44)
    alpha_res = config.get('alpha_res', 0.02)
    radius = config.get('radius_threshold', 0.2)
    
    alpha_values = np.arange(alpha_min, alpha_max + alpha_res, alpha_res)
    
    # Get all PNG files in the slide folder
    slide_seg_dir = Path(nuclei_seg_dir) / slide_name
    png_files = list(slide_seg_dir.glob('*.png'))
    
    if not png_files:
        logger.warning(f"No nuclei segmentation files found in {slide_seg_dir}")
        total_features = num_shape_features * num_haralick_features * num_stats
        return np.full((1, total_features), np.nan)
    
    logger.info(f"Processing {len(png_files)} nuclei segmentation masks...")
    
    # Storage for features from all tiles
    all_tile_features = {feat_name: [] for feat_name in feature_names}
    
    # Process each tile
    for png_file in tqdm(png_files, desc="Processing nuclei masks"):
        try:
            # Load nuclei segmentation mask
            mask = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not read file {png_file}")
                continue
            
            # Convert HoverNet output to binary nuclei mask
            # HoverNet outputs: 0=background, >0=nuclei (different types)
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Extract region properties
            props = regionprops_custom(binary_mask)
            
            # Skip if too few nuclei
            min_nuclei = config.get('min_nuclei_per_patch', 10)
            if len(props) <= min_nuclei:
                logger.debug(f"Too few nuclei found in {png_file.name} ({len(props)} nuclei)")
                continue
                    
            # Extract centroids
            centroids = [p['Centroid'] for p in props]
            if not centroids:
                logger.warning(f"No valid centroids found in {png_file.name}")
                continue
                    
            bounds = {
                'centroid_r': [c[0] for c in centroids],  # Row coordinates
                'centroid_c': [c[1] for c in centroids]   # Column coordinates
            }
            
            # Extract features for all shape properties
            for feat_idx, feature_name in enumerate(feature_names):
                try:
                    feature_max = feature_max_values[feat_idx]
                    feature_min = feature_min_values[feat_idx]
                    
                    # Extract features at different spatial scales
                    for alpha in alpha_values:
                        tile_features = nuclei_diversity_per_network(
                            bounds, props, alpha, radius,
                            feature_name, feature_max, feature_min
                        )
                        
                        if tile_features:
                            all_tile_features[feature_name].extend(tile_features)
                            
                except Exception as e:
                    logger.warning(f"Error processing feature {feature_name} in {png_file.name}: {str(e)}")
                    continue
                            
        except Exception as e:
            logger.warning(f"Error processing file {png_file}: {str(e)}")
            continue
    
    # Process features for each shape property
    sample_features = []
    
    for feature_name in feature_names:
        feature_tiles = all_tile_features[feature_name]
        
        if feature_tiles:
            # Convert to matrix
            feature_matrix = np.array(feature_tiles).T
            
            # Calculate statistics for each Haralick feature
            aggregated_features = []
            for feat_col in range(feature_matrix.shape[0]):
                stats_vec = get_stats_all(feature_matrix[feat_col, :])
                aggregated_features.append(stats_vec)
            
            # Flatten to single row
            aggregated_features = np.concatenate(aggregated_features)
        else:
            # No features found - fill with NaN
            aggregated_features = np.full(num_haralick_features * num_stats, np.nan)
        
        sample_features.append(aggregated_features)
    
    # Combine all shape features
    if sample_features:
        nuclei_diversity_features = np.concatenate(sample_features)
    else:
        # Fill with NaN if no features
        total_features = num_shape_features * num_haralick_features * num_stats
        nuclei_diversity_features = np.full(total_features, np.nan)
    
    # Ensure correct shape
    nuclei_diversity_features = nuclei_diversity_features.reshape(1, -1)
    
    logger.info(f"Nuclear diversity feature extraction completed: {nuclei_diversity_features.shape[1]} features")
    
    return nuclei_diversity_features


def generate_feature_column_names():
    """
    Generate descriptive column names for nuclear diversity features
    
    Returns:
    List of column names
    """
    shape_features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
                     'EquivDiameter', 'Solidity', 'Perimeter', 'Circularity']
    
    haralick_features = [
        'MaxProb', 'JointAvg', 'JointVar', 'Entropy', 'DiffAvg', 'DiffVar',
        'DiffEnt', 'SumAvg', 'SumVar', 'SumEnt', 'Energy', 'Contrast',
        'Dissimilarity', 'InvDiff', 'InvDiffNorm', 'InvDiffMom', 'InvDiffMomNorm',
        'InvVar', 'Correlation', 'AutoCorr', 'ClusterTend', 'ClusterShade',
        'ClusterProm', 'InfoCorr1'
    ]
    
    stat_names = [
        'Mean', 'var', 'Skewness', 'Kurtosis', 'Median', 'Min',
        'Prcnt10', 'Prcnt90', 'Max', 'IQR', 'Range',
        'MAD', 'RMAD', 'MedAD', 'CoV', 'Energy', 'RMS'
    ]
    
    column_names = ['slide_id']  # Start with slide_id column
    for shape_feat in shape_features:
        for haralick_feat in haralick_features:
            for stat in stat_names:
                column_names.append(f"{shape_feat}.{haralick_feat}.{stat}")
    
    return column_names


def save_nucdiv_features(features, slide_name, output_dir):
    """
    Save nuclear diversity features to CSV file
    
    Parameters:
    features: Feature matrix (1 x 3264)
    slide_name: Name of the slide
    output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate column names
    column_names = generate_feature_column_names()
    
    # Create DataFrame
    feature_row = {'slide_id': slide_name}
    feature_values = features.flatten()
    
    # Add each feature value with its descriptive name
    for i, col_name in enumerate(column_names[1:]):  # Skip slide_id
        feature_row[col_name] = feature_values[i]
    
    df = pd.DataFrame([feature_row])
    
    # Save to CSV
    csv_path = output_dir / f"{slide_name}_nuclear_diversity_features.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Nuclear diversity features saved to: {csv_path}")
    
    return csv_path


def run_nuclear_diversity_extraction(nuclei_seg_dir, slide_name, output_dir, config):
    """
    Main function to run nuclear diversity feature extraction for a single slide
    
    Parameters:
    nuclei_seg_dir: Directory containing nuclei segmentation results
    slide_name: Name of the slide to process
    output_dir: Output directory for features
    config: Nuclear diversity configuration
    
    Returns:
    Path to saved feature file
    """
    try:
        # Process slide
        features = process_slide_nucdiv(nuclei_seg_dir, slide_name, config)
        
        # Save features
        output_path = save_nucdiv_features(features, slide_name, output_dir)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Nuclear diversity extraction failed for {slide_name}: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Nuclear Diversity Feature Extraction")
    parser.add_argument("--nuclei_seg_dir", required=True, help="Directory containing nuclei segmentation results")
    parser.add_argument("--slide_name", required=True, help="Name of the slide to process")
    parser.add_argument("--output_dir", required=True, help="Output directory for features")
    parser.add_argument("--alpha_min", type=float, default=0.44, help="Minimum alpha value")
    parser.add_argument("--alpha_max", type=float, default=0.44, help="Maximum alpha value")
    parser.add_argument("--radius_threshold", type=float, default=0.2, help="Radius threshold for connectivity")
    parser.add_argument("--min_nuclei_per_patch", type=int, default=10, help="Minimum nuclei per patch")
    
    args = parser.parse_args()
    
    config = {
        'alpha_min': args.alpha_min,
        'alpha_max': args.alpha_max,
        'alpha_res': 0.02,
        'radius_threshold': args.radius_threshold,
        'min_nuclei_per_patch': args.min_nuclei_per_patch
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run extraction
    output_path = run_nuclear_diversity_extraction(
        args.nuclei_seg_dir, 
        args.slide_name, 
        args.output_dir, 
        config
    )
    
    print(f"Nuclear diversity features saved to: {output_path}")