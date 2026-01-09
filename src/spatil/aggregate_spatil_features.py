"""
Simple spaTIL Feature Aggregation
=================================
Aggregates patch-level spaTIL features to slide level using nanmean.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def aggregate_spatil_features(spatil_features_dir, slide_name, output_dir):
    """
    Aggregate patch-level spaTIL features to slide level
    
    Parameters:
    -----------
    spatil_features_dir : str or Path
        Directory containing spaTIL features (e.g., spatil_features/slide_name)
    slide_name : str
        Name of the slide
    output_dir : str or Path
        Output directory for aggregated features
    
    Returns:
    --------
    output_path : Path
        Path to saved CSV file
    """
    logger.info(f"Aggregating spaTIL features for slide: {slide_name}")
    
    # Get slide feature directory
    slide_features_dir = Path(spatil_features_dir) / slide_name
    csv_files = list(slide_features_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No spaTIL CSV files found for {slide_name}")
        return None
    
    # Load and concat all patch features
    patch_features = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)
        print(df.shape)
        if df.shape[0] == 350:  # Expected spaTIL feature count
            patch_features.append(df.iloc[:, 0].values)
    
    if not patch_features:
        logger.warning(f"No valid spaTIL features found for {slide_name}")
        return None
    
    # Stack and nanmean across patches
    features_matrix = np.stack(patch_features, axis=0)
    aggregated_features = np.nanmean(features_matrix, axis=0)
    
    # Create output DataFrame
    feature_names = [f"spaTIL_feat_{i:03d}" for i in range(len(aggregated_features))]
    data = {'slide_id': [slide_name]}
    data.update({name: [val] for name, val in zip(feature_names, aggregated_features)})
    
    df = pd.DataFrame(data)
    
    # Save
    output_path = Path(output_dir) / f"{slide_name}_spatil_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved aggregated spaTIL features: {len(aggregated_features)} features from {len(patch_features)} patches")
    return output_path