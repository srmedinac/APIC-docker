#!/usr/bin/env python3
"""
Adapted spaTIL Main Script for Pipeline Integration
=================================================

This script adapts your original spaTIL main script to work with command line arguments
from the biomarker pipeline.

Usage:
    python spatil_main_adapted.py --patches_dir /path/to/patches --nuclei_masks_dir /path/to/masks --results_features_dir /path/to/results --slide_name slide_name
"""

import os
from os.path import join, isfile
import cv2
import numpy as np
import argparse
import sys
from extract_til_features import extract_til_features
from tqdm.auto import tqdm
import warnings
import multiprocessing
from functools import partial

warnings.filterwarnings("ignore")

def process_image(image_path):
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image.astype(np.float32) / 255.0

def process_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def get_unprocessed_tiles(wsi, config):
    processed_tiles = set()
    wsi_result_dir = join(config["results_features_dir"], wsi)
    if os.path.exists(wsi_result_dir):
        for file in os.listdir(wsi_result_dir):
            if file.endswith('.csv'):
                processed_tiles.add(file.replace('.csv', '.png'))
    
    wsi_nuclei_masks_dir = join(config["nuclei_masks_dir"], wsi)
    if not os.path.exists(wsi_nuclei_masks_dir):
        print(f"Warning: Nuclei masks directory for WSI {wsi} does not exist.")
        return []
    
    all_masks = set(f for f in os.listdir(wsi_nuclei_masks_dir) if f.endswith('.png'))
    
    return list(all_masks - processed_tiles)

def process_wsi(wsi, config):
    os.makedirs(join(config["results_features_dir"], wsi), exist_ok=True)

    # Create visualization output directory if needed
    if config.get("viz_output_dir") and config.get("draw_option") == 1:
        os.makedirs(join(config["viz_output_dir"], wsi), exist_ok=True)

    unprocessed_tiles = get_unprocessed_tiles(wsi, config)
    total_tiles = len(unprocessed_tiles)
    processed_count = 0
    viz_count = 0

    if total_tiles == 0:
        print(f"No unprocessed tiles found for {wsi}")
        return 0

    # Progress bar for processing tiles
    print(f"Processing {total_tiles} tiles for {wsi}")
    for tile in tqdm(unprocessed_tiles, desc=f"Processing {wsi}", unit="tile", file=sys.stdout, ncols=80, position=0, leave=True):
        mask_paths = {
            "epi_stroma": join(config["epi_stroma_masks_dir"], wsi, tile),
            "nuclei": join(config["nuclei_masks_dir"], wsi, tile),
            "lymphocyte": join(config["lymphocyte_masks_dir"], wsi, tile)
        }
        
        if all(isfile(path) for path in mask_paths.values()):
            image_path = join(config["patches_dir"], wsi, "tiles", tile.replace(".png", ".jpeg"))
            if not isfile(image_path):
                print(f"Warning: Image file not found for {wsi}: {tile}")
                continue
            
            image = process_image(image_path)
            epi_stroma_mask = process_mask(mask_paths["epi_stroma"])
            nuclei_mask = process_mask(mask_paths["nuclei"])
            unique_values = np.unique(nuclei_mask)
            lymphocyte_value = 128
            if lymphocyte_value in unique_values:
                lymphocyte_mask = (nuclei_mask == lymphocyte_value).astype(np.uint8)
            else:
                lymphocyte_mask = np.zeros_like(nuclei_mask)
            if config['histoqc_mask'] is not None:
                epi_mask = cv2.bitwise_and(epi_stroma_mask, config["histoqc_mask"])
                stroma_mask = cv2.bitwise_and(cv2.bitwise_not(epi_stroma_mask), config["histoqc_mask"])
                lymphocyte_mask = cv2.bitwise_and(lymphocyte_mask, config["histoqc_mask"])
            else:
                config['histoqc_mask'] = np.ones(config["patch_size"], dtype=np.uint8)
                epi_mask = epi_stroma_mask
                stroma_mask = epi_stroma_mask
                
            nuclei_mask = cv2.bitwise_and(nuclei_mask, cv2.bitwise_not(lymphocyte_mask))

            # Determine if we should visualize this patch
            viz_path = None
            num_viz = config.get("num_viz_patches", 0)
            if config.get("viz_output_dir") and config.get("draw_option") == 1 and viz_count < num_viz:
                viz_filename = tile.replace(".png", ".png")
                viz_path = join(config["viz_output_dir"], wsi, viz_filename)

            features, flag = extract_til_features(
                    image,
                    nuclei_mask.astype(np.float32),
                    lymphocyte_mask.astype(np.float32),
                    epi_mask.astype(np.float32),
                    stroma_mask.astype(np.float32),
                    config["histoqc_mask"],
                    config["draw_option"] if viz_path else 0,
                    tile,
                    config["alpha"],
                    config["r"],
                    output_viz_path=viz_path
                )

            if viz_path:
                viz_count += 1
            
            if flag == 1:
                filename = tile.split(".png")[0]
                save_path = join(config["results_features_dir"], wsi, f"{filename}.csv")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savetxt(
                    save_path,
                    features,
                    delimiter=","
                )
                processed_count += 1
            else:
                print(f"Feature extraction failed for {tile}")
        else:
            print(f"Error: Not all required mask files found for {wsi}: {tile}")

    # Print summary
    print(f"\nSummary for {wsi}:")
    print(f"  - Total tiles to process: {total_tiles}")
    print(f"  - Successfully processed: {processed_count}")
    if viz_count > 0:
        print(f"  - Visualizations created: {viz_count}")

    return processed_count

def main():
    parser = argparse.ArgumentParser(description="spaTIL Feature Extraction")
    parser.add_argument("--patches_dir", required=True, help="Path to patches directory")
    parser.add_argument("--nuclei_masks_dir", required=True, help="Path to nuclei masks directory")
    parser.add_argument("--results_features_dir", required=True, help="Path to results directory")
    parser.add_argument("--viz_output_dir", help="Path to visualization output directory")
    parser.add_argument("--slide_name", required=True, help="Slide name to process")
    parser.add_argument("--alpha", nargs=2, type=float, default=[0.56, 0.56], help="Alpha parameters")
    parser.add_argument("--r", type=float, default=0.07, help="R parameter")
    parser.add_argument("--draw_option", type=int, default=0, help="Draw option (0=no viz, 1=save viz)")
    parser.add_argument("--num_processes", type=int, default=12, help="Number of processes")
    parser.add_argument("--num_viz_patches", type=int, default=10, help="Number of patches to visualize")

    args = parser.parse_args()

    # Build configuration from command line arguments
    config = {
        "patches_dir": args.patches_dir,
        "epi_stroma_masks_dir": args.nuclei_masks_dir,  # Use nuclei masks for epi/stroma
        "nuclei_masks_dir": args.nuclei_masks_dir,
        "lymphocyte_masks_dir": args.nuclei_masks_dir,  # Use nuclei masks for lymphocytes
        "results_features_dir": args.results_features_dir,
        "viz_output_dir": args.viz_output_dir,
        "draw_option": args.draw_option,
        "alpha": args.alpha,
        "r": args.r,
        "histoqc_mask": None,
        "patch_size": (1024, 1024),
        "num_processes": args.num_processes,
        "num_viz_patches": args.num_viz_patches
    }
    
    print(f"Processing slide: {args.slide_name}")
    print(f"Patches dir: {args.patches_dir}")
    print(f"Nuclei masks dir: {args.nuclei_masks_dir}")
    print(f"Results dir: {args.results_features_dir}")
    if args.viz_output_dir and args.draw_option == 1:
        print(f"Visualization dir: {args.viz_output_dir}")
        print(f"Number of patches to visualize: {args.num_viz_patches}")

    # Process the specific slide
    try:
        processed_count = process_wsi(args.slide_name, config)
        print(f"spaTIL feature extraction completed: {processed_count} patches processed")

    except Exception as e:
        print(f"Error processing slide {args.slide_name}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()