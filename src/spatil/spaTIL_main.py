import os
from os.path import join, isfile
import cv2
import numpy as np
from extract_til_features import extract_til_features
from tqdm.auto import tqdm
import warnings
import random
import multiprocessing
from functools import partial
import csv
import shutil
warnings.filterwarnings("ignore")

# Configuration
config = {
    "patches_dir": r"/Users/srmedinac/Desktop/CHAARTED/data/patches_latest_769",
    "epi_stroma_masks_dir": r"/Users/srmedinac/Desktop/CHAARTED/data/nuclei_segmentation_769",
    "nuclei_masks_dir": r"/Users/srmedinac/Desktop/CHAARTED/data/nuclei_segmentation_769",
    "lymphocyte_masks_dir": r"/Users/srmedinac/Desktop/CHAARTED/data/nuclei_segmentation_769",
    "results_features_dir": r"/Users/srmedinac/Desktop/CHAARTED/features/spaTIL/spaTIL_patches",
    "draw_option": 0,
    "alpha": [0.56, 0.56],
    "r": 0.07, # 0.07 for 1024. 0.04748 for 2048
    "histoqc_mask": None,
    "patch_size": (1024,1024),
    "test_run": False, # set to True to run on a single tile and test the visualization and pipeline
    "health_check": True,
    "num_processes": 12
}

def health_check(config, delete_extra=False):
    print("Running health check...")
    total_masks = 0
    total_csvs = 0
    extra_csvs = []

    for wsi in os.listdir(config["nuclei_masks_dir"]):
        mask_dir = join(config["nuclei_masks_dir"], wsi)
        csv_dir = join(config["results_features_dir"], wsi)
        
        if not os.path.isdir(mask_dir) or not os.path.isdir(csv_dir):
            continue

        masks = set(f[:-4] for f in os.listdir(mask_dir) if f.endswith('.png'))
        csvs = set(f[:-4] for f in os.listdir(csv_dir) if f.endswith('.csv'))
        
        total_masks += len(masks)
        total_csvs += len(csvs)
        
        extra = csvs - masks
        extra_csvs.extend(f"{wsi}/{f}.csv" for f in extra)

    print(f"Total masks: {total_masks}")
    print(f"Total CSVs: {total_csvs}")
    
    if extra_csvs:
        print(f"\nFound {len(extra_csvs)} CSVs without corresponding masks:")
        print(extra_csvs[:10])
        if len(extra_csvs) > 10:
            print(f"... and {len(extra_csvs) - 10} more.")
        
        if delete_extra:
            deleted_count = 0
            for csv_path in extra_csvs:
                full_path = join(config["results_features_dir"], csv_path)
                try:
                    os.remove(full_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {full_path}: {e}")
            print(f"\nDeleted {deleted_count} extra CSV files.")

    return extra_csvs

def process_image(image_path):
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image.astype(np.float32) / 255.0


def process_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def process_wsi(wsi, config):
    os.makedirs(join(config["results_features_dir"], wsi), exist_ok=True)
    
    unprocessed_tiles = get_unprocessed_tiles(wsi, config)
    processed_count = 0
    
    for tile in unprocessed_tiles:
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
            
            features, flag = extract_til_features(
                    image, 
                    nuclei_mask.astype(np.float32), 
                    lymphocyte_mask.astype(np.float32), 
                    epi_mask.astype(np.float32), 
                    stroma_mask.astype(np.float32),
                    config["histoqc_mask"], 
                    config["draw_option"], 
                    tile, 
                    config["alpha"], 
                    config["r"]
                )
            
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
    
    return processed_count

def process_chunk(chunk_and_id, config):
    chunk, chunk_id = chunk_and_id
    processed_count = 0
    for wsi in chunk:
        processed_count += process_wsi(wsi, config)
    return processed_count

def get_processed_wsis(config):
    processed_wsis = set()
    for wsi in os.listdir(config["results_features_dir"]):
        wsi_dir = join(config["results_features_dir"], wsi)
        if os.path.isdir(wsi_dir) and any(file.endswith('.csv') for file in os.listdir(wsi_dir)):
            processed_wsis.add(wsi)
    return processed_wsis

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

def main():
    if config["health_check"]:
        extra_csvs = health_check(config)
    if config["test_run"]:
        wsi = random.choice(os.listdir(config["patches_dir"]))
        tile = random.choice(os.listdir(join(config["patches_dir"], wsi, "tiles")))
    
        print(f"Processing random tile: {tile} from WSI: {wsi}")
        
        mask_paths = {
            "epi_stroma": join(config["epi_stroma_masks_dir"], wsi, tile.replace(".jpeg", ".png")),
            "nuclei": join(config["nuclei_masks_dir"], wsi, tile.replace(".jpeg", ".png")),
            "lymphocyte": join(config["lymphocyte_masks_dir"], wsi, tile.replace(".jpeg", ".png"))
        }
        
        if all(isfile(path) for path in mask_paths.values()):
            image = process_image(join(config["patches_dir"], wsi, "tiles", tile))
            epi_stroma_mask = process_mask(mask_paths["epi_stroma"])
            nuclei_mask = process_mask(mask_paths["nuclei"])
            unique_values = np.unique(nuclei_mask)
            print(f"Nuclei values: {unique_values}")
            lymphocyte_value = 128
            if lymphocyte_value in unique_values:
                lymphocyte_mask = (nuclei_mask == lymphocyte_value).astype(np.uint8)
            else:
                #print(f"Warning: No lymphocytes present in patch.")
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
            
            features, flag = extract_til_features(
                    image, 
                    nuclei_mask.astype(np.float32), 
                    lymphocyte_mask.astype(np.float32), 
                    epi_mask.astype(np.float32), 
                    stroma_mask.astype(np.float32),
                    config["histoqc_mask"], 
                    config["draw_option"], 
                    tile, 
                    config["alpha"], 
                    config["r"]
                )
                
            if flag == 1:
                filename = tile.split(".jpeg")[0]
                save_path = join(config["results_features_dir"], wsi, f"{filename}_test.csv")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savetxt(
                    save_path,
                    features,
                    delimiter=","
                )
                print(f"Test run completed. Features saved to {save_path}")
            else:
                print("Feature extraction failed for the test run.")
        else:
            print("Error: Not all required mask files found for the test run.")

    else:
        wsi_list = [wsi for wsi in os.listdir(config["nuclei_masks_dir"]) if os.path.isdir(join(config["nuclei_masks_dir"], wsi))]
        
        wsi_to_process = []
        total_tiles_to_process = 0
        print("Checking for unprocessed tiles...")
        for wsi in wsi_list:
            try:
                unprocessed_tiles = get_unprocessed_tiles(wsi, config)
                if unprocessed_tiles:
                    wsi_to_process.append(wsi)
                    total_tiles_to_process += len(unprocessed_tiles)
            except Exception as e:
                print(f"Error processing WSI {wsi}: {str(e)}")
        
        print(f"Total WSIs: {len(wsi_list)}")
        print(f"WSIs with unprocessed tiles: {len(wsi_to_process)}")
        print(f"Total tiles to process: {total_tiles_to_process}")
        
        num_processes = multiprocessing.cpu_count()
        wsi_chunks = np.array_split(wsi_to_process, config["num_processes"])
        print(f"Using {config['num_processes']} CPU cores")
        with multiprocessing.Pool(processes=config["num_processes"]) as pool:
            with tqdm(total=len(wsi_to_process), desc="WSI Progress") as pbar:
                for chunk in wsi_chunks:
                    results = pool.map(partial(process_wsi, config=config), chunk)
                    pbar.update(len(chunk))
        
        print(f"Processed {len(wsi_to_process)} WSIs in total")

if __name__ == "__main__":
    main()
