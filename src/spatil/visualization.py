import cv2
import numpy as np
from extract_til_features import extract_til_features

def process_single_patch(
    patch_path,
    mask_dir,
    draw_option=1,
    alpha=[0.56, 0.56],
    r=0.07
):
    """
    Process a single patch and its corresponding masks.
    
    Args:
        patch_path: Path to the patch image (jpeg)
        mask_dir: Directory containing the mask files
        alpha: List of two alpha values for visualization
        r: Radius parameter for feature extraction
    """
    # Get the base name to find corresponding masks
    base_name = patch_path.split('/')[-1].replace('.jpeg', '.png')
    
    # Read and process the patch image
    bgr_image = cv2.imread(patch_path)
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Read masks
    epi_stroma_mask = cv2.imread(f"{mask_dir}/{base_name}", cv2.IMREAD_GRAYSCALE)
    nuclei_mask = cv2.imread(f"{mask_dir}/{base_name}", cv2.IMREAD_GRAYSCALE)
    
    # Process lymphocyte mask
    lymphocyte_value = 128
    unique_values = np.unique(nuclei_mask)
    if lymphocyte_value in unique_values:
        lymphocyte_mask = (nuclei_mask == lymphocyte_value).astype(np.uint8)
    else:
        lymphocyte_mask = np.zeros_like(nuclei_mask)
    
    # Prepare masks
    epi_mask = epi_stroma_mask
    stroma_mask = epi_stroma_mask
    nuclei_mask = cv2.bitwise_and(nuclei_mask, cv2.bitwise_not(lymphocyte_mask))
    histoqc_mask = np.ones_like(epi_mask, dtype=np.uint8)
    
    # Extract features and visualize
    features, flag = extract_til_features(
        image,
        nuclei_mask.astype(np.float32),
        lymphocyte_mask.astype(np.float32),
        epi_mask.astype(np.float32),
        stroma_mask.astype(np.float32),
        histoqc_mask,
        draw_option,
        base_name,
        alpha,
        r
    )
    
    return features, flag

# Example usage
if __name__ == "__main__":
    patch_path = '/Users/srmedinac/Documents/PhD/2025_code/PhD_code/feature_extraction_scripts/APIC_HUG/106/PT 106 - 2022-11-29/tiles/x20480_y8192_w1024_h1024.jpeg'
    mask_dir = "/Users/srmedinac/Documents/PhD/2025_code/PhD_code/feature_extraction_scripts/APIC_HUG/nucseg_106/106/PT 106 - 2022-11-29"
    features, flag = process_single_patch(patch_path, mask_dir)