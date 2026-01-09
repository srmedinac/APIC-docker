import cv2
import numpy as np
from extract_til_features import extract_til_features_for_visualization
import re
import json
import os
from tqdm import tqdm

def process_slide_lymphocytes(slide_id, patches_dir, mask_dir):
   patch_dir = os.path.join(patches_dir, slide_id, "tiles")
   patches = [f for f in os.listdir(patch_dir) if f.endswith('.jpeg')]
   print(f"Found {len(patches)} patches")
   
   all_lymphocytes = []
   for patch_name in tqdm(patches):
       patch_path = os.path.join(patch_dir, patch_name)
       coords = re.findall(r'x(\d+)_y(\d+)_w(\d+)_h(\d+)', patch_name)[0]
       x, y, w, h = map(int, coords)
       
       bgr_image = cv2.imread(patch_path)
       image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
       
       mask_path = os.path.join(mask_dir, slide_id, f"x{x}_y{y}_w{w}_h{h}.png")
       if not os.path.exists(mask_path):
           continue
           
       nuclei_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
       if nuclei_mask is None:
           continue
           
       lymphocyte_value = 128
       unique_values = np.unique(nuclei_mask)
       
       if lymphocyte_value in unique_values:
           lymphocyte_mask = (nuclei_mask == lymphocyte_value).astype(np.uint8)
       else:
           lymphocyte_mask = np.zeros_like(nuclei_mask)
           
       lymph_data = extract_til_features_for_visualization(
           image, nuclei_mask, lymphocyte_mask, (x, y, w, h)
       )
       all_lymphocytes.append(lymph_data)
   
   return all_lymphocytes

def export_to_qupath(all_nuclei_data, output_path):
   geojson = {
       "type": "FeatureCollection",
       "features": []
   }
   
   for patch in all_nuclei_data:
       # Nuclei points
       for pos in patch['lymphocytes']:
           feature = {
               "type": "Feature",
               "geometry": {
                   "type": "Point",
                   "coordinates": pos
               },
               "properties": {
                   "classification": "Lymphocyte",
                   "color": [255, 0, 0, 128]
               }
           }
           geojson['features'].append(feature)
           
       for pos in patch['other_nuclei']:
           feature = {
               "type": "Feature",
               "geometry": {
                   "type": "Point",
                   "coordinates": pos
               },
               "properties": {
                   "classification": "Other Nuclei",
                   "color": [0, 0, 0, 128]
               }
           }
           geojson['features'].append(feature)
       
       # Lymphocyte connections
       for conn in patch['lymph_connections']:
           feature = {
               "type": "Feature",
               "geometry": {
                   "type": "LineString",
                   "coordinates": conn
               },
               "properties": {
                   "classification": "Lymphocyte Connection",
                   "color": [255, 255, 0, 128]  # Yellow
               }
           }
           geojson['features'].append(feature)
           
       # Other nuclei connections
       for conn in patch['other_connections']:
           feature = {
               "type": "Feature",
               "geometry": {
                   "type": "LineString",
                   "coordinates": conn
               },
               "properties": {
                   "classification": "Other Connection",
                   "color": [0, 255, 0, 128]  # Green
               }
           }
           geojson['features'].append(feature)
           
   with open(output_path, 'w') as f:
       json.dump(geojson, f)

# Usage
slide_id = "1ED085B7-0895-489D-93FE-E2948D83A79C"
patches_dir = "/Users/srmedinac/Downloads/"
mask_dir = "/Users/srmedinac/Downloads/masks/"
lymphocytes = process_slide_lymphocytes(slide_id, patches_dir, mask_dir)
export_to_qupath(lymphocytes, f"/Users/srmedinac/Downloads/{slide_id}_lymphocytes.json")