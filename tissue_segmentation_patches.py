import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import warnings

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BiomarkerPipelineStep1_2:
    
    def __init__(self, input_slide, output_dir, config_path=None):
        self.input_slide = Path(input_slide).resolve()
        self.slide_name = self.input_slide.stem
        self.slide_name_ext = self.input_slide.name
        self.output_dir = Path(output_dir).resolve()
        self.config = self.load_config(config_path)
        
        self.slide_output_dir = self.output_dir / self.slide_name
        self.dirs = {
            'qc': self.slide_output_dir / "qc",
            'patches': self.slide_output_dir / "patches",
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[STEP1_2] Pipeline initialized for: {self.slide_name}")
    
    def load_config(self, config_path):
        # minimal config needed for steps 1 & 2
        default_config = {
            'pipeline': {'resume_on_existing': True},
            'tissue_segmentation': {'level': 2},
            'patch_extraction': {
                'level': 0, 'patch_size': 1024, 'overlap': 0,
                'max_background': 0.75, 'num_workers': 4
            },
        }
        # you can later add YAML loading if needed
        return default_config
    
    def _find_histoqc_mask(self):
        """Find the HistoQC mask PNG for this slide."""
        qc_root = self.dirs['qc']

        # 1) Try the usual HistoQC naming: *_mask_use*.png
        candidates = list(qc_root.glob("**/*mask_use*.png"))

        # 2) If nothing, fall back to any *mask*.png
        if not candidates:
            candidates = list(qc_root.glob("**/*mask*.png"))

        if not candidates:
            raise FileNotFoundError(
                f"No HistoQC mask found under {qc_root}. "
                f"Expected a '*mask_use*.png' or '*mask*.png' file."
            )

        # Prefer a file that contains the full slide name with extension
        for c in candidates:
            if self.slide_name_ext in c.as_posix():
                return c

        # Next, prefer something containing slide_name (without extension)
        for c in candidates:
            if self.slide_name in c.stem:
                return c

        # Fallback: just return the first candidate
        return candidates[0]


    def segment_tissue(self):
        """Step 1: Tissue segmentation with HistoQC"""
        logger.info("[STEP1] Tissue segmentation with HistoQC")

        # Try to reuse existing mask if present
        try:
            existing_mask = self._find_histoqc_mask()
            if self._skip_if_exists(existing_mask):
                self.tissue_mask_path = existing_mask
                logger.info(f"[STEP1] Using existing mask: {existing_mask}")
                return
        except FileNotFoundError:
            # No existing mask, we will run HistoQC
            logger.info("[STEP1] No existing HistoQC mask found, running HistoQC now.")

        # Make sure qc/<something> exists (HistoQC may create its own subfolder)
        self.dirs['qc'].mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "histoqc",
            "-c", "/app/src/config_v3.ini",
            "-n", "3",
            "--force",                
            "-o", str(self.dirs['qc']),
            str(self.input_slide),
        ]

        # Try to cd into the HistoQC repo if present, run the command, then cd back.
        histoqc_dir = None
        for candidate in (Path("HistoQC"), Path("histoqc")):
            if candidate.exists() and candidate.is_dir():
                histoqc_dir = candidate.resolve()
                break

        if histoqc_dir:
            old_cwd = Path.cwd()
            try:
                os.chdir(str(histoqc_dir))
                subprocess.run(cmd, check=True)
            finally:
                os.chdir(str(old_cwd))
        else:
            # Fallback: run without changing cwd
            subprocess.run(cmd, check=True)

        # After running HistoQC, locate the actual mask file
        self.tissue_mask_path = self._find_histoqc_mask()

        # Optional: clean up extra QC PNGs, keep only the mask we found
        for png in self.dirs['qc'].glob("**/*.png"):
            if png.resolve() != self.tissue_mask_path.resolve():
                try:
                    png.unlink()
                    logger.debug(f"[STEP1] Deleted extra QC file: {png}")
                except Exception as e:
                    logger.warning(f"[STEP1] Could not delete {png}: {e}")

        logger.info(f"[STEP1] Tissue segmentation completed. Mask: {self.tissue_mask_path}")



    def extract_patches(self):
        """Step 2: Patch extraction using the HistoQC mask"""
        logger.info("[STEP2] Patch extraction")

        expected_tiles_dir = self.dirs['patches'] / self.slide_name / "tiles"
        if self._skip_if_exists(expected_tiles_dir) and any(expected_tiles_dir.glob("*.jpeg")):
            logger.info("[STEP2] Patches already exist, skipping.")
            return

        # Ensure tissue mask exists (HistoQC mask)
        if not hasattr(self, 'tissue_mask_path'):
            try:
                self.tissue_mask_path = self._find_histoqc_mask()
                logger.info(f"[STEP2] Using existing mask: {self.tissue_mask_path}")
            except FileNotFoundError:
                logger.info("[STEP2] No mask found yet, running segment_tissue() first.")
                self.segment_tissue()

        # Double-check that the path exists
        if not Path(self.tissue_mask_path).exists():
            raise FileNotFoundError(
                f"[STEP2] Tissue mask path does not exist: {self.tissue_mask_path}"
            )

        import openslide

        slide = openslide.OpenSlide(str(self.input_slide))
        config = self.config['patch_extraction']

        level = config['level']
        patch_size = int(config['patch_size'])
        overlap = config['overlap']
        max_background = float(config['max_background'])

        # Handle overlap as either fraction or pixels
        if overlap < 1:
            stride = int(patch_size * (1.0 - overlap))
        else:
            stride = patch_size - int(overlap)
        stride = max(1, stride)

        level_w, level_h = slide.level_dimensions[level]  # (width, height)

        # Load tissue mask (HistoQC mask) and map it to this level
        tissue_mask = np.array(Image.open(self.tissue_mask_path))
        if tissue_mask.ndim == 3:
            tissue_mask = tissue_mask[:, :, 0]  # assume single-channel
        mask_h, mask_w = tissue_mask.shape[:2]

        scale_x = level_w / mask_w
        scale_y = level_h / mask_h

        # Prepare output dirs
        tiles_dir = expected_tiles_dir
        tiles_dir.mkdir(parents=True, exist_ok=True)

        # Also create a thumbnail for later overlay step (used by the other env)
        thumbnail_dir = self.dirs['patches'] / self.slide_name
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumbnail_dir / "thumbnail.jpeg"
        if not thumb_path.exists():
            thumb = slide.get_thumbnail((1024, 1024))
            thumb.convert("RGB").save(thumb_path, "JPEG")

        kept = 0
        total = 0

        logger.info(
            f"[STEP2] Patch extraction on level {level} with "
            f"patch_size={patch_size}, stride={stride}, max_background={max_background}"
        )

        y = 0
        while y + patch_size <= level_h:
            x = 0
            while x + patch_size <= level_w:
                total += 1

                # Map level coords to mask coords
                mx0 = int(x / scale_x)
                mx1 = int((x + patch_size) / scale_x)
                my0 = int(y / scale_y)
                my1 = int((y + patch_size) / scale_y)

                # Clip to mask bounds
                mx0 = max(0, min(mask_w, mx0))
                mx1 = max(0, min(mask_w, mx1))
                my0 = max(0, min(mask_h, my0))
                my1 = max(0, min(mask_h, my1))

                if mx1 <= mx0 or my1 <= my0:
                    x += stride
                    continue

                patch_mask = tissue_mask[my0:my1, mx0:mx1]
                tissue_fraction = np.mean(patch_mask > 0)

                # Skip mostly background patches
                if 1.0 - tissue_fraction > max_background:
                    x += stride
                    continue

                # Read and save patch
                region = slide.read_region(
                    (x, y),  # top-left in level coords
                    level,
                    (patch_size, patch_size),
                ).convert("RGB")

                patch_name = f"{self.slide_name}_x{x}_y{y}_l{level}.jpeg"
                region.save(tiles_dir / patch_name, "JPEG")
                kept += 1

                x += stride
            y += stride

        logger.info(f"[STEP2] Extracted {kept} patches (evaluated {total} positions)")

    def _skip_if_exists(self, path=None):
        if not self.config['pipeline']['resume_on_existing']:
            return False
        return path.exists() if path else False

    def run_pipeline(self, steps=None):
        """Only steps: tissue, patches"""
        if steps is None:
            steps = ['tissue', 'patches']
        
        step_map = {
            'tissue': self.segment_tissue,
            'patches': self.extract_patches,
        }
        
        logger.info(f"[STEP1_2] Starting pipeline (steps: {steps}) for: {self.slide_name}")
        
        for step in steps:
            if step in step_map:
                try:
                    step_map[step]()
                except Exception as e:
                    logger.error(f"[STEP1_2] Step '{step}' failed: {e}")
                    raise
        
        logger.info("[STEP1_2] Pipeline (steps 1–2) completed.")


def main():
    parser = argparse.ArgumentParser(description="Computational Biomarker Pipeline - Steps 1 & 2 (HistoQC + patches)")
    parser.add_argument("--input", "-i", required=True, help="Input slide file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--config", "-c", help="Configuration YAML file (optional, default config used)")
    parser.add_argument("--steps", nargs='+', help="Specific steps to run: tissue patches")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input slide not found: {args.input}")
        sys.exit(1)
    
    try:
        pipeline = BiomarkerPipelineStep1_2(args.input, args.output, args.config)
        pipeline.run_pipeline(args.steps)
    except Exception as e:
        logger.error(f"Step1_2 pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import time
    print("Starting Step 1 & 2 pipeline...")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"Pipeline started at {start_time}")
    main()
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    total_time = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
    logger.info(f"Time taken to process steps 1&2: {round(total_time/60)} minutes")

#!/usr/bin/env bash
# run_step1_2.sh
# conda activate histoqc_env
# python pipeline_step1_2_histoqc.py \
#     -i /home/vputcha/emory/APIC-container-1/test_data/120329.svs \
#     -o /home/vputcha/emory/APIC-container-1/test_data/results
