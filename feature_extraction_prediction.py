#!/usr/bin/env python3
"""
APIC Pipeline Steps 3-8: Feature Extraction and Biomarker Prediction

This pipeline processes whole slide images through:
  - Step 3: Nuclei segmentation (HoVerNet)
  - Step 4: spaTIL feature extraction
  - Step 5: Nuclear diversity features (MATLAB)
  - Step 6: Feature aggregation
  - Step 7: Biomarker prediction (Cox model)
  - Step 8: APIC report generation
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import warnings
from PIL import Image

# Suppress common library warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Allow large images
Image.MAX_IMAGE_PIXELS = None

def setup_logging():
    """Configure logging with timestamps and flush-on-write."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout
    )
    # Force immediate flush
    for handler in logging.root.handlers:
        handler.flush()
    return logging.getLogger(__name__)

logger = setup_logging()


def log_step_header(step_name: str, step_num: int = None):
    """Print a clear step header."""
    if step_num:
        header = f"STEP {step_num}: {step_name}"
    else:
        header = step_name
    logger.info("=" * 60)
    logger.info(header)
    logger.info("=" * 60)
    sys.stdout.flush()


def log_step_complete(step_name: str, duration_sec: float = None):
    """Log step completion with optional duration."""
    if duration_sec:
        if duration_sec < 60:
            logger.info(f"✓ {step_name} completed ({duration_sec:.1f}s)")
        else:
            logger.info(f"✓ {step_name} completed ({duration_sec/60:.1f}min)")
    else:
        logger.info(f"✓ {step_name} completed")
    sys.stdout.flush()


def run_subprocess(cmd: list, description: str = None, env=None):
    """
    Run subprocess with real-time output streaming.

    Args:
        cmd: Command list to execute
        description: Optional description for logging
        env: Optional environment variables
    """
    if description:
        logger.info(f"Running: {description}")

    # Use Popen for real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    # Stream output in real-time
    for line in process.stdout:
        print(line, end='', flush=True)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


SHAPE_FEATURES = [
    'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
    'EquivDiameter', 'Solidity', 'Perimeter', 'Circularity'
]

HARALICK_FEATURES = [
    'MaxProb', 'JointAvg', 'JointVar', 'Entropy', 'DiffAvg', 'DiffVar',
    'DiffEnt', 'SumAvg', 'SumVar', 'SumEnt', 'Energy', 'Contrast',
    'Dissimilarity', 'InvDiff', 'InvDiffNorm', 'InvDiffMom', 'InvDiffMomNorm',
    'InvVar', 'Correlation', 'AutoCorr', 'ClusterTend', 'ClusterShade',
    'ClusterProm', 'InfoCorr1'
]

STAT_NAMES = [
    'Mean', 'var', 'Skewness', 'Kurtosis', 'Median', 'Min',
    'Prcnt10', 'Prcnt90', 'Max', 'IQR', 'Range',
    'MAD', 'RMAD', 'MedAD', 'CoV', 'Energy', 'RMS'
]


def generate_nucdiv_column_names():
    """Generate column names for nuclear diversity features (3264 total)."""
    return [
        f"{shape}.{haralick}.{stat}"
        for shape in SHAPE_FEATURES
        for haralick in HARALICK_FEATURES
        for stat in STAT_NAMES
    ]


def convert_mat_to_csv(mat_path: Path, csv_path: Path, slide_name: str):
    """Convert MATLAB .mat nuclear diversity features to CSV format."""
    mat_data = sio.loadmat(str(mat_path))
    features = mat_data['nucleiDiversityFeatures_shape'].flatten()

    column_names = generate_nucdiv_column_names()

    if len(features) != len(column_names):
        raise ValueError(
            f"Feature count mismatch: .mat has {len(features)}, expected {len(column_names)}"
        )

    # Build feature dictionary
    feature_dict = {'slide_id': slide_name}
    feature_dict.update(zip(column_names, features))

    # Save to CSV
    df = pd.DataFrame([feature_dict])
    df.to_csv(csv_path, index=False)

    logger.info(f"Converted .mat to CSV: {csv_path.name}")


class APICPipeline:
    """APIC Biomarker Pipeline - Steps 3-8."""

    # Default configuration
    DEFAULT_CONFIG = {
        'pipeline': {
            'resume_on_existing': True
        },
        'nuclei_segmentation': {
            'model_path': 'models/hovernet_fast_pannuke_type_tf2pytorch.tar',
            'mode': 'fast',
            'batch_size': 6,
            'type_info_path': 'types/type_info_pannuke.json'
        },
        'spatil': {
            'alpha': [0.56, 0.56],
            'r': 0.07,
            'num_processes': 12,
            'draw_option': 1,
            'num_viz_patches': 10
        },
        'nuclear_diversity': {
            'use_matlab': True
        },
        'cox_model': {
            'training_data_path': 'data/train_data.csv'
        }
    }

    def __init__(self, input_slide: str, output_dir: str, config: dict = None):
        """
        Initialize pipeline.

        Args:
            input_slide: Path to input slide file
            output_dir: Root output directory
            config: Optional configuration override
        """
        self.input_slide = Path(input_slide).resolve()
        self.slide_name = self.input_slide.stem
        self.slide_name_ext = self.input_slide.name
        self.output_dir = Path(output_dir).resolve()
        self.config = config or self.DEFAULT_CONFIG

        # Setup directories
        self.slide_output_dir = self.output_dir / self.slide_name
        self.dirs = {
            'qc': self.slide_output_dir / "qc",
            'patches': self.slide_output_dir / "patches",
            'nuclei': self.slide_output_dir / "nuclei_segmentation",
            'spatil': self.slide_output_dir / "spatil_features",
            'spatil_viz': self.slide_output_dir / "spatil_visualizations",
            'nucdiv': self.slide_output_dir / "nucdiv_features",
            'final': self.slide_output_dir / "final_features",
            'report': self.slide_output_dir / "report",
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pipeline initialized for: {self.slide_name}")

    def _should_skip(self, path: Path) -> bool:
        """Check if step should be skipped (output exists)."""
        if not self.config['pipeline']['resume_on_existing']:
            return False
        return path.exists() if path else False

    def step3_nuclei_segmentation(self):
        """Run HoVerNet nuclei segmentation on patches."""
        log_step_header("Nuclei Segmentation", 3)
        start = time.time()

        patches_dir = self.dirs['patches'] / self.slide_name / "tiles"
        output_dir = self.dirs['nuclei'] / self.slide_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not patches_dir.exists():
            raise FileNotFoundError(f"Patches not found: {patches_dir}. Run steps 1-2 first.")

        # Check existing progress
        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in output_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        if not unprocessed:
            logger.info(f"Already complete: {len(patch_files)} patches processed")
            return

        logger.info(f"Processing {len(unprocessed)}/{len(patch_files)} patches")

        # Create temp directory with symlinks to unprocessed patches
        # Clean any existing temp directory to remove stale symlinks from previous runs
        temp_dir = self.slide_output_dir / "temp_unprocessed_patches"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for patch in unprocessed:
            link = temp_dir / patch.name
            if not link.exists():
                link.symlink_to(patch.resolve())

        # Run HoVerNet
        cfg = self.config['nuclei_segmentation']
        cmd = [
            sys.executable, "src/nucleusSegmentationTiles.py",
            str(temp_dir), '.jpeg',
            cfg['model_path'], cfg['mode'],
            str(cfg['batch_size']), cfg['type_info_path'],
            str(output_dir)
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        run_subprocess(cmd, "HoVerNet nuclei segmentation", env=env)

        # Cleanup
        shutil.rmtree(temp_dir)

        final_count = len(list(output_dir.glob("*.png")))
        log_step_complete("Nuclei segmentation", time.time() - start)
        logger.info(f"Output: {final_count} segmentation masks")

    def step4_spatil_features(self):
        """Extract spaTIL spatial features from nuclei masks."""
        log_step_header("spaTIL Feature Extraction", 4)
        start = time.time()

        output_dir = self.dirs['spatil'] / self.slide_name

        if self._should_skip(output_dir) and any(output_dir.glob("*.csv")):
            logger.info("Already complete: spaTIL features exist")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config['spatil']
        cmd = [
            sys.executable, "src/spatil/spatil_main_adapted.py",
            "--patches_dir", str(self.dirs['patches']),
            "--nuclei_masks_dir", str(self.dirs['nuclei']),
            "--results_features_dir", str(self.dirs['spatil']),
            "--viz_output_dir", str(self.dirs['spatil_viz']),
            "--slide_name", self.slide_name,
            "--alpha", str(cfg['alpha'][0]), str(cfg['alpha'][1]),
            "--r", str(cfg['r']),
            "--draw_option", str(cfg['draw_option']),
            "--num_processes", str(cfg['num_processes']),
            "--num_viz_patches", str(cfg['num_viz_patches'])
        ]

        run_subprocess(cmd, "spaTIL feature extraction")

        # Count outputs
        feature_count = len(list(output_dir.glob("*.csv")))
        viz_count = len(list((self.dirs['spatil_viz'] / self.slide_name).glob("*.png"))) if (self.dirs['spatil_viz'] / self.slide_name).exists() else 0

        log_step_complete("spaTIL features", time.time() - start)
        logger.info(f"Output: {feature_count} feature files, {viz_count} visualizations")

    def step5_nucdiv_features(self):
        """Extract nuclear diversity features using MATLAB or Python."""
        log_step_header("Nuclear Diversity Features", 5)
        start = time.time()

        use_matlab = self.config['nuclear_diversity'].get('use_matlab', True)
        impl = "MATLAB" if use_matlab else "Python"
        logger.info(f"Implementation: {impl}")

        # Output paths
        csv_output = self.dirs['nucdiv'] / f"{self.slide_name}_nuclear_diversity_features.csv"

        if use_matlab:
            mat_output = self.dirs['nucdiv'] / self.slide_name / "slide1" / "nucleiDiversityFeatures_shape.mat"
        else:
            mat_output = None

        # Skip if already done
        if self._should_skip(csv_output):
            logger.info(f"Already complete: {csv_output.name} exists")
            return

        temp_dir = None

        try:
            if use_matlab:
                # Setup temp structure for MATLAB: <input>/<slide_name>/slide1/*.png
                temp_dir = self.slide_output_dir / "temp_matlab_input"
                temp_slide_dir = temp_dir / self.slide_name / "slide1"
                temp_slide_dir.mkdir(parents=True, exist_ok=True)

                # Symlink mask files
                source_dir = self.dirs['nuclei'] / self.slide_name
                mask_count = 0
                for mask in source_dir.glob("*.png"):
                    link = temp_slide_dir / mask.name
                    if not link.exists():
                        link.symlink_to(mask.resolve())
                    mask_count += 1

                logger.info(f"Linked {mask_count} masks to temp structure")

                # Run MATLAB executable
                executable = Path("src/nucdiv_executable_pipeline")
                if not executable.exists():
                    raise FileNotFoundError(f"MATLAB executable not found: {executable}")

                cmd = [str(executable.resolve()), str(temp_dir.resolve()), str(self.dirs['nucdiv'].resolve())]

                run_subprocess(cmd, "MATLAB nuclear diversity extraction")

                # Verify and convert output
                if not mat_output.exists():
                    raise FileNotFoundError(f"MATLAB output not created: {mat_output}")

                convert_mat_to_csv(mat_output, csv_output, self.slide_name)

            else:
                # Python implementation
                cmd = [
                    sys.executable, "src/nuclear_diversity.py",
                    "--nuclei_seg_dir", str(self.dirs['nuclei']),
                    "--slide_name", self.slide_name,
                    "--output_dir", str(self.dirs['nucdiv'])
                ]

                run_subprocess(cmd, "Python nuclear diversity extraction")

        finally:
            # Always cleanup temp directory
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

        log_step_complete("Nuclear diversity features", time.time() - start)
        logger.info(f"Output: {csv_output.name}")

    def step6_aggregate_features(self):
        """Aggregate patch-level features to slide-level."""
        log_step_header("Feature Aggregation", 6)
        start = time.time()

        spatil_output = self.dirs['final'] / f"{self.slide_name}_spatil_aggregated.csv"
        nucdiv_output = self.dirs['final'] / f"{self.slide_name}_nucdiv.csv"

        # Check if both outputs already exist
        spatil_exists = self._should_skip(spatil_output)
        nucdiv_exists = self._should_skip(nucdiv_output)

        if spatil_exists and nucdiv_exists:
            logger.info("Already complete: aggregated features exist")
            return

        # Aggregate spaTIL features (if not already done)
        if not spatil_exists:
            spatil_dir = self.dirs['spatil'] / self.slide_name
            if spatil_dir.exists():
                all_features = []
                for csv_file in spatil_dir.glob("*.csv"):
                    df = pd.read_csv(csv_file, header=None)
                    features = df.values.flatten()
                    if len(features) == 350:
                        all_features.append(features)

                if all_features:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        mean_features = np.nanmean(np.vstack(all_features), axis=0)

                    feature_cols = [f'feature_{i}' for i in range(350)]
                    pd.DataFrame([mean_features], columns=feature_cols).to_csv(spatil_output, index=False)
                    logger.info(f"Aggregated {len(all_features)} spaTIL patches → {spatil_output.name}")
                else:
                    logger.warning("No valid spaTIL feature files found")
        else:
            logger.info(f"spaTIL aggregation already exists: {spatil_output.name}")

        # Copy nucdiv features (if not already done)
        if not nucdiv_exists:
            nucdiv_source = self.dirs['nucdiv'] / f"{self.slide_name}_nuclear_diversity_features.csv"
            if nucdiv_source.exists():
                shutil.copy2(nucdiv_source, nucdiv_output)
                logger.info(f"Copied nuclear diversity → {nucdiv_output.name}")
            else:
                logger.warning(f"Nuclear diversity source not found: {nucdiv_source}")
        else:
            logger.info(f"NucDiv aggregation already exists: {nucdiv_output.name}")

        log_step_complete("Feature aggregation", time.time() - start)

    def step7_predict_biomarker(self):
        """Run Cox model prediction."""
        log_step_header("Biomarker Prediction", 7)
        start = time.time()

        prediction_output = self.dirs['final'] / f"{self.slide_name}_prediction.csv"

        if self._should_skip(prediction_output):
            logger.info("Already complete: prediction exists")
            return

        # Load features
        spatil_file = self.dirs['final'] / f"{self.slide_name}_spatil_aggregated.csv"
        nucdiv_file = self.dirs['final'] / f"{self.slide_name}_nucdiv.csv"

        if not spatil_file.exists() or not nucdiv_file.exists():
            raise FileNotFoundError("Required feature files missing. Run steps 4-6 first.")

        spatil_features = pd.read_csv(spatil_file)
        nucdiv_features = pd.read_csv(nucdiv_file)

        # Select features for Cox model
        patient_features = pd.DataFrame({
            'patient_id': [self.slide_name],
            'Area.Energy.var': [nucdiv_features['Area.Energy.var'].values[0]],
            'Area.InvDiffMom.Skewness': [nucdiv_features['Area.InvDiffMom.Skewness'].values[0]],
            'MinorAxisLength.Energy.Prcnt90': [nucdiv_features['MinorAxisLength.Energy.Prcnt90'].values[0]],
            'Area.DiffAvg.Prcnt10': [nucdiv_features['Area.DiffAvg.Prcnt10'].values[0]],
            'X341': [spatil_features['feature_342'].values[0]],
            'X51': [spatil_features['feature_52'].values[0]]
        })

        # Save temp features and run prediction
        temp_features = self.slide_output_dir / "temp_features.csv"
        patient_features.to_csv(temp_features, index=False)

        cmd = [
            sys.executable, "src/predict_biomarker.py",
            "--features", str(temp_features),
            "--training_data", self.config['cox_model']['training_data_path'],
            "--output", str(prediction_output)
        ]

        run_subprocess(cmd, "Cox model prediction")

        # Cleanup and report
        temp_features.unlink()

        prediction = pd.read_csv(prediction_output)
        risk_group = prediction['risk_group'].values[0]
        risk_score = prediction['risk_score'].values[0]

        log_step_complete("Biomarker prediction", time.time() - start)
        logger.info(f"Result: {risk_group} (risk score: {risk_score:.3f})")

    def step8_create_overlay(self):
        """Create tissue overlay visualization."""
        log_step_header("Tissue Overlay", 8)
        start = time.time()

        overlay_output = self.dirs['qc'] / f"{self.slide_name}_tissue_overlay.png"

        if self._should_skip(overlay_output):
            logger.info("Already complete: overlay exists")
            return

        import cv2

        thumbnail_path = self.dirs['patches'] / self.slide_name / "thumbnail.jpeg"
        mask_path = self.dirs['qc'] / self.slide_name_ext / f"{self.slide_name_ext}_mask_use.png"

        if not thumbnail_path.exists():
            logger.warning(f"Thumbnail not found: {thumbnail_path}")
            return

        if not mask_path.exists():
            logger.warning(f"Tissue mask not found: {mask_path}")
            return

        # Load and process
        thumbnail = cv2.imread(str(thumbnail_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if thumbnail is None or mask is None:
            logger.error("Failed to load thumbnail or mask")
            return

        # Resize and threshold
        mask_resized = cv2.resize(mask, (thumbnail.shape[1], thumbnail.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

        # Find and filter contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_area = (thumbnail.shape[0] * thumbnail.shape[1]) * 0.01
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # Draw overlay
        result = thumbnail.copy()
        cv2.drawContours(result, large_contours, -1, (0, 255, 0), 4)

        # Save with high quality (PNG compression level 0-9, lower = less compression = better quality)
        cv2.imwrite(str(overlay_output), result, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        log_step_complete("Tissue overlay", time.time() - start)
        logger.info(f"Output: {overlay_output.name}")

    def step8_build_report(self):
        """Generate APIC PDF report."""
        log_step_header("APIC Report Generation", 8)
        start = time.time()

        report_output = self.slide_output_dir / f"{self.slide_name}_APIC_report.pdf"

        if self._should_skip(report_output):
            logger.info("Already complete: report exists")
            return

        cmd = [
            sys.executable, "src/build_apic_report_final.py",
            "--results-root", str(self.output_dir),
            "--page1-template-pos", "data/HUG-REPORT-EMPTY-POSITIVE.pdf",
            "--page1-template-neg", "data/HUG-REPORT-EMPTY-NEGATIVE.pdf",
            "--page2-template", "data/HUG_report.pdf",
            "--out-dir", str(self.dirs['report']),
            "--out-dir", str(self.output_dir / "reports"),
            "--patient", self.slide_name,
        ]

        run_subprocess(cmd, "APIC report builder")

        log_step_complete("APIC report", time.time() - start)

    def run(self, steps: list = None):
        """
        Run pipeline steps.

        Args:
            steps: List of step names to run. None runs all steps.
        """
        all_steps = ['nuclei', 'spatil', 'nucdiv', 'aggregate', 'predict', 'overlay', 'report']
        steps = steps or all_steps

        step_map = {
            'nuclei': self.step3_nuclei_segmentation,
            'spatil': self.step4_spatil_features,
            'nucdiv': self.step5_nucdiv_features,
            'aggregate': self.step6_aggregate_features,
            'predict': self.step7_predict_biomarker,
            'overlay': self.step8_create_overlay,
            'report': self.step8_build_report
        }

        logger.info("=" * 60)
        logger.info(f"APIC PIPELINE - {self.slide_name}")
        logger.info(f"Steps: {', '.join(steps)}")
        logger.info("=" * 60)

        pipeline_start = time.time()

        for step in steps:
            if step not in step_map:
                logger.warning(f"Unknown step: {step}")
                continue

            try:
                step_map[step]()
            except Exception as e:
                logger.error(f"Step '{step}' failed: {e}")
                raise

        # Final summary
        self._print_summary(time.time() - pipeline_start)

    def _print_summary(self, total_time: float):
        """Print pipeline completion summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Slide: {self.slide_name}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info("")

        # Check outputs
        outputs = [
            (self.dirs['nuclei'] / self.slide_name, "Nuclei masks"),
            (self.dirs['spatil'] / self.slide_name, "spaTIL features"),
            (self.dirs['nucdiv'] / f"{self.slide_name}_nuclear_diversity_features.csv", "NucDiv features"),
            (self.dirs['final'] / f"{self.slide_name}_prediction.csv", "Prediction"),
            (self.dirs['qc'] / f"{self.slide_name}_tissue_overlay.png", "Overlay"),
        ]

        logger.info("Outputs:")
        for path, name in outputs:
            if path.exists():
                if path.is_dir():
                    count = len(list(path.glob("*")))
                    logger.info(f"  ✓ {name}: {count} files")
                else:
                    logger.info(f"  ✓ {name}")
            else:
                logger.info(f"  ✗ {name}: missing")

        logger.info("=" * 60)


class APICPatientPipeline:
    """APIC Multi-Slide Patient Pipeline - Aggregates features across slides."""

    def __init__(self, patient_id: str, patient_folder: str, output_dir: str, config: dict = None):
        """
        Initialize patient-level pipeline.

        Args:
            patient_id: Patient identifier (typically folder name)
            patient_folder: Path to folder containing multiple slides
            output_dir: Root output directory
            config: Optional configuration override
        """
        self.patient_id = patient_id
        self.patient_folder = Path(patient_folder).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config = config or APICPipeline.DEFAULT_CONFIG

        # Patient-level output directory
        self.patient_output_dir = self.output_dir / self.patient_id
        self.patient_output_dir.mkdir(parents=True, exist_ok=True)

        # Directories for patient-level outputs
        self.dirs = {
            'final': self.patient_output_dir / "final_features",
            'qc': self.patient_output_dir / "qc",
            'report': self.patient_output_dir / "report",
            'spatil_viz': self.patient_output_dir / "spatil_visualizations" / self.patient_id,
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Find all processed slide directories
        self.slide_dirs = self._find_slide_directories()

        logger.info(f"Patient pipeline initialized for: {self.patient_id}")
        logger.info(f"Found {len(self.slide_dirs)} processed slides")

    def _find_slide_directories(self) -> list:
        """Find all slide output directories for this patient's slides."""
        slide_dirs = []
        supported_exts = ['svs', 'tif', 'tiff', 'ndpi', 'mrxs', 'scn']

        for ext in supported_exts:
            for slide_path in self.patient_folder.glob(f"*.{ext}"):
                slide_name = slide_path.stem
                slide_output = self.output_dir / slide_name
                if slide_output.exists() and (slide_output / "final_features").exists():
                    slide_dirs.append(slide_output)

        return sorted(slide_dirs)

    def aggregate_patient_features(self):
        """Aggregate slide-level features to patient level using np.nanmean."""
        log_step_header("Patient-Level Feature Aggregation", "P1")
        start = time.time()

        spatil_output = self.dirs['final'] / f"{self.patient_id}_spatil_aggregated.csv"
        nucdiv_output = self.dirs['final'] / f"{self.patient_id}_nucdiv.csv"

        # Resume check: skip if both outputs exist
        if spatil_output.exists() and nucdiv_output.exists():
            logger.info("Already complete: aggregated features exist")
            return

        # Collect spaTIL features from all slides
        all_spatil = []
        spatil_columns = None
        for slide_dir in self.slide_dirs:
            spatil_file = slide_dir / "final_features" / f"{slide_dir.name}_spatil_aggregated.csv"
            if spatil_file.exists():
                df = pd.read_csv(spatil_file)
                spatil_columns = df.columns
                all_spatil.append(df.values[0])
                logger.info(f"  Loaded spaTIL from: {slide_dir.name}")

        if all_spatil:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                mean_spatil = np.nanmean(np.vstack(all_spatil), axis=0)

            pd.DataFrame([mean_spatil], columns=spatil_columns).to_csv(spatil_output, index=False)
            logger.info(f"  Aggregated {len(all_spatil)} slides -> {spatil_output.name}")
        else:
            logger.warning("No spaTIL features found to aggregate")

        # Collect nucdiv features from all slides
        all_nucdiv = []
        nucdiv_columns = None
        for slide_dir in self.slide_dirs:
            nucdiv_file = slide_dir / "final_features" / f"{slide_dir.name}_nucdiv.csv"
            if nucdiv_file.exists():
                df = pd.read_csv(nucdiv_file)
                # Exclude slide_id column (first column is string identifier)
                numeric_df = df.select_dtypes(include=[np.number])
                nucdiv_columns = numeric_df.columns
                all_nucdiv.append(numeric_df.values[0])
                logger.info(f"  Loaded nucdiv from: {slide_dir.name}")

        if all_nucdiv:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                mean_nucdiv = np.nanmean(np.vstack(all_nucdiv), axis=0)

            pd.DataFrame([mean_nucdiv], columns=nucdiv_columns).to_csv(nucdiv_output, index=False)
            logger.info(f"  Aggregated {len(all_nucdiv)} slides -> {nucdiv_output.name}")
        else:
            logger.warning("No nucdiv features found to aggregate")

        log_step_complete("Patient feature aggregation", time.time() - start)

    def predict_biomarker(self):
        """Run Cox model prediction on patient-averaged features."""
        log_step_header("Patient Biomarker Prediction", "P2")
        start = time.time()

        prediction_output = self.dirs['final'] / f"{self.patient_id}_prediction.csv"

        # Resume check: skip if prediction exists
        if prediction_output.exists():
            logger.info("Already complete: prediction exists")
            return

        # Load patient-level features
        spatil_file = self.dirs['final'] / f"{self.patient_id}_spatil_aggregated.csv"
        nucdiv_file = self.dirs['final'] / f"{self.patient_id}_nucdiv.csv"

        if not spatil_file.exists() or not nucdiv_file.exists():
            raise FileNotFoundError("Patient-level feature files missing.")

        spatil_features = pd.read_csv(spatil_file)
        nucdiv_features = pd.read_csv(nucdiv_file)

        # Select features for Cox model
        patient_features = pd.DataFrame({
            'patient_id': [self.patient_id],
            'Area.Energy.var': [nucdiv_features['Area.Energy.var'].values[0]],
            'Area.InvDiffMom.Skewness': [nucdiv_features['Area.InvDiffMom.Skewness'].values[0]],
            'MinorAxisLength.Energy.Prcnt90': [nucdiv_features['MinorAxisLength.Energy.Prcnt90'].values[0]],
            'Area.DiffAvg.Prcnt10': [nucdiv_features['Area.DiffAvg.Prcnt10'].values[0]],
            'X341': [spatil_features['feature_342'].values[0]],
            'X51': [spatil_features['feature_52'].values[0]]
        })

        temp_features = self.patient_output_dir / "temp_features.csv"
        patient_features.to_csv(temp_features, index=False)

        cmd = [
            sys.executable, "src/predict_biomarker.py",
            "--features", str(temp_features),
            "--training_data", APICPipeline.DEFAULT_CONFIG['cox_model']['training_data_path'],
            "--output", str(prediction_output)
        ]

        run_subprocess(cmd, "Cox model prediction")
        temp_features.unlink()

        prediction = pd.read_csv(prediction_output)
        risk_group = prediction['risk_group'].values[0]
        risk_score = prediction['risk_score'].values[0]

        log_step_complete("Patient biomarker prediction", time.time() - start)
        logger.info(f"Result: {risk_group} (risk score: {risk_score:.3f})")

    def select_representative_overlay(self):
        """Select overlay from first available slide."""
        log_step_header("Patient Tissue Overlay", "P3")
        start = time.time()

        overlay_output = self.dirs['qc'] / f"{self.patient_id}_tissue_overlay.png"

        if overlay_output.exists():
            logger.info("Already complete: overlay exists")
            return

        # Find first available overlay from any slide
        for slide_dir in self.slide_dirs:
            source_overlay = slide_dir / "qc" / f"{slide_dir.name}_tissue_overlay.png"
            if source_overlay.exists():
                shutil.copy2(source_overlay, overlay_output)
                logger.info(f"  Selected overlay from: {slide_dir.name}")
                log_step_complete("Patient overlay", time.time() - start)
                return

        logger.warning("No overlay found in any slide")
        log_step_complete("Patient overlay", time.time() - start)

    def copy_visualizations(self):
        """Copy spaTIL visualizations from slides to patient directory."""
        log_step_header("Patient Visualizations", "P4")

        # Resume check: skip if visualizations already exist
        existing_viz = list(self.dirs['spatil_viz'].glob("viz_*.png"))
        if len(existing_viz) >= 4:
            logger.info("Already complete: visualizations exist")
            return

        # Collect visualizations from all slides, pick representative ones
        all_viz = []
        for slide_dir in self.slide_dirs:
            viz_dir = slide_dir / "spatil_visualizations" / slide_dir.name
            if viz_dir.exists():
                all_viz.extend(list(viz_dir.glob("*.png")))

        # Copy first 4 visualizations to patient viz directory
        for i, viz_path in enumerate(all_viz[:4]):
            dest = self.dirs['spatil_viz'] / f"viz_{i:02d}.png"
            shutil.copy2(viz_path, dest)
            logger.info(f"  Copied: {viz_path.name} -> {dest.name}")

        if not all_viz:
            logger.warning("No visualizations found to copy")

    def build_report(self):
        """Generate APIC PDF report for patient."""
        log_step_header("Patient APIC Report Generation", "P5")
        start = time.time()

        # Resume check: skip if report exists
        report_path = self.dirs['report'] / f"{self.patient_id}_report.pdf"
        if report_path.exists():
            logger.info("Already complete: report exists")
            return

        cmd = [
            sys.executable, "src/build_apic_report_final.py",
            "--results-root", str(self.output_dir),
            "--page1-template-pos", "data/HUG-REPORT-EMPTY-POSITIVE.pdf",
            "--page1-template-neg", "data/HUG-REPORT-EMPTY-NEGATIVE.pdf",
            "--page2-template", "data/HUG_report.pdf",
            "--out-dir", str(self.output_dir / "reports"),
            "--patient", self.patient_id,
        ]

        run_subprocess(cmd, "APIC report builder")

        log_step_complete("Patient APIC report", time.time() - start)

    def run(self):
        """Run patient-level aggregation pipeline."""
        logger.info("=" * 60)
        logger.info(f"PATIENT PIPELINE - {self.patient_id}")
        logger.info(f"Slides: {len(self.slide_dirs)}")
        logger.info("=" * 60)

        pipeline_start = time.time()

        self.aggregate_patient_features()
        self.predict_biomarker()
        self.select_representative_overlay()
        self.copy_visualizations()
        self.build_report()

        logger.info("")
        logger.info("=" * 60)
        logger.info("PATIENT PIPELINE COMPLETE")
        logger.info(f"Patient ID: {self.patient_id}")
        logger.info(f"Total time: {(time.time() - pipeline_start)/60:.1f} minutes")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="APIC Pipeline Steps 3-8: Feature Extraction and Biomarker Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  nuclei     - HoVerNet nuclei segmentation
  spatil     - spaTIL spatial feature extraction
  nucdiv     - Nuclear diversity features (MATLAB)
  aggregate  - Aggregate patch features to slide level
  predict    - Cox model biomarker prediction
  overlay    - Create tissue overlay visualization
  report     - Generate APIC PDF report

Patient Mode (multi-slide):
  --patient-aggregate  - Run patient-level aggregation across multiple slides
  --patient-id         - Patient identifier (folder name)
  --patient-folder     - Folder containing patient's slides

Example:
  python feature_extraction_prediction.py -i /data/slide.ndpi -o /data/output
  python feature_extraction_prediction.py -i /data/slide.ndpi -o /data/output --steps nuclei spatil
  python feature_extraction_prediction.py --patient-aggregate --patient-id PT001 --patient-folder /data/PT001 -o /data/output
        """
    )
    parser.add_argument("-i", "--input", help="Input slide file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--steps", nargs='+', help="Steps to run (default: all)")

    # Patient mode arguments
    parser.add_argument("--patient-aggregate", action="store_true",
                       help="Run patient-level aggregation (multi-slide mode)")
    parser.add_argument("--patient-id", help="Patient identifier for multi-slide mode")
    parser.add_argument("--patient-folder", help="Folder containing patient's slides")

    args = parser.parse_args()

    # Patient aggregation mode
    if args.patient_aggregate:
        if not args.patient_id or not args.patient_folder:
            logger.error("Patient mode requires --patient-id and --patient-folder")
            sys.exit(1)

        try:
            pipeline = APICPatientPipeline(
                args.patient_id,
                args.patient_folder,
                args.output
            )
            pipeline.run()
        except Exception as e:
            logger.error(f"Patient pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Standard single-slide mode
        if not args.input:
            logger.error("Single-slide mode requires -i/--input")
            sys.exit(1)

        if not os.path.exists(args.input):
            logger.error(f"Input not found: {args.input}")
            sys.exit(1)

        try:
            pipeline = APICPipeline(args.input, args.output)
            pipeline.run(args.steps)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
