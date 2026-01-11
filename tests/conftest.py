"""
Shared pytest fixtures for APIC pipeline tests.

These tests use temporary directories and dummy data to verify pipeline
behavior without requiring actual model inference or full slide processing.
"""

import os
import shutil
import pytest
import numpy as np
from pathlib import Path
from PIL import Image


@pytest.fixture
def slide_name():
    """Standard test slide name."""
    return "test_slide_001"


@pytest.fixture
def temp_output_dir(tmp_path, slide_name):
    """
    Create temporary output directory structure matching pipeline expectations.

    Structure:
        tmp_path/
        ├── test_slide_001/
        │   ├── patches/test_slide_001/tiles/
        │   ├── nuclei_segmentation/test_slide_001/
        │   ├── spatil_features/test_slide_001/
        │   ├── nucdiv_features/
        │   ├── final_features/
        │   └── qc/
    """
    slide_dir = tmp_path / slide_name

    dirs = {
        'patches': slide_dir / "patches" / slide_name / "tiles",
        'nuclei': slide_dir / "nuclei_segmentation" / slide_name,
        'spatil': slide_dir / "spatil_features" / slide_name,
        'nucdiv': slide_dir / "nucdiv_features",
        'final': slide_dir / "final_features",
        'qc': slide_dir / "qc",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return {
        'root': tmp_path,
        'slide_dir': slide_dir,
        **dirs
    }


@pytest.fixture
def dummy_patches(temp_output_dir, slide_name):
    """
    Create 10 small dummy JPEG patches (64x64 pixels).

    Returns list of patch file paths.
    """
    patches_dir = temp_output_dir['patches']
    patch_files = []

    for i in range(10):
        # Create a simple gradient image for variety
        img_array = np.zeros((64, 64, 3), dtype=np.uint8)
        img_array[:, :, 0] = i * 25  # R channel varies by patch
        img_array[:, :, 1] = 128     # G channel constant
        img_array[:, :, 2] = 255 - i * 25  # B channel inverse

        img = Image.fromarray(img_array, mode='RGB')
        patch_name = f"{slide_name}_x{i*1000}_y{i*500}_l0.jpeg"
        patch_path = patches_dir / patch_name
        img.save(patch_path, 'JPEG')
        patch_files.append(patch_path)

    return patch_files


@pytest.fixture
def dummy_masks(temp_output_dir, dummy_patches):
    """
    Create dummy PNG masks for some patches (simulating partial processing).

    Creates masks for first 5 patches only, leaving 5 "unprocessed".
    Returns tuple of (processed_masks, unprocessed_patches).
    """
    nuclei_dir = temp_output_dir['nuclei']
    processed_masks = []

    # Create masks for first 5 patches
    for patch_path in dummy_patches[:5]:
        mask_array = np.zeros((64, 64), dtype=np.uint8)
        # Add some fake nuclei regions
        mask_array[10:20, 10:20] = 1
        mask_array[30:40, 40:50] = 2

        mask = Image.fromarray(mask_array, mode='L')
        mask_name = patch_path.stem + ".png"
        mask_path = nuclei_dir / mask_name
        mask.save(mask_path, 'PNG')
        processed_masks.append(mask_path)

    unprocessed_patches = dummy_patches[5:]

    return processed_masks, unprocessed_patches


@pytest.fixture
def dummy_spatil_csvs(temp_output_dir, slide_name):
    """
    Create dummy spaTIL feature CSV files.

    Each CSV has 350 features (matching real pipeline output).
    Some features contain NaN to test nanmean aggregation.
    """
    import pandas as pd

    spatil_dir = temp_output_dir['spatil']
    csv_files = []

    for i in range(5):
        # Create 350 features with some NaN values
        features = np.random.rand(350)
        if i % 2 == 0:
            # Add some NaN values
            features[10:15] = np.nan

        df = pd.DataFrame([features])
        csv_path = spatil_dir / f"{slide_name}_patch_{i}.csv"
        df.to_csv(csv_path, index=False, header=False)
        csv_files.append(csv_path)

    return csv_files


@pytest.fixture
def dummy_nucdiv_csv(temp_output_dir, slide_name):
    """
    Create dummy nuclear diversity feature CSV file.

    Includes slide_id string column and 3264 numeric features.
    """
    import pandas as pd

    nucdiv_dir = temp_output_dir['nucdiv']

    # Create feature columns matching real output
    feature_cols = [f"feature_{i}" for i in range(100)]  # Simplified
    feature_values = np.random.rand(100)

    data = {'slide_id': [slide_name]}
    data.update(dict(zip(feature_cols, feature_values)))

    df = pd.DataFrame(data)
    csv_path = nucdiv_dir / f"{slide_name}_nuclear_diversity_features.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def stale_symlinks(temp_output_dir, slide_name):
    temp_dir = temp_output_dir['slide_dir'] / "temp_unprocessed_patches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    stale_links = []
    for i in range(5):
        fake_target = Path(f"/nonexistent/path/patch_{i}.jpeg")
        link_path = temp_dir / f"stale_patch_{i}.jpeg"
        link_path.symlink_to(fake_target)
        stale_links.append(link_path)

    return temp_dir, stale_links


@pytest.fixture
def multi_slide_setup(tmp_path):
    """Create a patient folder with multiple slides."""
    import pandas as pd

    patient_id = "PT_TEST_001"
    patient_folder = tmp_path / "input" / patient_id
    output_dir = tmp_path / "output"

    patient_folder.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    slide_names = ["slide_A", "slide_B", "slide_C"]

    for slide_name in slide_names:
        slide_dir = output_dir / slide_name
        (slide_dir / "final_features").mkdir(parents=True)
        (slide_dir / "qc").mkdir(parents=True)
        (slide_dir / "spatil_visualizations" / slide_name).mkdir(parents=True)

        spatil_data = np.random.rand(350)
        pd.DataFrame([spatil_data], columns=[f'feature_{i}' for i in range(350)]).to_csv(
            slide_dir / "final_features" / f"{slide_name}_spatil_aggregated.csv", index=False
        )

        nucdiv_data = {'slide_id': [slide_name]}
        nucdiv_data.update({f'feature_{i}': [np.random.rand()] for i in range(100)})
        pd.DataFrame(nucdiv_data).to_csv(
            slide_dir / "final_features" / f"{slide_name}_nucdiv.csv", index=False
        )

        mask_dir = slide_dir / "qc" / f"{slide_name}.svs"
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new('L', (100, 100), 255)
        mask.save(mask_dir / f"{slide_name}.svs_mask_use.png")

        overlay = Image.new('RGB', (100, 100), (0, 255, 0))
        overlay.save(slide_dir / "qc" / f"{slide_name}_tissue_overlay.png")

        viz = Image.new('RGB', (64, 64), (128, 128, 128))
        viz.save(slide_dir / "spatil_visualizations" / slide_name / "viz_0.png")

        (patient_folder / f"{slide_name}.svs").touch()

    return {
        'patient_id': patient_id,
        'patient_folder': patient_folder,
        'output_dir': output_dir,
        'slide_names': slide_names,
    }


@pytest.fixture
def batch_slides_setup(tmp_path):
    """Create a batch of independent slides."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    slide_names = ["patient_A", "patient_B", "patient_C"]

    for name in slide_names:
        (input_dir / f"{name}.svs").touch()

    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'slide_names': slide_names,
    }


@pytest.fixture
def complete_slide_output(tmp_path):
    """Create a complete slide output directory with all expected files."""
    import pandas as pd

    slide_name = "complete_slide"
    slide_dir = tmp_path / slide_name

    dirs = {
        'qc': slide_dir / "qc" / f"{slide_name}.svs",
        'patches': slide_dir / "patches" / slide_name / "tiles",
        'nuclei': slide_dir / "nuclei_segmentation" / slide_name,
        'spatil': slide_dir / "spatil_features" / slide_name,
        'nucdiv': slide_dir / "nucdiv_features",
        'final': slide_dir / "final_features",
        'report': slide_dir / "report",
    }

    for d in dirs.values():
        d.mkdir(parents=True)

    Image.new('L', (100, 100), 255).save(dirs['qc'] / f"{slide_name}.svs_mask_use.png")
    Image.new('RGB', (100, 100), (0, 255, 0)).save(slide_dir / "qc" / f"{slide_name}_tissue_overlay.png")

    Image.new('RGB', (256, 256), (128, 128, 128)).save(slide_dir / "patches" / slide_name / "thumbnail.jpeg")
    for i in range(5):
        Image.new('RGB', (64, 64), (i*50, 100, 200)).save(dirs['patches'] / f"{slide_name}_x{i*1000}_y0_l0.jpeg")

    for i in range(5):
        Image.new('L', (64, 64), 128).save(dirs['nuclei'] / f"{slide_name}_x{i*1000}_y0_l0.png")

    for i in range(5):
        pd.DataFrame([np.random.rand(350)]).to_csv(dirs['spatil'] / f"patch_{i}.csv", index=False, header=False)

    nucdiv_cols = ['slide_id'] + [f'feat_{i}' for i in range(100)]
    nucdiv_vals = [slide_name] + list(np.random.rand(100))
    pd.DataFrame([nucdiv_vals], columns=nucdiv_cols).to_csv(
        dirs['nucdiv'] / f"{slide_name}_nuclear_diversity_features.csv", index=False
    )

    pd.DataFrame([np.random.rand(350)], columns=[f'feature_{i}' for i in range(350)]).to_csv(
        dirs['final'] / f"{slide_name}_spatil_aggregated.csv", index=False
    )
    pd.DataFrame([nucdiv_vals], columns=nucdiv_cols).to_csv(
        dirs['final'] / f"{slide_name}_nucdiv.csv", index=False
    )
    pd.DataFrame([{'risk_score': 0.65, 'risk_group': 'POSITIVE', 'threshold': 0.5}]).to_csv(
        dirs['final'] / f"{slide_name}_prediction.csv", index=False
    )

    return {
        'slide_name': slide_name,
        'slide_dir': slide_dir,
        'dirs': dirs,
    }
