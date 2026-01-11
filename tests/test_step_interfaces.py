import pytest
import numpy as np
import pandas as pd
import re
from pathlib import Path
from PIL import Image


class TestStep1Output:

    def test_qc_mask_exists(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        mask_pattern = f"{slide_name}*_mask_use.png"
        masks = list(dirs['qc'].glob(mask_pattern))

        assert len(masks) >= 1

    def test_mask_is_grayscale(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        mask_files = list(dirs['qc'].glob("*_mask_use.png"))

        if mask_files:
            mask = Image.open(mask_files[0])
            assert mask.mode == 'L'


class TestStep2Output:

    def test_patches_directory_exists(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        assert dirs['patches'].exists()

    def test_patches_are_jpeg(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        patches = list(dirs['patches'].glob("*.jpeg"))
        assert len(patches) > 0

    def test_patch_naming_convention(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        pattern = re.compile(rf"{slide_name}_x\d+_y\d+_l\d+\.jpeg")
        patches = list(dirs['patches'].glob("*.jpeg"))

        for patch in patches:
            assert pattern.match(patch.name)

    def test_thumbnail_exists(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        slide_dir = complete_slide_output['slide_dir']

        thumbnail = slide_dir / "patches" / slide_name / "thumbnail.jpeg"
        assert thumbnail.exists()


class TestStep3Input:

    def test_reads_patches_from_step2(self, complete_slide_output):
        dirs = complete_slide_output['dirs']

        patches = list(dirs['patches'].glob("*.jpeg"))
        assert len(patches) > 0

        for patch in patches:
            img = Image.open(patch)
            assert img.mode == 'RGB'
            img.close()


class TestStep3Output:

    def test_nuclei_masks_exist(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        masks = list(dirs['nuclei'].glob("*.png"))
        assert len(masks) > 0

    def test_one_mask_per_patch(self, complete_slide_output):
        dirs = complete_slide_output['dirs']

        patch_stems = {p.stem for p in dirs['patches'].glob("*.jpeg")}
        mask_stems = {m.stem for m in dirs['nuclei'].glob("*.png")}

        assert patch_stems == mask_stems

    def test_mask_is_grayscale(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        masks = list(dirs['nuclei'].glob("*.png"))

        for mask_path in masks:
            mask = Image.open(mask_path)
            assert mask.mode == 'L'
            mask.close()


class TestStep4Input:

    def test_reads_patches_and_masks(self, complete_slide_output):
        dirs = complete_slide_output['dirs']

        patches = list(dirs['patches'].glob("*.jpeg"))
        masks = list(dirs['nuclei'].glob("*.png"))

        patch_stems = {p.stem for p in patches}
        mask_stems = {m.stem for m in masks}

        aligned = patch_stems & mask_stems
        assert len(aligned) > 0


class TestStep4Output:

    def test_spatil_csvs_exist(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        csvs = list(dirs['spatil'].glob("*.csv"))
        assert len(csvs) > 0

    def test_csv_has_350_columns(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        csvs = list(dirs['spatil'].glob("*.csv"))

        for csv_path in csvs:
            df = pd.read_csv(csv_path, header=None)
            assert df.shape[1] == 350

    def test_values_are_numeric(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        csvs = list(dirs['spatil'].glob("*.csv"))

        for csv_path in csvs:
            df = pd.read_csv(csv_path, header=None)
            assert df.select_dtypes(include=[np.number]).shape == df.shape


class TestStep5Input:

    def test_reads_nuclei_masks(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        masks = list(dirs['nuclei'].glob("*.png"))
        assert len(masks) > 0


class TestStep5Output:

    def test_nucdiv_csv_exists(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        csv_path = dirs['nucdiv'] / f"{slide_name}_nuclear_diversity_features.csv"
        assert csv_path.exists()

    def test_has_slide_id_column(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        df = pd.read_csv(dirs['nucdiv'] / f"{slide_name}_nuclear_diversity_features.csv")
        assert 'slide_id' in df.columns or 'feat_0' in df.columns


class TestStep6Input:

    def test_reads_spatil_csvs(self, complete_slide_output):
        dirs = complete_slide_output['dirs']
        csvs = list(dirs['spatil'].glob("*.csv"))
        assert len(csvs) > 0

    def test_reads_nucdiv_csv(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        csv_path = dirs['nucdiv'] / f"{slide_name}_nuclear_diversity_features.csv"
        assert csv_path.exists()


class TestStep6Output:

    def test_aggregated_spatil_exists(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        csv_path = dirs['final'] / f"{slide_name}_spatil_aggregated.csv"
        assert csv_path.exists()

    def test_aggregated_has_350_columns(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        df = pd.read_csv(dirs['final'] / f"{slide_name}_spatil_aggregated.csv")
        assert len(df.columns) == 350

    def test_nucdiv_copied(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        csv_path = dirs['final'] / f"{slide_name}_nucdiv.csv"
        assert csv_path.exists()


class TestStep7Input:

    def test_reads_final_features(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        spatil = dirs['final'] / f"{slide_name}_spatil_aggregated.csv"
        nucdiv = dirs['final'] / f"{slide_name}_nucdiv.csv"

        assert spatil.exists()
        assert nucdiv.exists()


class TestStep7Output:

    def test_prediction_csv_exists(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        csv_path = dirs['final'] / f"{slide_name}_prediction.csv"
        assert csv_path.exists()

    def test_has_required_columns(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        df = pd.read_csv(dirs['final'] / f"{slide_name}_prediction.csv")

        assert 'risk_score' in df.columns
        assert 'risk_group' in df.columns
        assert 'threshold' in df.columns

    def test_risk_group_valid(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        df = pd.read_csv(dirs['final'] / f"{slide_name}_prediction.csv")

        assert df['risk_group'].values[0] in ['POSITIVE', 'NEGATIVE']


class TestStep8Input:

    def test_reads_prediction(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        csv_path = dirs['final'] / f"{slide_name}_prediction.csv"
        df = pd.read_csv(csv_path)

        assert 'risk_score' in df.columns


class TestStep8Output:

    def test_overlay_exists(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        slide_dir = complete_slide_output['slide_dir']

        overlay = slide_dir / "qc" / f"{slide_name}_tissue_overlay.png"
        assert overlay.exists()

    def test_overlay_is_rgb(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        slide_dir = complete_slide_output['slide_dir']

        overlay_path = slide_dir / "qc" / f"{slide_name}_tissue_overlay.png"
        if overlay_path.exists():
            img = Image.open(overlay_path)
            assert img.mode == 'RGB'
            img.close()
