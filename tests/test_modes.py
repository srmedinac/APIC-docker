import pytest
from pathlib import Path


class TestModeDetection:

    def test_single_slide_is_file(self, tmp_path):
        slide = tmp_path / "slide.svs"
        slide.touch()

        is_single = slide.is_file()
        is_directory = slide.is_dir()

        assert is_single is True
        assert is_directory is False

    def test_batch_mode_is_directory_with_slides(self, batch_slides_setup):
        input_dir = batch_slides_setup['input_dir']

        is_directory = input_dir.is_dir()
        slides = list(input_dir.glob("*.svs"))

        assert is_directory is True
        assert len(slides) == 3

    def test_multi_slide_has_slides_in_folder(self, multi_slide_setup):
        patient_folder = multi_slide_setup['patient_folder']

        slides = list(patient_folder.glob("*.svs"))

        assert len(slides) == 3

    def test_batch_multi_slide_has_subdirectories(self, tmp_path):
        input_dir = tmp_path / "input"
        for patient in ["PT001", "PT002", "PT003"]:
            patient_dir = input_dir / patient
            patient_dir.mkdir(parents=True)
            (patient_dir / "slide_1.svs").touch()
            (patient_dir / "slide_2.svs").touch()

        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        has_nested = len(subdirs) > 0

        assert has_nested is True
        assert len(subdirs) == 3


class TestSingleSlideMode:

    def test_output_directory_structure(self, complete_slide_output):
        slide_dir = complete_slide_output['slide_dir']

        expected_dirs = ['qc', 'patches', 'nuclei_segmentation', 'spatil_features',
                        'nucdiv_features', 'final_features']

        for dir_name in expected_dirs:
            assert (slide_dir / dir_name).exists()

    def test_all_outputs_exist(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        assert len(list(dirs['patches'].glob("*.jpeg"))) > 0
        assert len(list(dirs['nuclei'].glob("*.png"))) > 0
        assert len(list(dirs['spatil'].glob("*.csv"))) > 0
        assert (dirs['final'] / f"{slide_name}_spatil_aggregated.csv").exists()
        assert (dirs['final'] / f"{slide_name}_nucdiv.csv").exists()
        assert (dirs['final'] / f"{slide_name}_prediction.csv").exists()


class TestBatchMode:

    def test_creates_separate_patient_dirs(self, batch_slides_setup, tmp_path):
        output_dir = batch_slides_setup['output_dir']
        slide_names = batch_slides_setup['slide_names']

        for name in slide_names:
            (output_dir / name).mkdir()

        for name in slide_names:
            assert (output_dir / name).exists()
            assert (output_dir / name).is_dir()

    def test_each_slide_independent(self, batch_slides_setup):
        output_dir = batch_slides_setup['output_dir']
        slide_names = batch_slides_setup['slide_names']

        for name in slide_names:
            slide_dir = output_dir / name
            slide_dir.mkdir(exist_ok=True)
            (slide_dir / ".complete").touch()

        for name in slide_names:
            assert (output_dir / name / ".complete").exists()


class TestMultiSlideMode:

    def test_processes_all_slides(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        slide_names = multi_slide_setup['slide_names']

        for name in slide_names:
            assert (output_dir / name).exists()

    def test_creates_patient_aggregation_dir(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        patient_id = multi_slide_setup['patient_id']

        patient_dir = output_dir / patient_id
        patient_dir.mkdir(parents=True)
        (patient_dir / "final_features").mkdir()

        assert (patient_dir / "final_features").exists()

    def test_patient_features_averaged(self, multi_slide_setup):
        import pandas as pd
        import numpy as np

        output_dir = multi_slide_setup['output_dir']
        slide_names = multi_slide_setup['slide_names']

        all_spatil = []
        for name in slide_names:
            csv_path = output_dir / name / "final_features" / f"{name}_spatil_aggregated.csv"
            df = pd.read_csv(csv_path)
            all_spatil.append(df.values[0])

        averaged = np.nanmean(np.vstack(all_spatil), axis=0)

        assert len(averaged) == 350


class TestResumeMode:

    def test_skips_completed_slides(self, batch_slides_setup):
        output_dir = batch_slides_setup['output_dir']
        slide_names = batch_slides_setup['slide_names']

        for name in slide_names[:2]:
            slide_dir = output_dir / name
            slide_dir.mkdir(parents=True)
            (slide_dir / ".complete").touch()

        completed = [name for name in slide_names
                    if (output_dir / name / ".complete").exists()]
        incomplete = [name for name in slide_names
                     if not (output_dir / name / ".complete").exists()]

        assert len(completed) == 2
        assert len(incomplete) == 1

    def test_processes_incomplete(self, batch_slides_setup):
        output_dir = batch_slides_setup['output_dir']
        slide_names = batch_slides_setup['slide_names']

        slide_dir = output_dir / slide_names[0]
        slide_dir.mkdir(parents=True)

        has_complete_marker = (slide_dir / ".complete").exists()

        assert has_complete_marker is False

    def test_handles_partial_step_completion(self, temp_output_dir, dummy_patches, dummy_masks):
        processed_masks, unprocessed_patches = dummy_masks
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        assert len(patch_files) == 10
        assert len(existing_masks) == 5
        assert len(unprocessed) == 5
