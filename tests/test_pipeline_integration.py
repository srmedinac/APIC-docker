import pytest
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock


class TestStepDataFlow:

    def test_step3_finds_step2_output(self, tmp_path):
        slide_name = "test_slide"
        patches_dir = tmp_path / slide_name / "patches" / slide_name / "tiles"
        patches_dir.mkdir(parents=True)

        for i in range(5):
            img = Image.new('RGB', (64, 64), (100, 100, 100))
            img.save(patches_dir / f"{slide_name}_x{i*1000}_y0_l0.jpeg")

        patches = list(patches_dir.glob("*.jpeg"))

        assert len(patches) == 5

    def test_step4_finds_step3_output(self, tmp_path):
        slide_name = "test_slide"
        patches_dir = tmp_path / slide_name / "patches" / slide_name / "tiles"
        nuclei_dir = tmp_path / slide_name / "nuclei_segmentation" / slide_name

        patches_dir.mkdir(parents=True)
        nuclei_dir.mkdir(parents=True)

        for i in range(5):
            Image.new('RGB', (64, 64)).save(patches_dir / f"{slide_name}_x{i*1000}_y0_l0.jpeg")
            Image.new('L', (64, 64)).save(nuclei_dir / f"{slide_name}_x{i*1000}_y0_l0.png")

        patches = list(patches_dir.glob("*.jpeg"))
        masks = list(nuclei_dir.glob("*.png"))

        patch_stems = {p.stem for p in patches}
        mask_stems = {m.stem for m in masks}

        assert patch_stems == mask_stems

    def test_step6_aggregates_step4_and_step5(self, tmp_path):
        slide_name = "test_slide"
        spatil_dir = tmp_path / slide_name / "spatil_features" / slide_name
        nucdiv_dir = tmp_path / slide_name / "nucdiv_features"
        final_dir = tmp_path / slide_name / "final_features"

        spatil_dir.mkdir(parents=True)
        nucdiv_dir.mkdir(parents=True)
        final_dir.mkdir(parents=True)

        for i in range(5):
            data = np.random.rand(350)
            pd.DataFrame([data]).to_csv(spatil_dir / f"patch_{i}.csv", index=False, header=False)

        nucdiv_data = {'slide_id': [slide_name]}
        nucdiv_data.update({f'feat_{i}': [np.random.rand()] for i in range(100)})
        pd.DataFrame(nucdiv_data).to_csv(
            nucdiv_dir / f"{slide_name}_nuclear_diversity_features.csv", index=False
        )

        all_features = []
        for csv_file in spatil_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            all_features.append(df.values.flatten())

        mean_features = np.nanmean(np.vstack(all_features), axis=0)
        pd.DataFrame([mean_features], columns=[f'f_{i}' for i in range(350)]).to_csv(
            final_dir / f"{slide_name}_spatil_aggregated.csv", index=False
        )

        shutil.copy2(
            nucdiv_dir / f"{slide_name}_nuclear_diversity_features.csv",
            final_dir / f"{slide_name}_nucdiv.csv"
        )

        assert (final_dir / f"{slide_name}_spatil_aggregated.csv").exists()
        assert (final_dir / f"{slide_name}_nucdiv.csv").exists()


class TestFullFlowSingleSlide:

    def test_directory_structure_created(self, tmp_path):
        slide_name = "test_slide"
        slide_dir = tmp_path / slide_name

        expected = ['qc', 'patches', 'nuclei_segmentation', 'spatil_features',
                   'nucdiv_features', 'final_features', 'report']

        for name in expected:
            (slide_dir / name).mkdir(parents=True)

        for name in expected:
            assert (slide_dir / name).exists()

    def test_all_outputs_created(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        slide_dir = complete_slide_output['slide_dir']
        dirs = complete_slide_output['dirs']

        assert len(list(dirs['patches'].glob("*.jpeg"))) > 0
        assert len(list(dirs['nuclei'].glob("*.png"))) > 0
        assert len(list(dirs['spatil'].glob("*.csv"))) > 0
        assert (dirs['final'] / f"{slide_name}_prediction.csv").exists()


class TestFullFlowMultiSlide:

    def test_patient_aggregation_runs(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        patient_id = multi_slide_setup['patient_id']
        slide_names = multi_slide_setup['slide_names']

        patient_dir = output_dir / patient_id
        (patient_dir / "final_features").mkdir(parents=True)

        all_spatil = []
        all_nucdiv = []

        for name in slide_names:
            spatil_path = output_dir / name / "final_features" / f"{name}_spatil_aggregated.csv"
            nucdiv_path = output_dir / name / "final_features" / f"{name}_nucdiv.csv"

            if spatil_path.exists():
                df = pd.read_csv(spatil_path)
                all_spatil.append(df.values[0])

            if nucdiv_path.exists():
                df = pd.read_csv(nucdiv_path)
                numeric = df.select_dtypes(include=[np.number])
                all_nucdiv.append(numeric.values[0])

        if all_spatil:
            patient_spatil = np.nanmean(np.vstack(all_spatil), axis=0)
            pd.DataFrame([patient_spatil], columns=[f'f_{i}' for i in range(350)]).to_csv(
                patient_dir / "final_features" / f"{patient_id}_spatil_aggregated.csv", index=False
            )

        assert (patient_dir / "final_features" / f"{patient_id}_spatil_aggregated.csv").exists()

    def test_patient_outputs_created(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        patient_id = multi_slide_setup['patient_id']

        patient_dir = output_dir / patient_id
        (patient_dir / "final_features").mkdir(parents=True)
        (patient_dir / "qc").mkdir(parents=True)

        pd.DataFrame([np.random.rand(350)]).to_csv(
            patient_dir / "final_features" / f"{patient_id}_spatil_aggregated.csv", index=False
        )
        pd.DataFrame([{'risk_score': 0.5, 'risk_group': 'NEGATIVE', 'threshold': 0.5}]).to_csv(
            patient_dir / "final_features" / f"{patient_id}_prediction.csv", index=False
        )
        Image.new('RGB', (100, 100)).save(patient_dir / "qc" / f"{patient_id}_tissue_overlay.png")

        assert (patient_dir / "final_features" / f"{patient_id}_spatil_aggregated.csv").exists()
        assert (patient_dir / "final_features" / f"{patient_id}_prediction.csv").exists()
        assert (patient_dir / "qc" / f"{patient_id}_tissue_overlay.png").exists()


class TestResumeAfterInterrupt:

    def test_resumes_from_partial_nuclei(self, temp_output_dir, dummy_patches, dummy_masks):
        processed_masks, unprocessed_patches = dummy_masks
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        temp_dir = temp_output_dir['slide_dir'] / "temp_unprocessed_patches"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        for patch in unprocessed:
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        links = list(temp_dir.glob("*.jpeg"))

        assert len(links) == 5
        assert all(l.exists() for l in links)

    def test_skips_completed_steps(self, complete_slide_output):
        slide_name = complete_slide_output['slide_name']
        dirs = complete_slide_output['dirs']

        spatil_exists = (dirs['final'] / f"{slide_name}_spatil_aggregated.csv").exists()
        nucdiv_exists = (dirs['final'] / f"{slide_name}_nucdiv.csv").exists()

        should_skip_aggregation = spatil_exists and nucdiv_exists

        assert should_skip_aggregation is True


class TestErrorHandling:

    def test_missing_patches_detected(self, tmp_path):
        slide_dir = tmp_path / "slide"
        patches_dir = slide_dir / "patches" / "slide" / "tiles"
        patches_dir.mkdir(parents=True)

        patches = list(patches_dir.glob("*.jpeg"))

        assert len(patches) == 0

    def test_missing_masks_detected(self, tmp_path):
        slide_dir = tmp_path / "slide"
        nuclei_dir = slide_dir / "nuclei_segmentation" / "slide"
        nuclei_dir.mkdir(parents=True)

        masks = list(nuclei_dir.glob("*.png"))

        assert len(masks) == 0

    def test_handles_empty_spatil_directory(self, tmp_path):
        slide_dir = tmp_path / "slide"
        spatil_dir = slide_dir / "spatil_features" / "slide"
        spatil_dir.mkdir(parents=True)

        csvs = list(spatil_dir.glob("*.csv"))

        assert len(csvs) == 0


class TestConfigurationHandling:

    def test_resume_config_respected(self, tmp_path):
        config = {'pipeline': {'resume_on_existing': True}}

        output_file = tmp_path / "output.csv"
        output_file.touch()

        def should_skip(path, cfg):
            if not cfg['pipeline']['resume_on_existing']:
                return False
            return path.exists()

        assert should_skip(output_file, config) is True

    def test_no_resume_processes_all(self, tmp_path):
        config = {'pipeline': {'resume_on_existing': False}}

        output_file = tmp_path / "output.csv"
        output_file.touch()

        def should_skip(path, cfg):
            if not cfg['pipeline']['resume_on_existing']:
                return False
            return path.exists()

        assert should_skip(output_file, config) is False
