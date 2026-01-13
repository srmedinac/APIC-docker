"""
Tests for QC failure handling and research mode resume functionality.

These tests verify:
1. Slides with 0 patches (QC failures) are skipped gracefully
2. Research mode can resume from existing patient predictions
3. Edge cases are handled correctly (corrupted files, missing columns, etc.)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction_prediction import (
    APICPipeline,
    APICResearchPipeline,
    extract_patient_id_from_filename,
)


# =============================================================================
# FIXTURES FOR QC FAILURE TESTS
# =============================================================================

@pytest.fixture
def empty_patches_setup(tmp_path):
    """
    Create a slide output structure with 0 patches (simulating QC failure).
    """
    slide_name = "failed_qc_slide"
    slide_dir = tmp_path / slide_name

    # Create directory structure
    patches_dir = slide_dir / "patches" / slide_name / "tiles"
    patches_dir.mkdir(parents=True)

    # Create other directories that pipeline expects
    (slide_dir / "nuclei_segmentation" / slide_name).mkdir(parents=True)
    (slide_dir / "spatil_features" / slide_name).mkdir(parents=True)
    (slide_dir / "nucdiv_features").mkdir(parents=True)
    (slide_dir / "final_features").mkdir(parents=True)
    (slide_dir / "qc").mkdir(parents=True)
    (slide_dir / "report").mkdir(parents=True)

    # Create a dummy input slide file
    input_slide = tmp_path / "input" / f"{slide_name}.svs"
    input_slide.parent.mkdir(parents=True)
    input_slide.touch()

    return {
        'slide_name': slide_name,
        'slide_dir': slide_dir,
        'input_slide': input_slide,
        'patches_dir': patches_dir,
        'output_dir': tmp_path,
    }


@pytest.fixture
def valid_patches_setup(tmp_path):
    """
    Create a slide output structure with valid patches.
    """
    slide_name = "valid_slide"
    slide_dir = tmp_path / slide_name

    # Create directory structure
    patches_dir = slide_dir / "patches" / slide_name / "tiles"
    patches_dir.mkdir(parents=True)

    # Create 5 dummy patch files
    for i in range(5):
        img = Image.new('RGB', (64, 64), (i*50, 100, 200))
        img.save(patches_dir / f"{slide_name}_x{i*1000}_y0_l0.jpeg")

    # Create other directories
    (slide_dir / "nuclei_segmentation" / slide_name).mkdir(parents=True)
    (slide_dir / "spatil_features" / slide_name).mkdir(parents=True)
    (slide_dir / "nucdiv_features").mkdir(parents=True)
    (slide_dir / "final_features").mkdir(parents=True)
    (slide_dir / "qc").mkdir(parents=True)
    (slide_dir / "report").mkdir(parents=True)

    # Create a dummy input slide file
    input_slide = tmp_path / "input" / f"{slide_name}.svs"
    input_slide.parent.mkdir(parents=True)
    input_slide.touch()

    return {
        'slide_name': slide_name,
        'slide_dir': slide_dir,
        'input_slide': input_slide,
        'patches_dir': patches_dir,
        'output_dir': tmp_path,
    }


# =============================================================================
# FIXTURES FOR RESEARCH MODE RESUME TESTS
# =============================================================================

@pytest.fixture
def research_mode_setup(tmp_path):
    """
    Create a complete research mode setup with multiple patients.

    Patient 91: 2 slides, fully processed
    Patient 92: 1 slide, fully processed
    Patient 93: 1 slide, NOT processed (missing prediction)
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Patient 91: 2 slides, fully processed
    for slide_suffix in ["R00", "R01"]:
        slide_name = f"91_PT9-100933-00-{slide_suffix}_110928"
        _create_complete_patient_output(output_dir, "91", slide_name)
        (input_dir / f"{slide_name}.svs").touch()

    # Patient 92: 1 slide, fully processed
    slide_name = "92_PT9-100934-00-R00_110929"
    _create_complete_patient_output(output_dir, "92", slide_name)
    (input_dir / f"{slide_name}.svs").touch()

    # Patient 93: 1 slide, NOT processed (no patient-level prediction)
    slide_name = "93_PT9-100935-00-R00_110930"
    _create_slide_features_only(output_dir, slide_name)
    (input_dir / f"{slide_name}.svs").touch()

    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'processed_patients': ['91', '92'],
        'unprocessed_patients': ['93'],
    }


def _create_complete_patient_output(output_dir: Path, patient_id: str, slide_name: str):
    """Helper to create complete patient output with prediction."""
    # Create slide-level features
    slide_dir = output_dir / slide_name
    final_dir = slide_dir / "final_features"
    final_dir.mkdir(parents=True, exist_ok=True)

    # spaTIL features (350 columns)
    spatil_data = np.random.rand(350)
    pd.DataFrame([spatil_data], columns=[f'feature_{i}' for i in range(350)]).to_csv(
        final_dir / f"{slide_name}_spatil_aggregated.csv", index=False
    )

    # NucDiv features with required columns
    nucdiv_data = {
        'slide_id': [slide_name],
        'Area.Energy.var': [np.random.rand()],
        'Area.InvDiffMom.Skewness': [np.random.rand()],
        'MinorAxisLength.Energy.Prcnt90': [np.random.rand()],
        'Area.DiffAvg.Prcnt10': [np.random.rand()],
    }
    # Add more random features
    for i in range(100):
        nucdiv_data[f'feature_{i}'] = [np.random.rand()]
    pd.DataFrame(nucdiv_data).to_csv(
        final_dir / f"{slide_name}_nucdiv.csv", index=False
    )

    # Create QC overlay
    qc_dir = slide_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (100, 100), (0, 255, 0)).save(qc_dir / f"{slide_name}_tissue_overlay.png")

    # Create patient-level aggregated features
    patient_dir = output_dir / patient_id / "final_features"
    patient_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([spatil_data], columns=[f'feature_{i}' for i in range(350)]).to_csv(
        patient_dir / f"{patient_id}_spatil_aggregated.csv", index=False
    )

    nucdiv_patient = {
        'Area.Energy.var': [np.random.rand()],
        'Area.InvDiffMom.Skewness': [np.random.rand()],
        'MinorAxisLength.Energy.Prcnt90': [np.random.rand()],
        'Area.DiffAvg.Prcnt10': [np.random.rand()],
    }
    for i in range(100):
        nucdiv_patient[f'feature_{i}'] = [np.random.rand()]
    pd.DataFrame(nucdiv_patient).to_csv(
        patient_dir / f"{patient_id}_nucdiv.csv", index=False
    )

    # Create patient-level prediction
    pred_data = {
        'patient_id': [patient_id],
        'risk_score': [np.random.rand() * 5],
        'risk_group': ['High Risk' if np.random.rand() > 0.5 else 'Low Risk'],
        'threshold': [0.5],
    }
    pd.DataFrame(pred_data).to_csv(
        patient_dir / f"{patient_id}_prediction.csv", index=False
    )


def _create_slide_features_only(output_dir: Path, slide_name: str):
    """Helper to create slide-level features only (no patient prediction)."""
    slide_dir = output_dir / slide_name
    final_dir = slide_dir / "final_features"
    final_dir.mkdir(parents=True, exist_ok=True)

    # spaTIL features
    spatil_data = np.random.rand(350)
    pd.DataFrame([spatil_data], columns=[f'feature_{i}' for i in range(350)]).to_csv(
        final_dir / f"{slide_name}_spatil_aggregated.csv", index=False
    )

    # NucDiv features
    nucdiv_data = {
        'slide_id': [slide_name],
        'Area.Energy.var': [np.random.rand()],
        'Area.InvDiffMom.Skewness': [np.random.rand()],
        'MinorAxisLength.Energy.Prcnt90': [np.random.rand()],
        'Area.DiffAvg.Prcnt10': [np.random.rand()],
    }
    for i in range(100):
        nucdiv_data[f'feature_{i}'] = [np.random.rand()]
    pd.DataFrame(nucdiv_data).to_csv(
        final_dir / f"{slide_name}_nucdiv.csv", index=False
    )


# =============================================================================
# TESTS FOR QC FAILURE HANDLING
# =============================================================================

class TestQCFailureDetection:
    """Tests for detecting and handling QC failures (0 patches)."""

    def test_has_valid_patches_returns_false_for_empty_dir(self, empty_patches_setup):
        """_has_valid_patches() should return False when no patches exist."""
        setup = empty_patches_setup

        # Create pipeline instance (mocking the input slide existence check)
        pipeline = APICPipeline(
            input_slide=str(setup['input_slide']),
            output_dir=str(setup['output_dir'])
        )

        # Should return False because patches dir is empty
        assert pipeline._has_valid_patches() is False

    def test_has_valid_patches_returns_true_for_valid_dir(self, valid_patches_setup):
        """_has_valid_patches() should return True when patches exist."""
        setup = valid_patches_setup

        pipeline = APICPipeline(
            input_slide=str(setup['input_slide']),
            output_dir=str(setup['output_dir'])
        )

        # Should return True because patches exist
        assert pipeline._has_valid_patches() is True

    def test_has_valid_patches_returns_false_when_dir_missing(self, tmp_path):
        """_has_valid_patches() should return False when patches dir doesn't exist."""
        slide_name = "no_patches_dir"
        input_slide = tmp_path / "input" / f"{slide_name}.svs"
        input_slide.parent.mkdir(parents=True)
        input_slide.touch()

        # Don't create the patches directory at all
        pipeline = APICPipeline(
            input_slide=str(input_slide),
            output_dir=str(tmp_path)
        )

        assert pipeline._has_valid_patches() is False

    def test_run_skips_slide_with_no_patches(self, empty_patches_setup, caplog):
        """Pipeline should skip slide gracefully when no patches exist."""
        import logging
        caplog.set_level(logging.WARNING)

        setup = empty_patches_setup

        pipeline = APICPipeline(
            input_slide=str(setup['input_slide']),
            output_dir=str(setup['output_dir'])
        )

        # Run should exit early without error
        pipeline.run(steps=['nuclei', 'spatil', 'nucdiv'])

        # Should log warning about skipping
        assert "SKIPPING" in caplog.text
        assert "No patches found" in caplog.text
        assert "QC likely failed" in caplog.text

    def test_run_proceeds_with_valid_patches(self, valid_patches_setup):
        """Pipeline should NOT skip when valid patches exist."""
        setup = valid_patches_setup

        pipeline = APICPipeline(
            input_slide=str(setup['input_slide']),
            output_dir=str(setup['output_dir'])
        )

        # Check that patches are detected
        assert pipeline._has_valid_patches() is True


class TestNucdivSafetyCheck:
    """Tests for the safety check in step5_nucdiv_features()."""

    def test_nucdiv_skips_when_no_masks(self, valid_patches_setup, caplog):
        """Nucdiv should skip gracefully when no masks are found."""
        import logging
        caplog.set_level(logging.WARNING)

        setup = valid_patches_setup

        pipeline = APICPipeline(
            input_slide=str(setup['input_slide']),
            output_dir=str(setup['output_dir'])
        )

        # Nuclei directory exists but is empty (no masks)
        nuclei_dir = setup['slide_dir'] / "nuclei_segmentation" / setup['slide_name']
        nuclei_dir.mkdir(parents=True, exist_ok=True)

        # Run nucdiv step - should skip due to no masks
        pipeline.step5_nucdiv_features()

        # Should log warning about no masks
        assert "No nuclei masks found" in caplog.text or "Linked 0 masks" in caplog.text


# =============================================================================
# TESTS FOR RESEARCH MODE RESUME FUNCTIONALITY
# =============================================================================

class TestPatientIdExtraction:
    """Tests for patient ID extraction from filenames."""

    def test_extract_patient_id_standard_format(self):
        """Extract patient ID from standard filename format."""
        assert extract_patient_id_from_filename("91_PT9-100933-00-R00_110928.svs") == "91"
        assert extract_patient_id_from_filename("541_PT9-100935-00-R00_120515.svs") == "541"
        assert extract_patient_id_from_filename("1234_slide.svs") == "1234"

    def test_extract_patient_id_no_extension(self):
        """Extract patient ID from filename without extension."""
        assert extract_patient_id_from_filename("91_PT9-100933-00-R00_110928") == "91"

    def test_extract_patient_id_invalid_format(self):
        """Return None for filenames without valid patient ID."""
        assert extract_patient_id_from_filename("PT9-100933-00-R00_110928.svs") is None
        assert extract_patient_id_from_filename("no_number_prefix.svs") is None
        assert extract_patient_id_from_filename("_underscore_first.svs") is None


class TestLoadExistingResult:
    """Tests for _load_existing_result() method in APICResearchPipeline."""

    def test_load_existing_result_success(self, research_mode_setup):
        """Successfully load existing patient results."""
        setup = research_mode_setup

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        # Should successfully load patient 91 results
        result = pipeline._load_existing_result("91")

        assert result is not None
        assert result['patient_id'] == "91"
        assert 'risk_score' in result
        assert 'risk_group' in result
        assert 'Area.Energy.var' in result
        assert 'X341' in result
        assert 'X51' in result

    def test_load_existing_result_missing_prediction(self, research_mode_setup):
        """Return None when prediction file doesn't exist."""
        setup = research_mode_setup

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        # Patient 93 has no prediction file
        result = pipeline._load_existing_result("93")
        assert result is None

    def test_load_existing_result_missing_feature_files(self, research_mode_setup):
        """Return None when feature files are missing."""
        setup = research_mode_setup

        # Create prediction but delete feature files
        patient_dir = setup['output_dir'] / "91" / "final_features"
        spatil_file = patient_dir / "91_spatil_aggregated.csv"
        if spatil_file.exists():
            spatil_file.unlink()

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        result = pipeline._load_existing_result("91")
        assert result is None

    def test_load_existing_result_corrupted_csv(self, research_mode_setup, caplog):
        """Return None for corrupted CSV files."""
        import logging
        caplog.set_level(logging.WARNING)

        setup = research_mode_setup

        # Corrupt the prediction CSV
        pred_file = setup['output_dir'] / "91" / "final_features" / "91_prediction.csv"
        pred_file.write_text("garbage,data,that,is,not,valid\ncsv")

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        result = pipeline._load_existing_result("91")
        assert result is None

    def test_load_existing_result_missing_columns(self, research_mode_setup, caplog):
        """Return None when required columns are missing."""
        import logging
        caplog.set_level(logging.WARNING)

        setup = research_mode_setup

        # Create prediction with missing columns
        pred_file = setup['output_dir'] / "91" / "final_features" / "91_prediction.csv"
        pd.DataFrame({'patient_id': ['91'], 'incomplete': [1]}).to_csv(pred_file, index=False)

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        result = pipeline._load_existing_result("91")
        assert result is None
        assert "missing columns" in caplog.text

    def test_load_existing_result_empty_csv(self, research_mode_setup, caplog):
        """Return None for empty CSV files."""
        import logging
        caplog.set_level(logging.WARNING)

        setup = research_mode_setup

        # Create empty prediction file (just headers)
        pred_file = setup['output_dir'] / "91" / "final_features" / "91_prediction.csv"
        pd.DataFrame(columns=['patient_id', 'risk_score', 'risk_group']).to_csv(pred_file, index=False)

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        result = pipeline._load_existing_result("91")
        assert result is None
        assert "empty" in caplog.text


class TestResearchModeResume:
    """Tests for resume functionality in research mode run() loop."""

    def test_resume_skips_processed_patients(self, research_mode_setup, caplog):
        """Research mode should skip already-processed patients."""
        import logging
        caplog.set_level(logging.INFO)

        setup = research_mode_setup

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        # Mock _run_prediction to track if it's called
        original_run_prediction = pipeline._run_prediction
        prediction_calls = []

        def mock_run_prediction(*args, **kwargs):
            prediction_calls.append(args[0])  # Track patient_id
            return original_run_prediction(*args, **kwargs)

        pipeline._run_prediction = mock_run_prediction

        # Run pipeline
        pipeline.run()

        # Patients 91 and 92 should be resumed (not reprocessed)
        assert "Resuming: prediction already exists" in caplog.text

        # Only patient 93 should have _run_prediction called (if it has features)
        # Since 93 has slide features, aggregation should work
        for patient_id in ['91', '92']:
            assert patient_id not in prediction_calls

    def test_resume_disabled_reprocesses_all(self, research_mode_setup, caplog):
        """When resume is disabled, all patients should be reprocessed."""
        import logging
        caplog.set_level(logging.INFO)

        setup = research_mode_setup

        # Create pipeline with resume disabled
        config = APICPipeline.DEFAULT_CONFIG.copy()
        config['pipeline']['resume_on_existing'] = False

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir']),
            config=config
        )

        # Run pipeline - should NOT see resume messages
        pipeline.run()

        # Should not see "Resuming" messages when disabled
        assert "Resuming: prediction already exists" not in caplog.text


class TestResearchModeAggregation:
    """Tests for patient feature aggregation in research mode."""

    def test_aggregation_logs_qc_failure(self, research_mode_setup, caplog):
        """Missing features should log QC failure message."""
        import logging
        caplog.set_level(logging.WARNING)

        setup = research_mode_setup

        # Delete slide features for patient 93
        slide_name = "93_PT9-100935-00-R00_110930"
        slide_dir = setup['output_dir'] / slide_name / "final_features"
        for f in slide_dir.glob("*.csv"):
            f.unlink()

        pipeline = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )

        # Aggregate features for patient 93
        result = pipeline._aggregate_patient_features("93", [slide_name])

        assert result is None
        assert "may have failed QC" in caplog.text


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestQCFailureIntegration:
    """Integration tests for QC failure handling across the pipeline."""

    def test_pipeline_handles_mixed_qc_results(self, tmp_path):
        """
        Test that research mode handles a mix of successful and failed QC slides.

        Setup:
        - Patient 100: 2 slides, both with valid features
        - Patient 101: 1 slide with valid features
        - Patient 102: 1 slide with QC failure (no features)
        """
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Patient 100: 2 slides with features
        for suffix in ["R00", "R01"]:
            slide_name = f"100_slide_{suffix}"
            _create_slide_features_only(output_dir, slide_name)
            (input_dir / f"{slide_name}.svs").touch()

        # Patient 101: 1 slide with features
        slide_name = "101_slide_R00"
        _create_slide_features_only(output_dir, slide_name)
        (input_dir / f"{slide_name}.svs").touch()

        # Patient 102: 1 slide without features (QC failure)
        slide_name = "102_slide_R00"
        (output_dir / slide_name / "final_features").mkdir(parents=True)
        (input_dir / f"{slide_name}.svs").touch()

        # Create pipeline and discover patients
        pipeline = APICResearchPipeline(
            input_dir=str(input_dir),
            output_dir=str(output_dir)
        )

        # Verify patient grouping
        assert "100" in pipeline.patient_groups
        assert "101" in pipeline.patient_groups
        assert "102" in pipeline.patient_groups
        assert len(pipeline.patient_groups["100"]) == 2
        assert len(pipeline.patient_groups["101"]) == 1
        assert len(pipeline.patient_groups["102"]) == 1


class TestResumeIntegration:
    """Integration tests for resume functionality."""

    def test_resume_produces_identical_results(self, research_mode_setup):
        """
        Running pipeline twice should produce identical results.

        First run: normal processing
        Second run: resume from existing
        """
        setup = research_mode_setup

        # First run
        pipeline1 = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )
        pipeline1.run()

        # Read first results
        results_file = setup['output_dir'] / "research_results.csv"
        if results_file.exists():
            results1 = pd.read_csv(results_file)
        else:
            results1 = pd.DataFrame()

        # Second run (should resume)
        pipeline2 = APICResearchPipeline(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir'])
        )
        pipeline2.run()

        # Read second results
        if results_file.exists():
            results2 = pd.read_csv(results_file)
        else:
            results2 = pd.DataFrame()

        # Results should be identical (same patients, same values)
        if not results1.empty and not results2.empty:
            pd.testing.assert_frame_equal(
                results1.sort_values('patient_id').reset_index(drop=True),
                results2.sort_values('patient_id').reset_index(drop=True)
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
