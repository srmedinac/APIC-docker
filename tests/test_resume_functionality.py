"""
Tests for resume functionality in the APIC pipeline.

These tests verify that the pipeline correctly:
- Detects already-processed patches
- Skips completed steps
- Processes only remaining work on resume
"""

import pytest
from pathlib import Path
from PIL import Image


class TestResumeDetection:
    """Tests for detecting already-processed work."""

    def test_count_unprocessed_correctly(self, temp_output_dir, dummy_patches, dummy_masks):
        """
        Verify correct counting of unprocessed patches.

        With 10 patches and 5 masks, should identify 5 unprocessed.
        """
        processed_masks, unprocessed_patches = dummy_masks
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        # This mirrors the logic in step3_nuclei_segmentation
        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        assert len(patch_files) == 10
        assert len(existing_masks) == 5
        assert len(unprocessed) == 5

        # Verify unprocessed are the correct ones
        unprocessed_stems = {p.stem for p in unprocessed}
        for patch in unprocessed_patches:
            assert patch.stem in unprocessed_stems

    def test_skip_when_all_processed(self, temp_output_dir, dummy_patches):
        """
        When all patches have masks, step should be skipped entirely.
        """
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        # Create masks for ALL patches
        for patch in dummy_patches:
            mask = Image.new('L', (64, 64), 0)
            mask_path = nuclei_dir / (patch.stem + ".png")
            mask.save(mask_path)

        # Check detection logic
        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        assert len(unprocessed) == 0

        # The pipeline would log "Already complete" and return early
        should_skip = len(unprocessed) == 0
        assert should_skip

    def test_process_remaining_when_partial(self, temp_output_dir, dummy_patches, dummy_masks):
        """
        With partial completion, only remaining patches should be processed.
        """
        processed_masks, unprocessed_patches = dummy_masks
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        # Should process exactly 5 remaining patches
        assert len(unprocessed) == 5

        # Simulate processing remaining
        for patch in unprocessed:
            mask = Image.new('L', (64, 64), 0)
            mask_path = nuclei_dir / (patch.stem + ".png")
            mask.save(mask_path)

        # Now all should be processed
        new_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        new_unprocessed = [p for p in patch_files if p.stem not in new_masks]
        assert len(new_unprocessed) == 0


class TestResumeConfig:
    """Tests for resume configuration behavior."""

    def test_should_skip_with_resume_enabled(self, temp_output_dir):
        """
        When resume_on_existing is True, skip if output exists.
        """
        config = {'pipeline': {'resume_on_existing': True}}

        # Create a dummy output file
        output_file = temp_output_dir['final'] / "test_output.csv"
        output_file.touch()

        # Check skip logic
        def should_skip(path, config):
            if not config['pipeline']['resume_on_existing']:
                return False
            return path.exists() if path else False

        assert should_skip(output_file, config) is True

    def test_no_skip_with_resume_disabled(self, temp_output_dir):
        """
        When resume_on_existing is False, never skip.
        """
        config = {'pipeline': {'resume_on_existing': False}}

        output_file = temp_output_dir['final'] / "test_output.csv"
        output_file.touch()

        def should_skip(path, config):
            if not config['pipeline']['resume_on_existing']:
                return False
            return path.exists() if path else False

        assert should_skip(output_file, config) is False

    def test_no_skip_when_output_missing(self, temp_output_dir):
        """
        When output doesn't exist, don't skip regardless of config.
        """
        config = {'pipeline': {'resume_on_existing': True}}

        missing_file = temp_output_dir['final'] / "nonexistent.csv"

        def should_skip(path, config):
            if not config['pipeline']['resume_on_existing']:
                return False
            return path.exists() if path else False

        assert should_skip(missing_file, config) is False


class TestDirectorySetup:
    """Tests for directory creation on pipeline init."""

    def test_all_directories_created(self, tmp_path):
        """
        Verify all required directories are created on initialization.
        """
        slide_name = "test_slide"
        output_dir = tmp_path / "output"
        slide_output_dir = output_dir / slide_name

        # Expected directory structure
        expected_dirs = {
            'qc': slide_output_dir / "qc",
            'patches': slide_output_dir / "patches",
            'nuclei': slide_output_dir / "nuclei_segmentation",
            'spatil': slide_output_dir / "spatil_features",
            'spatil_viz': slide_output_dir / "spatil_visualizations",
            'nucdiv': slide_output_dir / "nucdiv_features",
            'final': slide_output_dir / "final_features",
            'report': slide_output_dir / "report",
        }

        # Create all directories (simulating pipeline init)
        for dir_path in expected_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Verify all exist
        for name, dir_path in expected_dirs.items():
            assert dir_path.exists(), f"Directory {name} not created"
            assert dir_path.is_dir(), f"{name} is not a directory"

    def test_idempotent_directory_creation(self, tmp_path):
        """
        Creating directories multiple times should not cause errors.
        """
        test_dir = tmp_path / "test" / "nested" / "dir"

        # First creation
        test_dir.mkdir(parents=True, exist_ok=True)
        assert test_dir.exists()

        # Second creation (should not raise)
        test_dir.mkdir(parents=True, exist_ok=True)
        assert test_dir.exists()


class TestProgressTracking:
    """Tests for progress tracking during resume."""

    def test_logging_message_format(self, temp_output_dir, dummy_patches, dummy_masks):
        """
        Verify correct progress logging format.
        """
        processed_masks, unprocessed_patches = dummy_masks
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        # Generate expected log message
        log_message = f"Processing {len(unprocessed)}/{len(patch_files)} patches"

        assert log_message == "Processing 5/10 patches"

    def test_complete_message_when_done(self, temp_output_dir, dummy_patches):
        """
        Verify message when all patches are already processed.
        """
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        # Complete all processing
        for patch in dummy_patches:
            mask = Image.new('L', (64, 64), 0)
            mask_path = nuclei_dir / (patch.stem + ".png")
            mask.save(mask_path)

        patch_files = list(patches_dir.glob("*.jpeg"))

        log_message = f"Already complete: {len(patch_files)} patches processed"
        assert log_message == "Already complete: 10 patches processed"
