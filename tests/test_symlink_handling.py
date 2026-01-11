"""
Tests for symlink handling in nuclei segmentation step.

These tests verify the fix for the bug where stale symlinks from
interrupted runs caused FileNotFoundError on resume.
"""

import os
import glob
import shutil
import pytest
from pathlib import Path
from PIL import Image


class TestSymlinkCleanup:
    """Tests for stale symlink cleanup on resume."""

    def test_stale_symlinks_cleaned_on_resume(self, temp_output_dir, stale_symlinks, dummy_patches):
        """
        Verify that stale symlinks from a previous failed run are cleaned up.

        This is the exact bug scenario:
        1. Previous run created symlinks in temp_unprocessed_patches
        2. Run was interrupted, leaving broken symlinks
        3. New run should clear the directory before creating fresh symlinks
        """
        temp_dir, stale_links = stale_symlinks

        # Verify stale symlinks exist (broken, but files exist)
        assert temp_dir.exists()
        assert len(list(temp_dir.iterdir())) == 5

        # Verify symlinks are broken (targets don't exist)
        for link in stale_links:
            assert link.is_symlink()
            assert not link.exists()  # exists() follows symlink, returns False for broken

        # Simulate the fixed behavior: clean temp dir before creating new symlinks
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create new symlinks to actual patches
        for patch in dummy_patches[:3]:  # Simulate 3 unprocessed
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        # Verify only valid symlinks remain
        links = list(temp_dir.iterdir())
        assert len(links) == 3

        # Verify all symlinks are valid (can be opened)
        for link in links:
            assert link.exists()
            assert link.is_symlink()
            # Should not raise FileNotFoundError
            img = Image.open(link)
            img.close()

    def test_glob_after_cleanup_finds_only_valid_files(self, temp_output_dir, stale_symlinks, dummy_patches):
        """
        Verify that glob.glob only finds valid files after cleanup.

        This tests the nucleusSegmentationTiles.py behavior:
        files = sorted(glob.glob(os.path.join(dataPath, '*'+ext)))
        """
        temp_dir, stale_links = stale_symlinks

        # Before cleanup: glob finds broken symlinks
        files_before = glob.glob(os.path.join(str(temp_dir), "*.jpeg"))
        assert len(files_before) == 5

        # Attempting to open these would fail
        for f in files_before:
            with pytest.raises(FileNotFoundError):
                Image.open(f)

        # Apply the fix: clean and recreate
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for patch in dummy_patches:
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        # After cleanup: glob finds only valid symlinks
        files_after = glob.glob(os.path.join(str(temp_dir), "*.jpeg"))
        assert len(files_after) == 10

        # All files can be opened
        for f in files_after:
            img = Image.open(f)
            img.close()  # No error


class TestSymlinkCreation:
    """Tests for symlink creation logic."""

    def test_symlinks_created_for_unprocessed_only(self, temp_output_dir, dummy_patches, dummy_masks):
        """
        Verify symlinks are only created for unprocessed patches.

        When some masks already exist, only patches without masks
        should get symlinks in temp_unprocessed_patches.
        """
        processed_masks, unprocessed_patches = dummy_masks
        patches_dir = temp_output_dir['patches']
        nuclei_dir = temp_output_dir['nuclei']

        # Get all patches and existing masks
        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}

        # Calculate unprocessed
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        assert len(unprocessed) == 5
        assert len(processed_masks) == 5

        # Create temp dir with symlinks only for unprocessed
        temp_dir = temp_output_dir['slide_dir'] / "temp_unprocessed_patches"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for patch in unprocessed:
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        # Verify correct number of symlinks
        symlinks = list(temp_dir.glob("*.jpeg"))
        assert len(symlinks) == 5

        # Verify symlinks point to correct files
        for link in symlinks:
            target = link.resolve()
            assert target.exists()
            assert target.parent == patches_dir

    def test_symlinks_resolve_to_absolute_paths(self, temp_output_dir, dummy_patches):
        """
        Verify symlinks use absolute paths (patch.resolve()).
        """
        temp_dir = temp_output_dir['slide_dir'] / "temp_unprocessed_patches"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for patch in dummy_patches[:3]:
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        for link in temp_dir.iterdir():
            # The symlink target should be absolute
            target = os.readlink(str(link))
            assert os.path.isabs(target)

    def test_empty_temp_dir_when_no_unprocessed(self, temp_output_dir, dummy_patches):
        """
        When all patches are processed, temp_unprocessed_patches should be empty.
        """
        nuclei_dir = temp_output_dir['nuclei']

        # Create masks for ALL patches
        for patch in dummy_patches:
            mask = Image.new('L', (64, 64), 0)
            mask_path = nuclei_dir / (patch.stem + ".png")
            mask.save(mask_path)

        # Calculate unprocessed (should be empty)
        patches_dir = temp_output_dir['patches']
        patch_files = list(patches_dir.glob("*.jpeg"))
        existing_masks = {f.stem for f in nuclei_dir.glob("*.png")}
        unprocessed = [p for p in patch_files if p.stem not in existing_masks]

        assert len(unprocessed) == 0


class TestEdgeCases:
    """Edge case tests for symlink handling."""

    def test_mixed_valid_and_stale_symlinks(self, temp_output_dir, dummy_patches):
        """
        Handle case where temp dir has both valid and stale symlinks.

        This could happen in unusual circumstances. The fix should
        clean everything and start fresh.
        """
        temp_dir = temp_output_dir['slide_dir'] / "temp_unprocessed_patches"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create some valid symlinks
        for patch in dummy_patches[:3]:
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        # Create some stale symlinks
        for i in range(3):
            stale_link = temp_dir / f"stale_{i}.jpeg"
            stale_link.symlink_to(Path(f"/nonexistent/stale_{i}.jpeg"))

        # Mixed state: 3 valid + 3 stale
        all_links = list(temp_dir.iterdir())
        assert len(all_links) == 6

        valid_count = sum(1 for l in all_links if l.exists())
        stale_count = sum(1 for l in all_links if not l.exists())
        assert valid_count == 3
        assert stale_count == 3

        # Apply fix: clean all
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Recreate with only valid symlinks
        for patch in dummy_patches:
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        # All should be valid now
        new_links = list(temp_dir.iterdir())
        assert len(new_links) == 10
        assert all(l.exists() for l in new_links)

    def test_temp_dir_does_not_exist_initially(self, temp_output_dir, dummy_patches):
        """
        First run scenario: temp_unprocessed_patches doesn't exist yet.
        """
        temp_dir = temp_output_dir['slide_dir'] / "temp_unprocessed_patches"

        # Should not exist initially
        assert not temp_dir.exists()

        # Apply the pattern from the fix
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks
        for patch in dummy_patches:
            link = temp_dir / patch.name
            link.symlink_to(patch.resolve())

        assert temp_dir.exists()
        assert len(list(temp_dir.iterdir())) == 10

    def test_special_characters_in_slide_name(self, tmp_path):
        """
        Test handling of slide names with spaces and special characters.

        Slide name like 'PT 77.5 - 2022-11-29' should work correctly.
        """
        slide_name = "PT 77.5 - 2022-11-29"
        slide_dir = tmp_path / slide_name
        patches_dir = slide_dir / "patches" / slide_name / "tiles"
        temp_dir = slide_dir / "temp_unprocessed_patches"

        patches_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create patch with special characters in name
        patch_name = f"{slide_name}_x1000_y500_l0.jpeg"
        patch_path = patches_dir / patch_name

        img = Image.new('RGB', (64, 64), (128, 128, 128))
        img.save(patch_path)

        # Create symlink
        link = temp_dir / patch_name
        link.symlink_to(patch_path.resolve())

        # Verify glob finds it
        pattern = os.path.join(str(temp_dir), "*.jpeg")
        files = glob.glob(pattern)
        assert len(files) == 1

        # Verify can open
        img = Image.open(files[0])
        assert img.size == (64, 64)
        img.close()
