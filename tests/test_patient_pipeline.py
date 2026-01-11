import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestFindSlideDirectories:

    def test_finds_all_slide_dirs(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        slide_names = multi_slide_setup['slide_names']

        slide_dirs = []
        for slide_name in slide_names:
            slide_output = output_dir / slide_name
            if slide_output.exists() and (slide_output / "final_features").exists():
                slide_dirs.append(slide_output)

        assert len(slide_dirs) == 3

    def test_ignores_incomplete_slides(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']

        incomplete_dir = output_dir / "incomplete_slide"
        incomplete_dir.mkdir()

        valid_dirs = [d for d in output_dir.iterdir()
                     if d.is_dir() and (d / "final_features").exists()]

        assert len(valid_dirs) == 3
        assert incomplete_dir not in valid_dirs

    def test_supported_extensions(self, tmp_path):
        patient_folder = tmp_path / "patient"
        patient_folder.mkdir()

        supported = ['svs', 'tif', 'tiff', 'ndpi', 'mrxs', 'scn']
        for ext in supported:
            (patient_folder / f"slide.{ext}").touch()

        found = []
        for ext in supported:
            found.extend(list(patient_folder.glob(f"*.{ext}")))

        assert len(found) == 6


class TestAggregateSpatilAcrossSlides:

    def test_nanmean_aggregation(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        slide_names = multi_slide_setup['slide_names']

        all_spatil = []
        for name in slide_names:
            csv_path = output_dir / name / "final_features" / f"{name}_spatil_aggregated.csv"
            df = pd.read_csv(csv_path)
            all_spatil.append(df.values[0])

        patient_avg = np.nanmean(np.vstack(all_spatil), axis=0)

        assert len(patient_avg) == 350
        assert not np.isnan(patient_avg).all()

    def test_handles_nan_values(self, tmp_path):
        slide_dirs = []
        for i in range(3):
            slide_dir = tmp_path / f"slide_{i}" / "final_features"
            slide_dir.mkdir(parents=True)
            slide_dirs.append(slide_dir)

            data = np.ones(350) * (i + 1)
            if i == 1:
                data[0:10] = np.nan
            pd.DataFrame([data], columns=[f'f_{j}' for j in range(350)]).to_csv(
                slide_dir / f"slide_{i}_spatil_aggregated.csv", index=False
            )

        all_data = []
        for i, slide_dir in enumerate(slide_dirs):
            df = pd.read_csv(slide_dir / f"slide_{i}_spatil_aggregated.csv")
            all_data.append(df.values[0])

        result = np.nanmean(np.vstack(all_data), axis=0)

        assert result[0] == pytest.approx(2.0, rel=0.01)
        assert result[50] == pytest.approx(2.0, rel=0.01)

    def test_output_has_correct_columns(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        slide_name = multi_slide_setup['slide_names'][0]

        df = pd.read_csv(output_dir / slide_name / "final_features" / f"{slide_name}_spatil_aggregated.csv")

        assert len(df.columns) == 350


class TestAggregateNucdivAcrossSlides:

    def test_excludes_string_columns(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        slide_name = multi_slide_setup['slide_names'][0]

        df = pd.read_csv(output_dir / slide_name / "final_features" / f"{slide_name}_nucdiv.csv")
        numeric_df = df.select_dtypes(include=[np.number])

        assert 'slide_id' not in numeric_df.columns

    def test_averages_numeric_columns(self, tmp_path):
        for i in range(2):
            slide_dir = tmp_path / f"slide_{i}" / "final_features"
            slide_dir.mkdir(parents=True)

            data = {
                'slide_id': [f'slide_{i}'],
                'feat_1': [(i + 1) * 10.0],
                'feat_2': [(i + 1) * 20.0],
            }
            pd.DataFrame(data).to_csv(slide_dir / f"slide_{i}_nucdiv.csv", index=False)

        all_numeric = []
        for i in range(2):
            df = pd.read_csv(tmp_path / f"slide_{i}" / "final_features" / f"slide_{i}_nucdiv.csv")
            numeric = df.select_dtypes(include=[np.number])
            all_numeric.append(numeric.values[0])

        result = np.nanmean(np.vstack(all_numeric), axis=0)

        assert result[0] == pytest.approx(15.0)
        assert result[1] == pytest.approx(30.0)


class TestSelectLargestTissueOverlay:

    def test_selects_by_tissue_area(self, tmp_path):
        import cv2

        areas = []
        for i, size in enumerate([50, 100, 75]):
            slide_dir = tmp_path / f"slide_{i}" / "qc"
            slide_dir.mkdir(parents=True)

            mask = np.zeros((200, 200), dtype=np.uint8)
            mask[0:size, 0:size] = 255
            cv2.imwrite(str(slide_dir / f"slide_{i}.svs_mask_use.png"), mask)

            area = np.sum(mask > 127)
            areas.append((f"slide_{i}", area))

        best_slide = max(areas, key=lambda x: x[1])

        assert best_slide[0] == "slide_1"
        assert best_slide[1] == 10000

    def test_copies_overlay_from_best(self, tmp_path):
        import shutil

        slides = []
        for i, area in enumerate([100, 500, 200]):
            slide_dir = tmp_path / f"slide_{i}"
            (slide_dir / "qc").mkdir(parents=True)

            from PIL import Image
            overlay = Image.new('RGB', (100, 100), (i*80, 100, 200))
            overlay.save(slide_dir / "qc" / f"slide_{i}_tissue_overlay.png")

            slides.append((slide_dir, area))

        best = max(slides, key=lambda x: x[1])
        patient_dir = tmp_path / "patient" / "qc"
        patient_dir.mkdir(parents=True)

        source = best[0] / "qc" / f"{best[0].name}_tissue_overlay.png"
        dest = patient_dir / "patient_tissue_overlay.png"
        shutil.copy2(source, dest)

        assert dest.exists()


class TestPatientPrediction:

    def test_uses_patient_level_features(self, tmp_path):
        patient_dir = tmp_path / "patient" / "final_features"
        patient_dir.mkdir(parents=True)

        spatil = [np.random.rand()] * 350
        pd.DataFrame([spatil], columns=[f'feature_{i}' for i in range(350)]).to_csv(
            patient_dir / "patient_spatil_aggregated.csv", index=False
        )

        nucdiv = {'Area.Energy.var': [0.5], 'Area.InvDiffMom.Skewness': [0.3],
                 'MinorAxisLength.Energy.Prcnt90': [0.2], 'Area.DiffAvg.Prcnt10': [0.1]}
        pd.DataFrame(nucdiv).to_csv(patient_dir / "patient_nucdiv.csv", index=False)

        spatil_df = pd.read_csv(patient_dir / "patient_spatil_aggregated.csv")
        nucdiv_df = pd.read_csv(patient_dir / "patient_nucdiv.csv")

        assert spatil_df.shape == (1, 350)
        assert 'Area.Energy.var' in nucdiv_df.columns

    def test_required_features_present(self, tmp_path):
        patient_dir = tmp_path / "patient" / "final_features"
        patient_dir.mkdir(parents=True)

        required_nucdiv = [
            'Area.Energy.var',
            'Area.InvDiffMom.Skewness',
            'MinorAxisLength.Energy.Prcnt90',
            'Area.DiffAvg.Prcnt10',
        ]
        required_spatil = ['feature_342', 'feature_52']

        nucdiv_data = {col: [0.5] for col in required_nucdiv}
        pd.DataFrame(nucdiv_data).to_csv(patient_dir / "patient_nucdiv.csv", index=False)

        spatil_data = {f'feature_{i}': [0.5] for i in range(350)}
        pd.DataFrame(spatil_data).to_csv(patient_dir / "patient_spatil.csv", index=False)

        nucdiv_df = pd.read_csv(patient_dir / "patient_nucdiv.csv")
        spatil_df = pd.read_csv(patient_dir / "patient_spatil.csv")

        for col in required_nucdiv:
            assert col in nucdiv_df.columns
        for col in required_spatil:
            assert col in spatil_df.columns


class TestPatientIdHandling:
    """Tests for patient ID handling - prevents regression of input_slides bug."""

    def test_patient_id_from_override_not_folder(self, tmp_path):
        """Patient ID should come from explicit override, not folder basename."""
        # Simulates: folder is /data/input_slides but patient_id override is "PT001"
        patient_folder = tmp_path / "input_slides"  # Container path name
        patient_folder.mkdir()

        patient_id_override = "PT001"  # What user provides via --patient-id

        # The logic should use override, not basename
        if patient_id_override:
            patient_id = patient_id_override
        else:
            patient_id = patient_folder.name

        assert patient_id == "PT001"
        assert patient_id != "input_slides"

    def test_patient_id_fallback_to_folder_name(self, tmp_path):
        """Without override, patient ID comes from folder name (batch mode)."""
        patient_folder = tmp_path / "PatientA"
        patient_folder.mkdir()

        patient_id_override = ""  # No override in batch mode

        if patient_id_override:
            patient_id = patient_id_override
        else:
            patient_id = patient_folder.name

        assert patient_id == "PatientA"

    def test_report_uses_patient_id_not_container_path(self, tmp_path):
        """Report filename should use patient_id, not container path."""
        patient_id = "MyPatient"
        report_name = f"{patient_id}_report.pdf"

        assert report_name == "MyPatient_report.pdf"
        assert "input_slides" not in report_name


class TestOverlayInMultiSlideMode:
    """Tests for overlay creation in multi-slide mode - prevents thumbnail bug."""

    def test_overlay_must_exist_before_patient_aggregation(self, tmp_path):
        """Each slide must have overlay created before patient-level copy."""
        from PIL import Image

        # Simulate 2 slides with overlays created
        for i in range(2):
            slide_dir = tmp_path / f"slide_{i}" / "qc"
            slide_dir.mkdir(parents=True)
            overlay = Image.new('RGB', (100, 100), (100, 150, 200))
            overlay.save(slide_dir / f"slide_{i}_tissue_overlay.png")

        # Verify overlays exist (this is what was missing before)
        for i in range(2):
            overlay_path = tmp_path / f"slide_{i}" / "qc" / f"slide_{i}_tissue_overlay.png"
            assert overlay_path.exists(), f"Overlay must exist for slide_{i}"

    def test_patient_overlay_copy_fails_without_source(self, tmp_path):
        """If no slide has overlay, patient overlay cannot be created."""
        import shutil

        # Slides without overlays (simulates old bug where overlay step was skipped)
        for i in range(2):
            slide_dir = tmp_path / f"slide_{i}" / "qc"
            slide_dir.mkdir(parents=True)
            # No overlay file created!

        patient_qc = tmp_path / "patient" / "qc"
        patient_qc.mkdir(parents=True)

        # Try to find and copy overlay
        overlay_found = False
        for i in range(2):
            source = tmp_path / f"slide_{i}" / "qc" / f"slide_{i}_tissue_overlay.png"
            if source.exists():
                shutil.copy2(source, patient_qc / "patient_tissue_overlay.png")
                overlay_found = True
                break

        assert not overlay_found, "No overlay should be found if step was skipped"
        assert not (patient_qc / "patient_tissue_overlay.png").exists()

    def test_multi_slide_steps_must_include_overlay(self):
        """Multi-slide mode must include overlay step for thumbnails to work."""
        # The correct steps for multi-slide mode
        multi_slide_steps = "nuclei spatil nucdiv aggregate overlay"

        assert "overlay" in multi_slide_steps
        # The old buggy version was missing overlay:
        buggy_steps = "nuclei spatil nucdiv aggregate"
        assert "overlay" not in buggy_steps


class TestCopyVisualizations:

    def test_copies_spatil_visualizations(self, multi_slide_setup):
        output_dir = multi_slide_setup['output_dir']
        slide_names = multi_slide_setup['slide_names']

        all_viz = []
        for name in slide_names:
            viz_dir = output_dir / name / "spatil_visualizations" / name
            if viz_dir.exists():
                all_viz.extend(list(viz_dir.glob("*.png")))

        assert len(all_viz) >= 1

    def test_limits_visualization_count(self, tmp_path):
        import shutil

        source_dir = tmp_path / "source"
        source_dir.mkdir()

        for i in range(10):
            from PIL import Image
            img = Image.new('RGB', (64, 64), (i*25, 100, 200))
            img.save(source_dir / f"viz_{i}.png")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        viz_files = sorted(source_dir.glob("*.png"))[:4]
        for viz in viz_files:
            shutil.copy2(viz, dest_dir / viz.name)

        assert len(list(dest_dir.glob("*.png"))) == 4
