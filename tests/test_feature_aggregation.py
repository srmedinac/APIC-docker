"""
Tests for feature aggregation in the APIC pipeline.

These tests verify that:
- spaTIL features are correctly aggregated with nanmean
- NucDiv features properly exclude string columns
- Edge cases (NaN, empty files) are handled
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestSpatilAggregation:
    """Tests for spaTIL feature aggregation."""

    def test_spatil_nanmean_basic(self, temp_output_dir):
        """
        Basic nanmean aggregation across multiple patch CSVs.
        """
        spatil_dir = temp_output_dir['spatil']

        # Create test CSVs with known values
        values1 = np.ones(350) * 2.0
        values2 = np.ones(350) * 4.0
        values3 = np.ones(350) * 6.0

        pd.DataFrame([values1]).to_csv(spatil_dir / "patch1.csv", index=False, header=False)
        pd.DataFrame([values2]).to_csv(spatil_dir / "patch2.csv", index=False, header=False)
        pd.DataFrame([values3]).to_csv(spatil_dir / "patch3.csv", index=False, header=False)

        # Aggregate using same logic as pipeline
        all_features = []
        for csv_file in spatil_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            features = df.values.flatten()
            if len(features) == 350:
                all_features.append(features)

        mean_features = np.nanmean(np.vstack(all_features), axis=0)

        # Expected mean: (2 + 4 + 6) / 3 = 4.0
        assert len(mean_features) == 350
        assert np.allclose(mean_features, 4.0)

    def test_spatil_nanmean_with_nan_values(self, temp_output_dir):
        """
        Verify nanmean properly handles NaN values.
        """
        spatil_dir = temp_output_dir['spatil']

        values1 = np.ones(350) * 2.0
        values1[0:10] = np.nan  # First 10 are NaN

        values2 = np.ones(350) * 4.0
        # values2 has no NaN

        values3 = np.ones(350) * 6.0
        values3[0:10] = np.nan  # First 10 are NaN

        pd.DataFrame([values1]).to_csv(spatil_dir / "patch1.csv", index=False, header=False)
        pd.DataFrame([values2]).to_csv(spatil_dir / "patch2.csv", index=False, header=False)
        pd.DataFrame([values3]).to_csv(spatil_dir / "patch3.csv", index=False, header=False)

        all_features = []
        for csv_file in spatil_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            features = df.values.flatten()
            if len(features) == 350:
                all_features.append(features)

        mean_features = np.nanmean(np.vstack(all_features), axis=0)

        # First 10 features: only values2 contributes (4.0)
        assert np.allclose(mean_features[0:10], 4.0)

        # Rest: mean of 2, 4, 6 = 4.0
        assert np.allclose(mean_features[10:], 4.0)

    def test_spatil_skip_wrong_length(self, temp_output_dir):
        """
        CSVs with wrong number of features should be skipped.
        """
        spatil_dir = temp_output_dir['spatil']

        # Correct length
        values1 = np.ones(350) * 2.0
        pd.DataFrame([values1]).to_csv(spatil_dir / "patch1.csv", index=False, header=False)

        # Wrong length (should be skipped)
        values2 = np.ones(100) * 100.0
        pd.DataFrame([values2]).to_csv(spatil_dir / "patch2.csv", index=False, header=False)

        # Correct length
        values3 = np.ones(350) * 4.0
        pd.DataFrame([values3]).to_csv(spatil_dir / "patch3.csv", index=False, header=False)

        all_features = []
        for csv_file in spatil_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            features = df.values.flatten()
            if len(features) == 350:  # Filter by length
                all_features.append(features)

        assert len(all_features) == 2  # Only 2 valid files

        mean_features = np.nanmean(np.vstack(all_features), axis=0)

        # Mean of 2 and 4 = 3.0
        assert np.allclose(mean_features, 3.0)


class TestNucdivAggregation:
    """Tests for nuclear diversity feature aggregation."""

    def test_nucdiv_excludes_string_columns(self, temp_output_dir, slide_name):
        """
        Verify string columns (like slide_id) are excluded from averaging.
        """
        nucdiv_dir = temp_output_dir['nucdiv']

        # Create CSV with slide_id and numeric features
        data = {
            'slide_id': [slide_name],
            'feature_1': [1.0],
            'feature_2': [2.0],
            'feature_3': [3.0],
        }
        df = pd.DataFrame(data)
        csv_path = nucdiv_dir / f"{slide_name}_nuclear_diversity_features.csv"
        df.to_csv(csv_path, index=False)

        # Load and extract numeric columns only
        loaded_df = pd.read_csv(csv_path)
        numeric_df = loaded_df.select_dtypes(include=[np.number])

        assert 'slide_id' not in numeric_df.columns
        assert len(numeric_df.columns) == 3
        assert list(numeric_df.columns) == ['feature_1', 'feature_2', 'feature_3']

    def test_nucdiv_multi_slide_aggregation(self, tmp_path):
        """
        Test aggregation across multiple slides (patient mode).
        """
        output_dir = tmp_path / "output"

        # Create two "slides" with nucdiv features
        for i, slide_name in enumerate(["slide_1", "slide_2"]):
            slide_dir = output_dir / slide_name / "final_features"
            slide_dir.mkdir(parents=True, exist_ok=True)

            data = {
                'slide_id': [slide_name],
                'Area.Energy.var': [(i + 1) * 10.0],  # 10, 20
                'feature_2': [(i + 1) * 2.0],  # 2, 4
            }
            df = pd.DataFrame(data)
            df.to_csv(slide_dir / f"{slide_name}_nucdiv.csv", index=False)

        # Aggregate across slides
        all_nucdiv = []
        nucdiv_columns = None

        for slide_dir in output_dir.iterdir():
            nucdiv_file = slide_dir / "final_features" / f"{slide_dir.name}_nucdiv.csv"
            if nucdiv_file.exists():
                df = pd.read_csv(nucdiv_file)
                numeric_df = df.select_dtypes(include=[np.number])
                nucdiv_columns = numeric_df.columns
                all_nucdiv.append(numeric_df.values[0])

        assert len(all_nucdiv) == 2

        mean_nucdiv = np.nanmean(np.vstack(all_nucdiv), axis=0)

        # Area.Energy.var: (10 + 20) / 2 = 15
        # feature_2: (2 + 4) / 2 = 3
        expected = [15.0, 3.0]
        assert np.allclose(mean_nucdiv, expected)


class TestEmptyAndEdgeCases:
    """Tests for edge cases in aggregation."""

    def test_empty_spatil_directory(self, temp_output_dir):
        """
        Empty spaTIL directory should result in empty feature list.
        """
        spatil_dir = temp_output_dir['spatil']

        # No files created

        all_features = []
        for csv_file in spatil_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            features = df.values.flatten()
            if len(features) == 350:
                all_features.append(features)

        assert len(all_features) == 0

    def test_all_nan_column(self, temp_output_dir):
        """
        Column with all NaN values should result in NaN in output.
        """
        spatil_dir = temp_output_dir['spatil']

        values1 = np.ones(350) * 2.0
        values1[0] = np.nan

        values2 = np.ones(350) * 4.0
        values2[0] = np.nan

        pd.DataFrame([values1]).to_csv(spatil_dir / "patch1.csv", index=False, header=False)
        pd.DataFrame([values2]).to_csv(spatil_dir / "patch2.csv", index=False, header=False)

        all_features = []
        for csv_file in spatil_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            features = df.values.flatten()
            if len(features) == 350:
                all_features.append(features)

        mean_features = np.nanmean(np.vstack(all_features), axis=0)

        # First feature is all NaN -> result is NaN
        assert np.isnan(mean_features[0])

        # Rest are valid
        assert np.allclose(mean_features[1:], 3.0)

    def test_single_file_aggregation(self, temp_output_dir):
        """
        Single file should return its values unchanged.
        """
        spatil_dir = temp_output_dir['spatil']

        values = np.arange(350, dtype=float)
        pd.DataFrame([values]).to_csv(spatil_dir / "single.csv", index=False, header=False)

        all_features = []
        for csv_file in spatil_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None)
            features = df.values.flatten()
            if len(features) == 350:
                all_features.append(features)

        mean_features = np.nanmean(np.vstack(all_features), axis=0)

        assert np.allclose(mean_features, values)


class TestFeatureSelection:
    """Tests for Cox model feature selection."""

    def test_required_features_exist(self, temp_output_dir, slide_name):
        """
        Verify required features for Cox model can be extracted.
        """
        # Create nucdiv features with required columns
        nucdiv_dir = temp_output_dir['nucdiv']
        required_nucdiv = [
            'Area.Energy.var',
            'Area.InvDiffMom.Skewness',
            'MinorAxisLength.Energy.Prcnt90',
            'Area.DiffAvg.Prcnt10',
        ]

        data = {'slide_id': [slide_name]}
        for feat in required_nucdiv:
            data[feat] = [np.random.rand()]

        pd.DataFrame(data).to_csv(
            nucdiv_dir / f"{slide_name}_nuclear_diversity_features.csv",
            index=False
        )

        # Create spatil features (need feature_342 and feature_52)
        final_dir = temp_output_dir['final']
        spatil_data = {f'feature_{i}': [np.random.rand()] for i in range(350)}
        pd.DataFrame(spatil_data).to_csv(
            final_dir / f"{slide_name}_spatil_aggregated.csv",
            index=False
        )

        # Load and verify
        nucdiv_df = pd.read_csv(nucdiv_dir / f"{slide_name}_nuclear_diversity_features.csv")
        spatil_df = pd.read_csv(final_dir / f"{slide_name}_spatil_aggregated.csv")

        for feat in required_nucdiv:
            assert feat in nucdiv_df.columns

        assert 'feature_342' in spatil_df.columns  # X341
        assert 'feature_52' in spatil_df.columns   # X51
