#!/usr/bin/env python3
"""
Cox Model Prediction Wrapper

Calls R script to predict biomarker risk score and group for a new patient.
Usage: python predict_biomarker.py --features features.csv --output output.csv
"""

import os
import argparse
import subprocess
import pandas as pd
from pathlib import Path


def predict_cox_model(features_file, training_data_path, output_file, r_script_path="src/predict_risk_score.R"):
    """
    Predict Cox model risk score and group for new patient
    
    Args:
        features_file: CSV with patient features (patient_id, feature columns)
        training_data_path: Path to training data
        output_file: Where to save predictions
        r_script_path: Path to R prediction script
    """
    
    # Check files exist
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data not found: {training_data_path}")
    if not os.path.exists(r_script_path):
        raise FileNotFoundError(f"R script not found: {r_script_path}")
    
    # Call R script
    cmd = [
        "Rscript", r_script_path,
        training_data_path,
        features_file,
        output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"R script error:\n{result.stderr}")
        raise RuntimeError("R prediction failed")
    
    print(result.stdout)
    
    # Read and return results
    predictions = pd.read_csv(output_file)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict Cox model biomarker")
    parser.add_argument("--features", "-f", required=True, help="Patient features CSV")
    parser.add_argument("--training_data", "-t", required=True, help="CHAARTED training data CSV")
    parser.add_argument("--output", "-o", required=True, help="Output predictions CSV")
    parser.add_argument("--r_script", default="src/predict_risk_score.R", help="R script path")
    
    args = parser.parse_args()
    
    predictions = predict_cox_model(
        args.features, 
        args.training_data, 
        args.output,
        args.r_script
    )
    
    print("\nPredictions:")
    print(predictions)


if __name__ == "__main__":
    main()