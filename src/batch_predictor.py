"""
StudentAI — Batch CSV Intelligence & Prediction Module (U4)
===========================================================
Handles ingestion, rigorous validation, vectorized inference, and export
for batches of student records.

Key Capabilities:
1. Schema & Column Verification:
   - Validates presence of all required numeric and categorical columns.
   - Preserves arbitrary row-identifier columns (e.g., 'student_id', 'name', 'roll_no').
2. Deep Row-Level Data Quality Auditing:
   - Identifies NaN / null / empty cells without hallucinating or inventing data.
   - Validates numeric boundaries (e.g., G1/G2 in [0, 20], absences in [0, 93]).
   - Checks categorical values against the authentic UCI domain vocabulary.
3. High-Throughput Vectorized Inference:
   - Reuses the existing production and behavioral pipelines (zero retraining).
   - Clamps predictions strictly within the academic range [0.0, 20.0].
   - Derives performance tiers, risk flags, and educational advisories.
4. Export & Telemetry Reporting:
   - Preserves original row indices and ordering.
   - Returns both a rich merged DataFrame and an actionable cohort summary.
   - Provides sample CSV template generation.

Author: StudentAI Intelligence Engineering Team
Lab: 3rd Year B.Tech IT — Data Science & Visualization
"""

import os
import sys
import io
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import pandas as pd

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.logger import get_logger
from src.preprocessor import (
    NUMERIC_FEATURES,
    BEHAVIORAL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.predict import (
    load_trained_model,
    load_behavioral_model,
    categorize_grade,
)

logger = get_logger("batch_predictor")

# Authentic UCI domain definitions
VALID_CATEGORICAL_VALUES: Dict[str, Set[str]] = {
    "school": {"GP", "MS"},
    "sex": {"F", "M"},
    "address": {"R", "U"},
    "famsize": {"GT3", "LE3"},
    "Pstatus": {"A", "T"},
    "Mjob": {"at_home", "health", "other", "services", "teacher"},
    "Fjob": {"at_home", "health", "other", "services", "teacher"},
    "reason": {"course", "home", "other", "reputation"},
    "guardian": {"father", "mother", "other"},
    "schoolsup": {"no", "yes"},
    "famsup": {"no", "yes"},
    "paid": {"no", "yes"},
    "activities": {"no", "yes"},
    "nursery": {"no", "yes"},
    "higher": {"no", "yes"},
    "internet": {"no", "yes"},
    "romantic": {"no", "yes"},
}

NUMERIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "age": (10.0, 30.0),
    "Medu": (0.0, 4.0),
    "Fedu": (0.0, 4.0),
    "traveltime": (1.0, 4.0),
    "studytime": (1.0, 4.0),
    "failures": (0.0, 4.0),
    "famrel": (1.0, 5.0),
    "freetime": (1.0, 5.0),
    "goout": (1.0, 5.0),
    "Dalc": (1.0, 5.0),
    "Walc": (1.0, 5.0),
    "health": (1.0, 5.0),
    "absences": (0.0, 93.0),
    "G1": (0.0, 20.0),
    "G2": (0.0, 20.0),
}


def get_required_columns(use_behavioral: bool = False) -> List[str]:
    """
    Returns the ordered list of required feature column names.
    Excludes G1 and G2 if behavioral mode is selected.
    """
    num_cols = BEHAVIORAL_NUMERIC_FEATURES if use_behavioral else NUMERIC_FEATURES
    return list(num_cols) + list(CATEGORICAL_FEATURES)


def validate_batch_dataframe(
    df: pd.DataFrame,
    use_behavioral: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Validates an uploaded DataFrame of student records for batch inference.

    Parameters:
        df (pd.DataFrame): Raw uploaded DataFrame.
        use_behavioral (bool): If True, validates for early-semester mode (no G1/G2).

    Returns:
        Tuple[valid_df, invalid_df, summary_dict]:
            - valid_df: DataFrame containing only valid rows.
            - invalid_df: DataFrame of rejected rows with 'validation_error' column.
            - summary_dict: Metrics on row validity and column schema compliance.

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if df is None or df.empty:
        logger.warning("Batch validation failed: uploaded DataFrame is empty.")
        raise ValueError("The uploaded CSV file is empty. Please provide a file with student records.")

    # Clean whitespace in column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = get_required_columns(use_behavioral=use_behavioral)
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        logger.warning("Batch validation failed: missing columns %s", missing_cols)
        raise ValueError(
            f"The uploaded CSV is missing {len(missing_cols)} required column(s): "
            f"{', '.join(missing_cols)}. Please check the column headers or download the sample template."
        )

    valid_indices: List[int] = []
    invalid_rows: List[Dict[str, Any]] = []

    num_cols = BEHAVIORAL_NUMERIC_FEATURES if use_behavioral else NUMERIC_FEATURES

    for idx, row in df.iterrows():
        errors: List[str] = []

        # 1. Check for missing / NaN values in required columns
        for col in required_cols:
            val = row[col]
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                errors.append(f"Missing value in '{col}'")

        # 2. Check numeric boundaries and data types
        for col in num_cols:
            val = row[col]
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                continue  # already reported as missing

            try:
                num_val = float(val)
                min_b, max_b = NUMERIC_BOUNDS[col]
                if not (min_b <= num_val <= max_b):
                    errors.append(f"'{col}' value {val} out of range [{min_b:.0f}, {max_b:.0f}]")
            except (ValueError, TypeError):
                errors.append(f"'{col}' must be a number, got '{val}'")

        # 3. Check categorical values
        for col in CATEGORICAL_FEATURES:
            val = row[col]
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                continue

            str_val = str(val).strip()
            valid_set = VALID_CATEGORICAL_VALUES.get(col, set())
            if str_val not in valid_set:
                errors.append(f"'{col}' invalid value '{str_val}' (expected: {', '.join(sorted(valid_set))})")

        if errors:
            row_dict = row.to_dict()
            row_dict["_row_number"] = idx + 1
            row_dict["validation_error"] = "; ".join(errors)
            invalid_rows.append(row_dict)
        else:
            valid_indices.append(idx)

    valid_df = df.loc[valid_indices].copy()
    invalid_df = pd.DataFrame(invalid_rows) if invalid_rows else pd.DataFrame()

    total_rows = len(df)
    valid_count = len(valid_df)
    invalid_count = len(invalid_df)

    summary = {
        "total_rows": total_rows,
        "valid_rows": valid_count,
        "invalid_rows": invalid_count,
        "valid_percentage": round((valid_count / total_rows) * 100.0, 1) if total_rows > 0 else 0.0,
        "missing_columns": [],
    }

    logger.info(
        "Batch validated: %d total rows, %d valid (%.1f%%), %d invalid.",
        total_rows, valid_count, summary["valid_percentage"], invalid_count,
    )
    return valid_df, invalid_df, summary


def predict_batch(
    df: pd.DataFrame,
    model=None,
    use_behavioral: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes batch grade prediction on an uploaded DataFrame.
    Validates all rows, runs model inference on valid rows, and merges
    the predictions, performance tiers, risk flags, and advisories
    into a structured telemetry output DataFrame.

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.
        model (optional): Pre-loaded Scikit-learn Pipeline. If None, loaded from disk.
        use_behavioral (bool): If True, uses behavioral pipeline (excluding G1/G2).

    Returns:
        Tuple[results_df, cohort_summary]:
            - results_df: Formatted output DataFrame preserving all original columns,
                          row numbers, prediction outcomes, and validation notes.
            - cohort_summary: Aggregated metrics across the student cohort.
    """
    if model is None:
        model = load_behavioral_model() if use_behavioral else load_trained_model()

    # Step 1: Validate rows
    valid_df, invalid_df, validation_summary = validate_batch_dataframe(
        df, use_behavioral=use_behavioral
    )

    required_cols = get_required_columns(use_behavioral=use_behavioral)

    # Initialize results containers
    enriched_rows: List[Dict[str, Any]] = []

    # Map predictions for valid rows
    if not valid_df.empty:
        # Cast numeric columns to float to ensure clean pipeline processing
        num_cols = BEHAVIORAL_NUMERIC_FEATURES if use_behavioral else NUMERIC_FEATURES
        X_infer = valid_df[required_cols].copy()
        for c in num_cols:
            X_infer[c] = pd.to_numeric(X_infer[c], errors="coerce")

        raw_preds = model.predict(X_infer)
        clamped_preds = np.clip(raw_preds, 0.0, 20.0)

        for (orig_idx, row), raw_p, clamp_p in zip(valid_df.iterrows(), raw_preds, clamped_preds):
            final_grade = round(float(clamp_p), 2)
            tier, risk, advisory = categorize_grade(final_grade)

            row_data = row.to_dict()
            row_data["_row_number"] = orig_idx + 1
            row_data["status"] = "SUCCESS"
            row_data["validation_error"] = ""
            row_data["predicted_grade"] = final_grade
            row_data["raw_prediction"] = round(float(raw_p), 3)
            row_data["performance_tier"] = tier
            row_data["risk_level"] = risk
            row_data["actionable_advisory"] = advisory
            enriched_rows.append(row_data)

    # Append invalid rows (preserving error notes)
    if not invalid_df.empty:
        for _, row in invalid_df.iterrows():
            row_data = row.to_dict()
            row_data["status"] = "VALIDATION_FAILED"
            row_data["predicted_grade"] = np.nan
            row_data["raw_prediction"] = np.nan
            row_data["performance_tier"] = "N/A (Invalid Data)"
            row_data["risk_level"] = "Unassessed"
            row_data["actionable_advisory"] = "Correct input errors to generate prediction."
            enriched_rows.append(row_data)

    # Reconstruct final DataFrame sorted by original row number
    results_df = pd.DataFrame(enriched_rows)
    if not results_df.empty and "_row_number" in results_df.columns:
        results_df = results_df.sort_values("_row_number").reset_index(drop=True)

    # Calculate cohort summary metrics for valid predictions
    valid_predictions = results_df[results_df["status"] == "SUCCESS"]

    if not valid_predictions.empty:
        mean_grade = float(valid_predictions["predicted_grade"].mean())
        median_grade = float(valid_predictions["predicted_grade"].median())
        min_grade = float(valid_predictions["predicted_grade"].min())
        max_grade = float(valid_predictions["predicted_grade"].max())

        risk_counts = valid_predictions["risk_level"].value_counts().to_dict()
        tier_counts = valid_predictions["performance_tier"].value_counts().to_dict()
    else:
        mean_grade = 0.0
        median_grade = 0.0
        min_grade = 0.0
        max_grade = 0.0
        risk_counts = {}
        tier_counts = {}

    cohort_summary = {
        "total_records": len(df),
        "valid_predictions": len(valid_predictions),
        "failed_validations": len(df) - len(valid_predictions),
        "mean_predicted_grade": round(mean_grade, 2),
        "median_predicted_grade": round(median_grade, 2),
        "min_predicted_grade": round(min_grade, 2),
        "max_predicted_grade": round(max_grade, 2),
        "risk_breakdown": risk_counts,
        "tier_breakdown": tier_counts,
    }

    return results_df, cohort_summary


def generate_sample_batch_csv(n_samples: int = 5) -> str:
    """
    Generates a sample CSV template string pre-populated with valid diverse student profiles
    (high performers, average students, and at-risk students) for testing.

    Parameters:
        n_samples (int): Number of sample rows to include.

    Returns:
        str: CSV-formatted string with headers and valid records.
    """
    sample_records = [
        {
            "student_id": "STU_001",
            "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 4, "Fedu": 3, "Mjob": "health", "Fjob": "services", "reason": "home", "guardian": "mother",
            "traveltime": 1, "studytime": 3, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "yes",
            "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
            "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 1, "health": 5, "absences": 2,
            "G1": 15, "G2": 16,
        },
        {
            "student_id": "STU_002",
            "school": "GP", "sex": "M", "age": 17, "address": "U", "famsize": "LE3", "Pstatus": "T",
            "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "father",
            "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
            "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
            "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4, "absences": 4,
            "G1": 11, "G2": 11,
        },
        {
            "student_id": "STU_003",
            "school": "MS", "sex": "M", "age": 18, "address": "R", "famsize": "GT3", "Pstatus": "A",
            "Medu": 1, "Fedu": 1, "Mjob": "at_home", "Fjob": "other", "reason": "other", "guardian": "mother",
            "traveltime": 2, "studytime": 1, "failures": 2, "schoolsup": "no", "famsup": "no", "paid": "no",
            "activities": "no", "nursery": "no", "higher": "no", "internet": "no", "romantic": "yes",
            "famrel": 3, "freetime": 4, "goout": 4, "Dalc": 2, "Walc": 4, "health": 2, "absences": 18,
            "G1": 7, "G2": 6,
        },
        {
            "student_id": "STU_004",
            "school": "GP", "sex": "F", "age": 15, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 3, "Fedu": 3, "Mjob": "services", "Fjob": "services", "reason": "reputation", "guardian": "mother",
            "traveltime": 1, "studytime": 4, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
            "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
            "famrel": 5, "freetime": 2, "goout": 2, "Dalc": 1, "Walc": 1, "health": 5, "absences": 0,
            "G1": 17, "G2": 18,
        },
        {
            "student_id": "STU_005",
            "school": "MS", "sex": "F", "age": 19, "address": "R", "famsize": "LE3", "Pstatus": "T",
            "Medu": 2, "Fedu": 1, "Mjob": "other", "Fjob": "at_home", "reason": "course", "guardian": "other",
            "traveltime": 3, "studytime": 2, "failures": 1, "schoolsup": "no", "famsup": "no", "paid": "no",
            "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
            "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 3, "absences": 8,
            "G1": 10, "G2": 10,
        },
    ]

    sample_df = pd.DataFrame(sample_records[:n_samples])
    output = io.StringIO()
    sample_df.to_csv(output, index=False)
    return output.getvalue()
