"""
Inference and Prediction Module.

Provides robust prediction functions with input validation, grade clamping (0 to 20),
academic tier categorization, and actionable recommendations.
"""

import os
import sys
import joblib
from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.logger import get_logger

logger = get_logger(__name__)

# Default representative values for any omitted features
DEFAULT_STUDENT_PROFILE: Dict[str, Any] = {
    # Categorical defaults (mode values)
    "school": "GP",
    "sex": "F",
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    # Numeric defaults (median values)
    "age": 16,
    "Medu": 3,
    "Fedu": 2,
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 4,
    "absences": 4,
    "G1": 11,
    "G2": 11,
}


def load_trained_model(model_path: str = "models/student_model.pkl"):
    """
    Loads the serialized Scikit-learn Pipeline from disk.

    Parameters:
        model_path (str): Filepath to the pickled model.

    Returns:
        Pipeline: Loaded Scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model file is not present.
    """
    if not os.path.exists(model_path):
        logger.error("Model file not found: '%s'", model_path)
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'src/train.py' first to build and save the model."
        )
    pipeline = joblib.load(model_path)
    logger.info("Loaded trained pipeline from '%s'.", model_path)
    return pipeline


def load_behavioral_model(model_path: str = "models/behavioral_model.pkl"):
    """
    Loads the serialized behavioral-only Scikit-learn Pipeline from disk.

    Parameters:
        model_path (str): Filepath to the pickled behavioral model.

    Returns:
        Pipeline: Loaded Scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model file is not present.
    """
    if not os.path.exists(model_path):
        logger.error("Behavioral model file not found: '%s'", model_path)
        raise FileNotFoundError(
            f"Trained behavioral model not found at '{model_path}'. "
            "Please run 'src/train.py' first to build and save the model."
        )
    pipeline = joblib.load(model_path)
    logger.info("Loaded trained behavioral pipeline from '%s'.", model_path)
    return pipeline


def categorize_grade(grade: float) -> Tuple[str, str, str]:
    """
    Categorizes the numeric grade (0-20) into educational performance tiers.

    Returns:
        Tuple[str, str, str]: (tier_name, risk_level, advisory_text)
    """
    if grade >= 16.0:
        return (
            "Excellent (Tier 1: 16-20)",
            "Low Risk",
            "Outstanding academic performance. Strong foundation for advanced coursework.",
        )
    elif grade >= 14.0:
        return (
            "Good (Tier 2: 14-15)",
            "Low Risk",
            "Solid performance. Maintain current study consistency to reach top tier.",
        )
    elif grade >= 10.0:
        return (
            "Pass / Satisfactory (Tier 3: 10-13)",
            "Moderate Risk",
            "Passing grade achieved. Recommend increasing weekly study hours and addressing weak topics.",
        )
    else:
        return (
            "Needs Intervention / Fail (Tier 4: 0-9)",
            "High Risk",
            "Urgent academic support needed. Consider remedial classes, attendance monitoring, and teacher mentoring.",
        )


def validate_and_prepare_input(
    user_input: Union[Dict[str, Any], pd.DataFrame]
) -> pd.DataFrame:
    """
    Validates user input, fills missing columns with sensible defaults,
    and returns a clean DataFrame matching the trained pipeline schema.

    Parameters:
        user_input (dict or pd.DataFrame): Feature key-values.

    Returns:
        pd.DataFrame: Formatted 1-row DataFrame ready for pipeline.predict().
    """
    if isinstance(user_input, pd.DataFrame):
        input_dict = user_input.iloc[0].to_dict()
    elif isinstance(user_input, dict):
        input_dict = user_input.copy()
    else:
        logger.error("Invalid input type for validate_and_prepare_input: %s", type(user_input))
        raise TypeError("user_input must be a dictionary or a Pandas DataFrame.")

    # Validate numerical boundaries if provided
    if "G1" in input_dict and not (0 <= float(input_dict["G1"]) <= 20):
        logger.warning("Invalid G1 value: %s", input_dict["G1"])
        raise ValueError(f"G1 grade must be between 0 and 20, got {input_dict['G1']}")
    if "G2" in input_dict and not (0 <= float(input_dict["G2"]) <= 20):
        logger.warning("Invalid G2 value: %s", input_dict["G2"])
        raise ValueError(f"G2 grade must be between 0 and 20, got {input_dict['G2']}")
    if "absences" in input_dict and not (0 <= float(input_dict["absences"]) <= 93):
        logger.warning("Invalid absences value: %s", input_dict["absences"])
        raise ValueError(f"Absences must be between 0 and 93, got {input_dict['absences']}")
    if "studytime" in input_dict and not (1 <= int(input_dict["studytime"]) <= 4):
        logger.warning("Invalid studytime value: %s", input_dict["studytime"])
        raise ValueError(f"studytime must be 1, 2, 3, or 4, got {input_dict['studytime']}")

    # Build complete record using defaults for missing features
    full_record = DEFAULT_STUDENT_PROFILE.copy()
    for key, value in input_dict.items():
        if key in full_record:
            full_record[key] = value

    return pd.DataFrame([full_record])


def predict_student_grade(
    student_profile: Union[Dict[str, Any], pd.DataFrame],
    model=None,
    model_path: str = "models/student_model.pkl",
) -> Dict[str, Any]:
    """
    End-to-end inference function: validates inputs, predicts grade,
    clamps to valid range (0-20), and returns detailed educational diagnostics.

    Parameters:
        student_profile (dict or DataFrame): Input features.
        model (optional): Pre-loaded model pipeline.
        model_path (str): Path to model file if model is not pre-loaded.

    Returns:
        Dict[str, Any]: Prediction results, academic tier, and recommendations.
    """
    if model is None:
        model = load_trained_model(model_path)

    # Validate and build DataFrame
    input_df = validate_and_prepare_input(student_profile)

    # Raw model prediction
    raw_pred = float(model.predict(input_df)[0])

    # Rule 9: Clamp predicted grade strictly within the 0 to 20 range
    clamped_grade = max(0.0, min(20.0, raw_pred))
    final_grade = round(clamped_grade, 2)

    tier, risk, advisory = categorize_grade(final_grade)
    logger.info(
        "Prediction: raw=%.3f  clamped=%.2f  tier='%s'  risk='%s'",
        raw_pred, final_grade, tier, risk,
    )

    return {
        "predicted_grade": final_grade,
        "raw_prediction": round(raw_pred, 3),
        "tier": tier,
        "risk_level": risk,
        "advisory": advisory,
        "input_summary": {
            "G1": input_df["G1"].iloc[0],
            "G2": input_df["G2"].iloc[0],
            "studytime": input_df["studytime"].iloc[0],
            "absences": input_df["absences"].iloc[0],
            "failures": input_df["failures"].iloc[0],
        },
    }


if __name__ == "__main__":
    # Test cases
    print("Testing Prediction Engine...")

    # Case 1: High performing student
    high_student = {
        "G1": 17,
        "G2": 18,
        "studytime": 4,
        "absences": 1,
        "failures": 0,
        "higher": "yes",
    }
    result_high = predict_student_grade(high_student)
    print("\nCase 1 (High Achiever):")
    print(f"  Predicted G3: {result_high['predicted_grade']} / 20")
    print(f"  Tier:         {result_high['tier']}")
    print(f"  Risk:         {result_high['risk_level']}")

    # Case 2: At-risk student
    low_student = {
        "G1": 5,
        "G2": 6,
        "studytime": 1,
        "absences": 18,
        "failures": 2,
        "higher": "no",
    }
    result_low = predict_student_grade(low_student)
    print("\nCase 2 (At-Risk Student):")
    print(f"  Predicted G3: {result_low['predicted_grade']} / 20")
    print(f"  Tier:         {result_low['tier']}")
    print(f"  Risk:         {result_low['risk_level']}")
    print(f"  Advisory:     {result_low['advisory']}")
