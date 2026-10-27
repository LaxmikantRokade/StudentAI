"""
Preprocessor Module for Student Performance Prediction.

This module provides preprocessing pipelines using Scikit-learn ColumnTransformer
and Pipelines to correctly transform numeric and categorical features without
data leakage.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Target column name
TARGET_COLUMN = "G3"

# Explicit column type definitions based on the UCI dataset schema
NUMERIC_FEATURES: List[str] = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
]

# Numeric features excluding period evaluation grades (for early-semester behavioral predictions)
BEHAVIORAL_NUMERIC_FEATURES: List[str] = [
    col for col in NUMERIC_FEATURES if col not in ("G1", "G2")
]

CATEGORICAL_FEATURES: List[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]


def separate_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates independent feature matrix X from target vector y.

    Parameters:
        df (pd.DataFrame): Full dataframe including target.
        target_col (str): Name of target column (default: 'G3').

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessor(include_grades: bool = True) -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer that:
    1. Standardizes numeric features (Z-score scaling: (x - mean) / std).
    2. One-hot encodes categorical features with handle_unknown='ignore'.
    3. Handles missing values safely with SimpleImputer.

    Parameters:
        include_grades (bool): If True, includes G1 and G2 in numeric features (15 cols).
                              If False, excludes G1 and G2 for behavioral prediction (13 cols).

    Note on Data Leakage Prevention:
    This transformer is never fitted on the entire dataset. It is integrated
    into a Scikit-learn Pipeline and fitted strictly on X_train during training.

    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    numeric_cols = NUMERIC_FEATURES if include_grades else BEHAVIORAL_NUMERIC_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def create_full_pipeline(regressor_model, include_grades: bool = True) -> Pipeline:
    """
    Creates an end-to-end Pipeline combining the feature preprocessor
    and a regression estimator.

    Parameters:
        regressor_model: A Scikit-learn regression estimator.
        include_grades (bool): Whether to include G1 and G2 in the pipeline features.

    Returns:
        Pipeline: Combined preprocessing and regression pipeline.
    """
    preprocessor = build_preprocessor(include_grades=include_grades)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor_model),
        ]
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from src.data_loader import load_data
    except ImportError:
        from data_loader import load_data

    raw_df = load_data()
    X, y = separate_features_target(raw_df)
    preprocessor = build_preprocessor()

    # Verify fit_transform works without error
    X_transformed = preprocessor.fit_transform(X)
    print(f"Features X shape: {X.shape}")
    print(f"Target y shape: {y.shape}")
    print(f"Transformed matrix shape: {X_transformed.shape}")
    print(f"Numeric features count: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features count: {len(CATEGORICAL_FEATURES)}")
    print("Preprocessor test passed successfully!")
