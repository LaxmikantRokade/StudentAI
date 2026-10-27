"""
Data Loader Module for Student Performance Prediction.

This module provides functions to safely load, inspect, and calculate
summary statistics for the student dataset.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


def load_data(filepath: str = "data/student_performance.csv") -> pd.DataFrame:
    """
    Loads student performance data from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or missing the target variable 'G3'.
    """
    if not os.path.exists(filepath):
        logger.error("Dataset file not found: '%s'", filepath)
        raise FileNotFoundError(
            f"Dataset file not found at '{filepath}'. "
            "Please ensure data/student_performance.csv exists."
        )

    df = pd.read_csv(filepath)
    logger.info("Loaded dataset from '%s' — shape: %s", filepath, df.shape)

    if df.empty:
        logger.error("Dataset at '%s' is empty.", filepath)
        raise ValueError(f"The dataset at '{filepath}' is empty.")

    if "G3" not in df.columns:
        logger.error("Target column 'G3' missing in '%s'. Columns: %s", filepath, list(df.columns))
        raise ValueError("Required target column 'G3' is missing from the dataset.")

    return df


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Checks the basic health and quality of the dataset.

    Returns a dictionary containing:
        - row_count: Total number of records
        - column_count: Total number of features + target
        - missing_values_total: Total NaN count across all cells
        - missing_per_column: Series of missing values per column
        - duplicate_rows: Number of identical duplicate rows
        - data_types: Counts of data types in the DataFrame
    """
    result = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_per_column": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
    }
    logger.debug(
        "Data quality — rows: %d, missing: %d, duplicates: %d",
        result["row_count"], result["missing_values_total"], result["duplicate_rows"],
    )
    return result


def get_summary_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculates detailed descriptive statistics for numeric features,
    including mean, median, mode, standard deviation, min, and max.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Table of descriptive statistics for numeric columns.
            - Dictionary of categorical column summaries.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    stats_df = pd.DataFrame(index=numeric_df.columns)

    stats_df["mean"] = numeric_df.mean()
    stats_df["median"] = numeric_df.median()
    stats_df["std"] = numeric_df.std()
    stats_df["min"] = numeric_df.min()
    stats_df["25%"] = numeric_df.quantile(0.25)
    stats_df["75%"] = numeric_df.quantile(0.75)
    stats_df["max"] = numeric_df.max()

    # Calculate mode for numeric columns (first mode if multimodal)
    mode_series = numeric_df.mode().iloc[0]
    stats_df["mode"] = mode_series

    # Categorical summary
    categorical_df = df.select_dtypes(exclude=[np.number])
    cat_summary = {}
    for col in categorical_df.columns:
        cat_summary[col] = {
            "unique_values": int(categorical_df[col].nunique()),
            "top_value": str(categorical_df[col].mode().iloc[0]),
            "top_frequency": int(categorical_df[col].value_counts().iloc[0]),
        }

    return stats_df.round(2), cat_summary


if __name__ == "__main__":
    # Self-test block
    data = load_data()
    quality = check_data_quality(data)
    num_stats, cat_stats = get_summary_statistics(data)
    print(f"Data successfully loaded. Shape: {quality['row_count']} rows x {quality['column_count']} columns")
    print(f"Missing values: {quality['missing_values_total']}, Duplicates: {quality['duplicate_rows']}")
    print("\nSample Numeric Stats (G1, G2, G3):")
    print(num_stats.loc[["G1", "G2", "G3"]])
