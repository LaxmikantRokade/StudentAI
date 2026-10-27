"""
Model Training and Evaluation Module.

Trains baseline and candidate regression models on student performance data,
evaluates them with standard metrics (MAE, RMSE, R-squared), and saves the
best-performing Scikit-learn Pipeline.
"""

import os
import sys
import json
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data
from src.preprocessor import separate_features_target, create_full_pipeline
from src.logger import get_logger

logger = get_logger(__name__)


def train_and_evaluate_models(
    data_path: str = "data/student_performance.csv",
    test_size: float = 0.20,
    random_state: int = 42,
    models_dir: str = "models",
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Splits data (80/20), trains multiple regression models via full pipelines,
    evaluates on the unseen test set, and saves the best model to disk.

    Parameters:
        data_path (str): Path to CSV dataset.
        test_size (float): Fraction of data reserved for test set (0.20 = 20%).
        random_state (int): Reproducibility seed (42).
        models_dir (str): Folder where trained model artifact is stored.

    Returns:
        Tuple[Dict[str, Dict[str, float]], str]:
            - Dictionary of model metrics.
            - Name of best-performing model.
    """
    # 1. Load data
    df = load_data(data_path)
    X, y = separate_features_target(df)
    logger.info("Dataset loaded — %d rows, %d features + target.", len(df), X.shape[1])

    # 2. Reproducible 80/20 train/test split
    # Stratification is not applied here because y is continuous numeric marks (0-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Train/test split: %d train (%.0f%%), %d test (%.0f%%), random_state=%d.",
        X_train.shape[0], (1 - test_size) * 100,
        X_test.shape[0], test_size * 100,
        random_state,
    )
    print(f"Training set: {X_train.shape[0]} samples ({(1 - test_size) * 100:.0f}%)")
    print(f"Test set:     {X_test.shape[0]} samples ({test_size * 100:.0f}%)")

    # 3. Define candidate models to evaluate (including Gradient Boosting)
    candidate_estimators = {
        "Baseline (Dummy Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=random_state
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
        ),
    }

    metrics: Dict[str, Dict[str, float]] = {}
    trained_pipelines = {}

    for name, estimator in candidate_estimators.items():
        # Create end-to-end pipeline (ColumnTransformer + Estimator)
        pipeline = create_full_pipeline(estimator, include_grades=True)

        # FIT ONLY ON TRAINING SET - completely avoids data leakage
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predict on unseen test set
        y_pred = pipeline.predict(X_test)

        # Compute standard evaluation metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }
        logger.info("[%s] MAE=%.4f  RMSE=%.4f  R²=%.4f", name, mae, rmse, r2)
        print(f"[{name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

    # Compute 5-fold cross-validation metrics for ensemble models
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_summary: Dict[str, Dict[str, float]] = {}
    for ensemble_name in ["Random Forest Regressor", "Gradient Boosting Regressor"]:
        ens_pipe = trained_pipelines[ensemble_name]
        cv_r2_scores = cross_val_score(ens_pipe, X, y, cv=cv, scoring="r2")
        cv_rmse_scores = -cross_val_score(ens_pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        cv_summary[ensemble_name] = {
            "CV_R2_mean": round(float(cv_r2_scores.mean()), 4),
            "CV_R2_std": round(float(cv_r2_scores.std()), 4),
            "CV_RMSE_mean": round(float(cv_rmse_scores.mean()), 4),
            "CV_RMSE_std": round(float(cv_rmse_scores.std()), 4),
        }
        logger.info(
            "[%s] 5-Fold CV: R²=%.4f ± %.4f, RMSE=%.4f ± %.4f",
            ensemble_name,
            cv_summary[ensemble_name]["CV_R2_mean"],
            cv_summary[ensemble_name]["CV_R2_std"],
            cv_summary[ensemble_name]["CV_RMSE_mean"],
            cv_summary[ensemble_name]["CV_RMSE_std"],
        )

    # 4. Select the best performing full model based on lowest test RMSE
    # Exclude dummy baseline from model selection
    non_baseline_models = {k: v for k, v in metrics.items() if "Baseline" not in k}
    best_model_name = min(non_baseline_models, key=lambda k: non_baseline_models[k]["RMSE"])
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\nBest Full Model Selected: {best_model_name} (RMSE: {metrics[best_model_name]['RMSE']}, R²: {metrics[best_model_name]['R2']})")
    logger.info("Best full model selected: '%s'  RMSE=%.4f  R²=%.4f", best_model_name, metrics[best_model_name]["RMSE"], metrics[best_model_name]["R2"])

    # 5. Train and evaluate Behavioral-Only Model (strictly excluding G1 and G2)
    print("\n--- Training Behavioral-Only Model (Excluding G1 and G2) ---")
    X_train_beh = X_train.drop(columns=["G1", "G2"])
    X_test_beh = X_test.drop(columns=["G1", "G2"])

    beh_estimator = RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=random_state
    )
    behavioral_pipeline = create_full_pipeline(beh_estimator, include_grades=False)
    behavioral_pipeline.fit(X_train_beh, y_train)

    y_pred_beh = behavioral_pipeline.predict(X_test_beh)
    beh_mae = float(mean_absolute_error(y_test, y_pred_beh))
    beh_rmse = float(root_mean_squared_error(y_test, y_pred_beh))
    beh_r2 = float(r2_score(y_test, y_pred_beh))

    beh_metrics = {
        "model_name": "Random Forest Regressor (Behavioral Only)",
        "features_excluded": ["G1", "G2"],
        "input_features_count": int(X_train_beh.shape[1]),
        "MAE": round(beh_mae, 4),
        "RMSE": round(beh_rmse, 4),
        "R2": round(beh_r2, 4),
        "description": "Early-semester predictive model using only behavioral, demographic, attendance, and lifestyle indicators prior to any grading period.",
    }
    logger.info("[Behavioral Model] MAE=%.4f  RMSE=%.4f  R²=%.4f", beh_mae, beh_rmse, beh_r2)
    print(f"[Behavioral Model] MAE: {beh_mae:.3f} | RMSE: {beh_rmse:.3f} | R²: {beh_r2:.3f}")

    # 6. Persist both trained model pipelines and metrics to disk
    os.makedirs(models_dir, exist_ok=True)
    full_model_path = os.path.join(models_dir, "student_model.pkl")
    beh_model_path = os.path.join(models_dir, "behavioral_model.pkl")
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")

    joblib.dump(best_pipeline, full_model_path)
    logger.info("Saved trained production pipeline → %s", full_model_path)
    print(f"Saved production pipeline to: {full_model_path}")

    joblib.dump(behavioral_pipeline, beh_model_path)
    logger.info("Saved trained behavioral pipeline → %s", beh_model_path)
    print(f"Saved behavioral pipeline to: {beh_model_path}")

    # Save complete metadata
    metadata = {
        "best_model_name": best_model_name,
        "test_size": test_size,
        "random_state": random_state,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "metrics": metrics,
        "cross_validation": cv_summary,
        "behavioral_model": beh_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved evaluation metrics → %s", metrics_path)
    print(f"Saved evaluation metrics to: {metrics_path}")

    return metrics, best_model_name


if __name__ == "__main__":
    train_and_evaluate_models()
