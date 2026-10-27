"""
Unit and Integration Tests for Student Performance Pipeline.

Verifies:
1. Data loading and CSV schema integrity.
2. Summary statistics and data health calculations.
3. Preprocessing ColumnTransformer behavior (scaling, one-hot encoding, missing values).
4. Data leakage prevention.
5. Model loading and inference.
6. Prediction range clamping (0 <= G3 <= 20).
7. Input boundary validation and exception handling.
8. Logging infrastructure (logger module, file creation, log records).
9. Error handling paths in data_loader and predict modules.
"""

import os
import sys
import logging
import pytest
import numpy as np
import pandas as pd

# Ensure project root is available to pytest
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, check_data_quality, get_summary_statistics
from src.preprocessor import (
    separate_features_target,
    build_preprocessor,
    create_full_pipeline,
    NUMERIC_FEATURES,
    BEHAVIORAL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from src.predict import (
    load_trained_model,
    load_behavioral_model,
    validate_and_prepare_input,
    predict_student_grade,
    categorize_grade,
)
from src.logger import get_logger
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# =====================================================================
# 1. DATA LOADING & INTEGRITY TESTS
# =====================================================================

def test_data_loading_success():
    """Verify CSV loads with authentic dimensions and target column."""
    df = load_data("data/student_performance.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 395, f"Expected 395 rows, found {df.shape[0]}"
    assert df.shape[1] == 33, f"Expected 33 columns, found {df.shape[1]}"
    assert TARGET_COLUMN in df.columns, f"Target '{TARGET_COLUMN}' must be present"


def test_data_loading_missing_file():
    """Verify FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_data("data/non_existent_dataset.csv")


def test_data_quality_metrics():
    """Verify dataset quality checks report zero missing values and zero duplicates."""
    df = load_data("data/student_performance.csv")
    quality = check_data_quality(df)

    assert quality["row_count"] == 395
    assert quality["column_count"] == 33
    assert quality["missing_values_total"] == 0, "Clean benchmark data must have 0 missing values"
    assert quality["duplicate_rows"] == 0, "No duplicate records should be present"


def test_summary_statistics_structure():
    """Verify summary statistics output contains required metrics."""
    df = load_data("data/student_performance.csv")
    num_stats, cat_summary = get_summary_statistics(df)

    assert isinstance(num_stats, pd.DataFrame)
    assert isinstance(cat_summary, dict)

    # Check key columns
    for col in ["G1", "G2", "G3"]:
        assert col in num_stats.index
        assert "mean" in num_stats.columns
        assert "median" in num_stats.columns
        assert "std" in num_stats.columns
        assert num_stats.loc[col, "min"] >= 0
        assert num_stats.loc[col, "max"] <= 20


# =====================================================================
# 2. PREPROCESSING & FEATURE PIPELINE TESTS
# =====================================================================

def test_feature_target_separation():
    """Verify X and y are properly separated without column overlap."""
    df = load_data("data/student_performance.csv")
    X, y = separate_features_target(df)

    assert TARGET_COLUMN not in X.columns
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y) == 395
    assert X.shape[1] == 32


def test_preprocessor_transformation_shape():
    """Verify preprocessor produces correct 2D array without NaNs."""
    df = load_data("data/student_performance.csv")
    X, _ = separate_features_target(df)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 395
    # 15 numeric + 43 one-hot categories = 58 columns
    assert transformed.shape[1] == 58
    assert not np.isnan(transformed).any(), "Transformed output must not contain NaNs"


def test_preprocessor_handles_unseen_categories():
    """Verify preprocessor handles unknown categorical values gracefully (no crash)."""
    df = load_data("data/student_performance.csv")
    X, _ = separate_features_target(df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    # Novel record with unseen categorical values
    unseen_record = X.iloc[:1].copy()
    unseen_record.loc[0, "Mjob"] = "astronaut"      # Unseen job
    unseen_record.loc[0, "school"] = "UNKNOWN_SCH"  # Unseen school

    # Should transform without raising an error due to handle_unknown='ignore'
    transformed = preprocessor.transform(unseen_record)
    assert transformed.shape == (1, 58)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_missing_values():
    """Verify SimpleImputer handles NaN values during transformation."""
    df = load_data("data/student_performance.csv")
    X, _ = separate_features_target(df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    # Test record with missing fields
    missing_record = X.iloc[:1].copy()
    missing_record.loc[0, "age"] = np.nan
    missing_record.loc[0, "G1"] = np.nan
    missing_record.loc[0, "schoolsup"] = np.nan

    transformed = preprocessor.transform(missing_record)
    assert transformed.shape == (1, 58)
    assert not np.isnan(transformed).any(), "Imputer must fill missing values"


# =====================================================================
# 3. MODEL TRAINING & PERSISTENCE TESTS
# =====================================================================

def test_full_pipeline_creation_and_fit():
    """Verify end-to-end pipeline fits and predicts without data leakage."""
    df = load_data("data/student_performance.csv")
    X, y = separate_features_target(df)

    pipeline = create_full_pipeline(LinearRegression())
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert isinstance(preds, np.ndarray)


def test_saved_model_file_exists_and_loads():
    """Verify models/student_model.pkl exists and can be loaded."""
    model_path = "models/student_model.pkl"
    assert os.path.exists(model_path), f"Saved model not found at {model_path}"

    model = load_trained_model(model_path)
    assert hasattr(model, "predict"), "Loaded model must have a predict method"


# =====================================================================
# 4. PREDICTION LOGIC & RANGE CONSTRAINTS (0-20)
# =====================================================================

@pytest.mark.parametrize(
    "profile",
    [
        {"G1": 20, "G2": 20, "studytime": 4, "absences": 0},  # Maximum performance
        {"G1": 0, "G2": 0, "studytime": 1, "absences": 50},   # Minimum performance
        {"G1": 10, "G2": 10, "studytime": 2, "absences": 5},  # Average performance
        {"G1": 15, "G2": 14, "studytime": 3, "absences": 2},  # Above average
    ],
)
def test_predict_grade_within_0_to_20_range(profile):
    """Verify that predictions strictly stay within 0.0 to 20.0 across edge cases."""
    result = predict_student_grade(profile)

    grade = result["predicted_grade"]
    assert 0.0 <= grade <= 20.0, f"Grade {grade} out of bounds [0, 20]"
    assert isinstance(grade, float)
    assert "tier" in result
    assert "risk_level" in result
    assert "advisory" in result


def test_grade_tier_categorization():
    """Verify the categorization tiers map correctly."""
    tier1, risk1, _ = categorize_grade(18.5)
    assert "Excellent" in tier1 and risk1 == "Low Risk"

    tier2, risk2, _ = categorize_grade(14.5)
    assert "Good" in tier2 and risk2 == "Low Risk"

    tier3, risk3, _ = categorize_grade(11.0)
    assert "Pass" in tier3 and risk3 == "Moderate Risk"

    tier4, risk4, _ = categorize_grade(5.0)
    assert "Needs Intervention" in tier4 and risk4 == "High Risk"


# =====================================================================
# 5. INPUT VALIDATION & EXCEPTION TESTS
# =====================================================================

def test_input_validation_invalid_g1():
    """Verify invalid G1 values raise ValueError."""
    with pytest.raises(ValueError, match="G1 grade must be between 0 and 20"):
        validate_and_prepare_input({"G1": 25})

    with pytest.raises(ValueError, match="G1 grade must be between 0 and 20"):
        validate_and_prepare_input({"G1": -2})


def test_input_validation_invalid_absences():
    """Verify invalid absences raise ValueError."""
    with pytest.raises(ValueError, match="Absences must be between 0 and 93"):
        validate_and_prepare_input({"absences": -5})

    with pytest.raises(ValueError, match="Absences must be between 0 and 93"):
        validate_and_prepare_input({"absences": 150})


def test_input_validation_invalid_studytime():
    """Verify invalid studytime raises ValueError."""
    with pytest.raises(ValueError, match="studytime must be 1, 2, 3, or 4"):
        validate_and_prepare_input({"studytime": 8})


def test_input_validation_invalid_type():
    """Verify non-dict / non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError, match="must be a dictionary or a Pandas DataFrame"):
        validate_and_prepare_input(["not", "a", "dict"])


# =====================================================================
# 6. LOGGING INFRASTRUCTURE TESTS  (NEW — U6)
# =====================================================================

def test_get_logger_returns_logger_instance():
    """Verify get_logger returns a properly named Logger object."""
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert "test_module" in logger.name


def test_logger_name_hierarchy():
    """Verify loggers follow the 'studentai.<module>' naming convention."""
    logger = get_logger("src.data_loader")
    assert logger.name.startswith("studentai.")


def test_logger_writes_to_file(tmp_path, monkeypatch):
    """
    Verify that a log record at INFO level is written to the log file.

    This test uses pytest's tmp_path fixture to create a fresh temporary
    directory, then monkeypatches the logger module's _LOG_DIR and _LOG_FILE
    so the test doesn't pollute the real outputs/logs/ directory.
    """
    import src.logger as logger_module

    # Point log output to a clean temp directory for this test
    tmp_log_dir = str(tmp_path / "logs")
    tmp_log_file = str(tmp_path / "logs" / "test_studentai.log")

    # Reset the configured flag so a fresh setup runs
    original_configured = logger_module._configured
    original_log_dir = logger_module._LOG_DIR
    original_log_file = logger_module._LOG_FILE

    logger_module._configured = False
    logger_module._LOG_DIR = tmp_log_dir
    logger_module._LOG_FILE = tmp_log_file

    try:
        test_logger = logger_module.get_logger("u6_test")
        test_logger.info("U6 test log entry written successfully.")

        assert os.path.exists(tmp_log_file), "Log file must be created after first log call."
        with open(tmp_log_file, encoding="utf-8") as f:
            content = f.read()
        assert "U6 test log entry written successfully." in content
    finally:
        # Restore original state
        logger_module._configured = original_configured
        logger_module._LOG_DIR = original_log_dir
        logger_module._LOG_FILE = original_log_file
        # Remove test handlers
        root_log = logging.getLogger("studentai.u6_test")
        root_log.handlers.clear()


def test_data_loader_logs_on_file_not_found(caplog):
    """Verify data_loader emits an ERROR log when file is missing."""
    import src.logger as logger_module
    logger_module._setup_root_logger()

    with caplog.at_level(logging.ERROR, logger="studentai"):
        with pytest.raises(FileNotFoundError):
            load_data("data/completely_missing_file.csv")

    error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("not found" in msg.lower() for msg in error_messages), (
        f"Expected an ERROR log about file not found. Got: {error_messages}"
    )


def test_predict_logs_warning_on_invalid_input(caplog):
    """Verify predict module emits a WARNING when an invalid G1 value is supplied."""
    import src.logger as logger_module
    logger_module._setup_root_logger()

    with caplog.at_level(logging.WARNING, logger="studentai"):
        with pytest.raises(ValueError):
            validate_and_prepare_input({"G1": 99})

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("G1" in msg or "99" in msg for msg in warning_messages), (
        f"Expected a WARNING log about invalid G1. Got: {warning_messages}"
    )


# =====================================================================
# 7. LOAD_TRAINED_MODEL ERROR PATH TEST  (NEW — U6)
# =====================================================================

def test_load_model_missing_file_raises_error():
    """Verify load_trained_model raises FileNotFoundError for a missing path."""
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        load_trained_model("models/definitely_does_not_exist.pkl")


# =====================================================================
# 8. ML PIPELINE ENHANCEMENT & BEHAVIORAL MODEL TESTS (NEW — U1)
# =====================================================================

def test_behavioral_numeric_features_excludes_grades():
    """Verify that BEHAVIORAL_NUMERIC_FEATURES contains no period grades."""
    assert "G1" not in BEHAVIORAL_NUMERIC_FEATURES
    assert "G2" not in BEHAVIORAL_NUMERIC_FEATURES
    assert len(BEHAVIORAL_NUMERIC_FEATURES) == 13
    assert len(NUMERIC_FEATURES) == 15


def test_behavioral_preprocessor_transformation_shape():
    """Verify preprocessor without grades produces correct 56-column transformed array."""
    df = load_data("data/student_performance.csv")
    X, _ = separate_features_target(df)
    X_beh = X.drop(columns=["G1", "G2"])
    preprocessor = build_preprocessor(include_grades=False)

    transformed = preprocessor.fit_transform(X_beh)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 395
    # 13 numeric + 43 one-hot categories = 56 columns
    assert transformed.shape[1] == 56
    assert not np.isnan(transformed).any(), "Transformed output must not contain NaNs"


def test_behavioral_pipeline_creation_and_fit():
    """Verify behavioral pipeline fits and predicts without grade features."""
    df = load_data("data/student_performance.csv")
    X, y = separate_features_target(df)
    X_beh = X.drop(columns=["G1", "G2"])

    pipeline = create_full_pipeline(LinearRegression(), include_grades=False)
    pipeline.fit(X_beh, y)
    preds = pipeline.predict(X_beh)

    assert len(preds) == len(y)
    assert isinstance(preds, np.ndarray)


def test_behavioral_model_file_exists_and_loads():
    """Verify models/behavioral_model.pkl exists and can be loaded."""
    model_path = "models/behavioral_model.pkl"
    assert os.path.exists(model_path), f"Saved behavioral model not found at {model_path}"

    model = load_behavioral_model(model_path)
    assert hasattr(model, "predict"), "Loaded behavioral model must have a predict method"


def test_predict_grade_with_behavioral_model():
    """Verify prediction using the behavioral model stays within 0 to 20 range."""
    beh_model = load_behavioral_model("models/behavioral_model.pkl")

    profile = {
        "studytime": 3,
        "absences": 4,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "higher": "yes",
    }
    result = predict_student_grade(profile, model=beh_model)

    grade = result["predicted_grade"]
    assert 0.0 <= grade <= 20.0, f"Predicted grade {grade} out of bounds"
    assert "tier" in result
    assert "risk_level" in result
    assert "advisory" in result


def test_gradient_boosting_pipeline_train_and_predict():
    """Verify GradientBoostingRegressor integrates into the pipeline cleanly."""
    df = load_data("data/student_performance.csv")
    X, y = separate_features_target(df)

    pipeline = create_full_pipeline(GradientBoostingRegressor(n_estimators=10, random_state=42))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert isinstance(preds, np.ndarray)


def test_evaluation_metrics_json_schema():
    """Verify evaluation_metrics.json has complete, verified schema."""
    import json
    metrics_path = "models/evaluation_metrics.json"
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check key root fields
    assert "best_model_name" in data
    assert data["best_model_name"] == "Random Forest Regressor"
    assert "metrics" in data
    assert "Gradient Boosting Regressor" in data["metrics"]
    assert "Random Forest Regressor" in data["metrics"]
    assert "Baseline (Dummy Mean)" in data["metrics"]

    # Check behavioral model metadata
    assert "behavioral_model" in data
    beh = data["behavioral_model"]
    assert beh["features_excluded"] == ["G1", "G2"]
    assert beh["input_features_count"] == 30
    assert 0.0 <= beh["R2"] <= 1.0
    assert beh["RMSE"] > 0

    # Check cross validation summary
    assert "cross_validation" in data
    assert "Random Forest Regressor" in data["cross_validation"]
    assert "Gradient Boosting Regressor" in data["cross_validation"]


# =====================================================================
# U2 TESTS: EXPLAINABILITY & INTERPRETABILITY
# =====================================================================


def test_get_pipeline_feature_names():
    """Verify feature name extraction for production and behavioral pipelines."""
    from src.explainability import get_pipeline_feature_names

    prod_model = load_trained_model()
    beh_model = load_behavioral_model()

    prod_features = get_pipeline_feature_names(prod_model)
    beh_features = get_pipeline_feature_names(beh_model)

    # 15 numeric + 43 one-hot = 58
    assert len(prod_features) == 58
    assert "G1" in prod_features
    assert "G2" in prod_features
    assert "absences" in prod_features
    assert "Mjob_services" in prod_features

    # 13 numeric + 43 one-hot = 56
    assert len(beh_features) == 56
    assert "G1" not in beh_features
    assert "G2" not in beh_features
    assert "absences" in beh_features
    assert "failures" in beh_features


def test_global_feature_importances_structure():
    """Verify global feature importance computation and schema."""
    from src.explainability import get_global_feature_importances

    prod_model = load_trained_model()
    df_top = get_global_feature_importances(prod_model, top_n=10)

    assert len(df_top) == 10
    assert list(df_top.columns) == ["feature", "importance", "display_name", "cumulative_share"]
    # Verify descending sort
    assert df_top["importance"].is_monotonic_decreasing
    # Verify non-negative
    assert (df_top["importance"] >= 0).all()

    # Full set should sum to 1.0
    df_all = get_global_feature_importances(prod_model, top_n=None)
    assert len(df_all) == 58
    assert np.isclose(df_all["importance"].sum(), 1.0, atol=1e-3)


def test_aggregated_feature_importances():
    """Verify aggregation of one-hot features back to parent attributes."""
    from src.explainability import get_aggregated_feature_importances

    prod_model = load_trained_model()
    df_agg = get_aggregated_feature_importances(prod_model, top_n=10)

    assert len(df_agg) == 10
    assert "parent_attribute" in df_agg.columns
    assert "importance_pct" in df_agg.columns
    assert "display_name" in df_agg.columns

    # Verify key parents exist
    parents = list(df_agg["parent_attribute"])
    assert "G2" in parents
    assert "absences" in parents

    # Full aggregated set sums to ~100%
    df_agg_all = get_aggregated_feature_importances(prod_model, top_n=None)
    assert np.isclose(df_agg_all["importance_pct"].sum(), 100.0, atol=1e-2)


def test_local_prediction_tree_path_exactness():
    """Verify that tree path decomposition exactly reconstructs model prediction."""
    from src.explainability import explain_prediction_tree_path

    prod_model = load_trained_model()
    test_profiles = [
        {"G1": 15, "G2": 16, "absences": 2, "studytime": 3},
        {"G1": 6, "G2": 7, "absences": 18, "failures": 2},
        {"G1": 10, "G2": 10, "absences": 4, "studytime": 2},
    ]

    for profile in test_profiles:
        result = explain_prediction_tree_path(prod_model, profile, top_k=5)
        assert result["supported"] is True
        assert result["is_exact"] is True
        assert result["discrepancy"] < 1e-4

        # Verify mathematical identity: base + sum(contributions) == raw_prediction
        raw_pred = result["predicted_grade"]
        recon_pred = result["reconstructed_grade"]
        base_val = result["base_value"]
        total_delta = result["all_contributions"]["contribution"].sum()

        assert np.isclose(recon_pred, raw_pred, atol=1e-4)
        assert np.isclose(base_val + total_delta, raw_pred, atol=1e-4)

        # Verify positive and negative partitions
        for item in result["positive_contributions"]:
            assert item["contribution"] > 0
        for item in result["negative_contributions"]:
            assert item["contribution"] < 0


def test_local_prediction_behavioral_model():
    """Verify local explanations on the early-semester behavioral model."""
    from src.explainability import explain_prediction_tree_path

    beh_model = load_behavioral_model()
    profile = {"absences": 20, "failures": 3, "studytime": 1}

    result = explain_prediction_tree_path(beh_model, profile, top_k=4)
    assert result["supported"] is True
    assert result["is_exact"] is True

    # Ensure G1 and G2 do not appear anywhere in behavioral attributions
    all_feats = list(result["all_contributions"]["feature"])
    assert "G1" not in all_feats
    assert "G2" not in all_feats


def test_explainability_handles_invalid_input_gracefully():
    """Verify input validation triggers appropriate errors."""
    from src.explainability import explain_prediction_tree_path

    prod_model = load_trained_model()
    with pytest.raises(ValueError, match="G1 grade must be between 0 and 20"):
        explain_prediction_tree_path(prod_model, {"G1": 999})


# =====================================================================
# U4 TESTS: BATCH CSV INGESTION, VALIDATION & INFERENCE
# =====================================================================


def test_batch_sample_csv_generation():
    """Verify sample CSV template generation schema and parseability."""
    import io
    from src.batch_predictor import generate_sample_batch_csv

    csv_text = generate_sample_batch_csv(5)
    assert isinstance(csv_text, str)
    assert len(csv_text) > 100

    df_parsed = pd.read_csv(io.StringIO(csv_text))
    assert len(df_parsed) == 5
    assert "student_id" in df_parsed.columns
    assert "G1" in df_parsed.columns
    assert "G2" in df_parsed.columns
    assert "studytime" in df_parsed.columns
    assert "absences" in df_parsed.columns


def test_batch_prediction_valid_cohort():
    """Verify batch prediction processes valid records cleanly."""
    import io
    from src.batch_predictor import generate_sample_batch_csv, predict_batch

    df = pd.read_csv(io.StringIO(generate_sample_batch_csv(4)))
    results_df, summary = predict_batch(df)

    assert summary["total_records"] == 4
    assert summary["valid_predictions"] == 4
    assert summary["failed_validations"] == 0
    assert 0.0 <= summary["mean_predicted_grade"] <= 20.0

    # Output dataframe structure
    assert "status" in results_df.columns
    assert (results_df["status"] == "SUCCESS").all()
    assert "predicted_grade" in results_df.columns
    assert "performance_tier" in results_df.columns
    assert "risk_level" in results_df.columns
    assert "student_id" in results_df.columns

    for grade in results_df["predicted_grade"]:
        assert 0.0 <= grade <= 20.0


def test_batch_prediction_empty_dataframe_raises():
    """Verify empty DataFrame triggers ValueError."""
    from src.batch_predictor import predict_batch

    with pytest.raises(ValueError, match="uploaded CSV file is empty"):
        predict_batch(pd.DataFrame())


def test_batch_prediction_missing_columns_raises():
    """Verify missing required columns triggers ValueError listing the missing columns."""
    from src.batch_predictor import predict_batch

    df_incomplete = pd.DataFrame([{"student_id": "STU_1", "age": 16, "studytime": 2}])
    with pytest.raises(ValueError, match="missing .* required column"):
        predict_batch(df_incomplete)


def test_batch_prediction_with_invalid_rows():
    """Verify invalid rows are flagged and valid rows are predicted."""
    import io
    from src.batch_predictor import generate_sample_batch_csv, predict_batch

    df = pd.read_csv(io.StringIO(generate_sample_batch_csv(3)))

    # Corrupt row 1: out-of-range G1
    df.loc[1, "G1"] = 99
    # Corrupt row 2: invalid categorical value for sex
    df.loc[2, "sex"] = "NonBinary"

    results_df, summary = predict_batch(df)

    assert summary["total_records"] == 3
    assert summary["valid_predictions"] == 1
    assert summary["failed_validations"] == 2

    # Row 0 is valid
    assert results_df.loc[0, "status"] == "SUCCESS"
    assert not pd.isna(results_df.loc[0, "predicted_grade"])

    # Row 1 has G1 validation failure
    assert results_df.loc[1, "status"] == "VALIDATION_FAILED"
    assert "G1" in results_df.loc[1, "validation_error"]
    assert pd.isna(results_df.loc[1, "predicted_grade"])

    # Row 2 has sex validation failure
    assert results_df.loc[2, "status"] == "VALIDATION_FAILED"
    assert "sex" in results_df.loc[2, "validation_error"]
    assert pd.isna(results_df.loc[2, "predicted_grade"])


def test_batch_prediction_behavioral_mode():
    """Verify batch prediction in behavioral mode (excluding G1/G2)."""
    import io
    from src.batch_predictor import generate_sample_batch_csv, predict_batch

    df = pd.read_csv(io.StringIO(generate_sample_batch_csv(3)))
    df_no_grades = df.drop(columns=["G1", "G2"])

    results_df, summary = predict_batch(df_no_grades, use_behavioral=True)

    assert summary["valid_predictions"] == 3
    assert summary["failed_validations"] == 0
    assert (results_df["status"] == "SUCCESS").all()


# =====================================================================
# U5 TESTS: INDIVIDUAL STUDENT REPORT & PDF EXPORT
# =====================================================================


def test_generate_student_pdf_report_success():
    """Verify that a valid student PDF report is generated with correct magic bytes."""
    from src.predict import load_trained_model, predict_student_grade, DEFAULT_STUDENT_PROFILE
    from src.explainability import explain_prediction_tree_path
    from src.report_generator import generate_student_pdf_report

    model = load_trained_model()
    student = dict(DEFAULT_STUDENT_PROFILE, G1=15, G2=16, studytime=3, absences=2)
    pred_res = predict_student_grade(student, model=model)
    expl_res = explain_prediction_tree_path(model, student)

    pdf_bytes = generate_student_pdf_report(
        student_profile=student,
        prediction_result=pred_res,
        explanation_result=expl_res,
        model_name="Production Model (Full History)",
        student_identifier="TEST_STU_001",
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 10000
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_student_pdf_report_behavioral():
    """Verify PDF report generation for the behavioral model."""
    from src.predict import load_behavioral_model, predict_student_grade, DEFAULT_STUDENT_PROFILE
    from src.explainability import explain_prediction_tree_path
    from src.report_generator import generate_student_pdf_report

    model = load_behavioral_model()
    student = dict(DEFAULT_STUDENT_PROFILE, absences=18, failures=2, studytime=1)
    pred_res = predict_student_grade(student, model=model)
    expl_res = explain_prediction_tree_path(model, student)

    pdf_bytes = generate_student_pdf_report(
        student_profile=student,
        prediction_result=pred_res,
        explanation_result=expl_res,
        model_name="Early-Semester Behavioral Model",
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 10000
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_student_pdf_report_without_explanation():
    """Verify PDF report degrades gracefully when explanation_result is None."""
    from src.predict import load_trained_model, predict_student_grade, DEFAULT_STUDENT_PROFILE
    from src.report_generator import generate_student_pdf_report

    model = load_trained_model()
    student = dict(DEFAULT_STUDENT_PROFILE)
    pred_res = predict_student_grade(student, model=model)

    pdf_bytes = generate_student_pdf_report(
        student_profile=student,
        prediction_result=pred_res,
        explanation_result=None,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_student_pdf_report_invalid_inputs_raise():
    """Verify invalid inputs raise appropriate ValueErrors."""
    from src.report_generator import generate_student_pdf_report

    with pytest.raises(ValueError, match="student_profile must be a non-empty dictionary"):
        generate_student_pdf_report({}, {"predicted_grade": 12.0})

    with pytest.raises(ValueError, match="prediction_result must contain 'predicted_grade'"):
        generate_student_pdf_report({"age": 16}, {})



