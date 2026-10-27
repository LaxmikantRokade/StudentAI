"""
StudentAI — Explainability & Interpretability Module (U2)
=========================================================
Provides both global and local feature explanations for student grade predictions
using mathematically rigorous, dependency-free methods supported by Scikit-learn
and NumPy.

Explanations Supported:
1. Global Feature Importance:
   - Mean Decrease in Impurity (MDI) across ensemble decision trees.
   - Raw transformed feature importances (58 features).
   - Parent feature aggregations (32 parent attributes) to combine one-hot dummies.
   - Clear educational disclaimers regarding MDI limitations (split frequency vs. causation).

2. Local Prediction Attribution (Per-Student Explanation):
   - Exact Tree Path Contribution Decomposition (Saabas algorithm):
     Each individual decision tree decomposes a student's score from the
     training population baseline (root node mean) through incremental
     feature-split deltas:
         y_pred(x) = baseline + sum_j(contribution_j(x))
   - Separates net positive drivers (lifting score) and net negative drivers (lowering score).
   - Verifies exact numerical reconstruction against model inference.

Author: StudentAI Intelligence Engineering Team
Lab: 3rd Year B.Tech IT — Data Science & Visualization
"""

import sys
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.logger import get_logger
from src.predict import validate_and_prepare_input, DEFAULT_STUDENT_PROFILE

logger = get_logger("explainability")

# Human-friendly descriptions and labels for dataset attributes
FEATURE_DISPLAY_NAMES = {
    "G2": "Second Period Grade (G2)",
    "G1": "First Period Grade (G1)",
    "absences": "School Absences",
    "failures": "Past Class Failures",
    "studytime": "Weekly Study Time",
    "age": "Student Age",
    "Medu": "Mother's Education Level",
    "Fedu": "Father's Education Level",
    "health": "Current Health Status",
    "goout": "Going Out with Friends",
    "Dalc": "Workday Alcohol Intake",
    "Walc": "Weekend Alcohol Intake",
    "freetime": "Free Time After School",
    "famrel": "Family Relationship Quality",
    "traveltime": "Home-to-School Travel Time",
    "schoolsup": "School Academic Support",
    "famsup": "Family Academic Support",
    "paid": "Extra Paid Tutoring",
    "activities": "Extra-Curricular Activities",
    "nursery": "Attended Nursery School",
    "higher": "Aspires to Higher Education",
    "internet": "Home Internet Access",
    "romantic": "In a Romantic Relationship",
    "school": "School Attended",
    "sex": "Student Gender",
    "address": "Living Address Type",
    "famsize": "Family Size",
    "Pstatus": "Parent Cohabitation Status",
    "Mjob": "Mother's Occupation",
    "Fjob": "Father's Occupation",
    "reason": "School Choice Reason",
    "guardian": "Primary Guardian",
}


def get_pipeline_feature_names(pipeline: Pipeline) -> List[str]:
    """
    Extracts the ordered list of transformed feature names produced by the
    pipeline's ColumnTransformer preprocessor.

    Parameters:
        pipeline (Pipeline): Fitted Scikit-learn Pipeline with a ColumnTransformer.

    Returns:
        List[str]: 58 feature names (numeric + one-hot encoded categories).
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline does not contain a 'preprocessor' step.")

    # 1. Numeric feature names
    num_cols = list(preprocessor.transformers_[0][2])

    # 2. Categorical one-hot feature names
    cat_cols = list(preprocessor.transformers_[1][2])
    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot_encoder = cat_pipeline.named_steps["onehot"]
    cat_feature_names = list(onehot_encoder.get_feature_names_out(cat_cols))

    return num_cols + cat_feature_names


def get_global_feature_importances(
    pipeline: Pipeline,
    top_n: Optional[int] = 15,
) -> pd.DataFrame:
    """
    Computes global feature importances based on Gini/Mean Decrease in Impurity (MDI)
    from the fitted ensemble regressor.

    Parameters:
        pipeline (Pipeline): Fitted Scikit-learn Pipeline containing a tree regressor.
        top_n (int, optional): Number of top features to return. If None, returns all.

    Returns:
        pd.DataFrame: Columns ['feature', 'display_name', 'importance', 'cumulative_share'].
    """
    regressor = pipeline.named_steps.get("regressor")
    if regressor is None:
        raise ValueError("Pipeline does not contain a 'regressor' step.")

    if not hasattr(regressor, "feature_importances_"):
        logger.warning("Regressor %s does not provide feature_importances_ attribute.", type(regressor))
        raise AttributeError(f"Model {type(regressor).__name__} does not support Gini feature importances.")

    feature_names = get_pipeline_feature_names(pipeline)
    raw_importances = regressor.feature_importances_

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": raw_importances,
    })

    # Add display names
    def _map_display_name(feat: str) -> str:
        if feat in FEATURE_DISPLAY_NAMES:
            return FEATURE_DISPLAY_NAMES[feat]
        # Check if it's a one-hot column like 'Mjob_services' or 'schoolsup_yes'
        for parent_col in FEATURE_DISPLAY_NAMES:
            if feat.startswith(f"{parent_col}_"):
                suffix = feat[len(parent_col) + 1 :]
                parent_display = FEATURE_DISPLAY_NAMES[parent_col]
                return f"{parent_display} ({suffix})"
        return feat

    df_imp["display_name"] = df_imp["feature"].apply(_map_display_name)
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["cumulative_share"] = df_imp["importance"].cumsum() / df_imp["importance"].sum()

    if top_n is not None and top_n > 0:
        return df_imp.head(top_n).copy()
    return df_imp


def get_aggregated_feature_importances(
    pipeline: Pipeline,
    top_n: Optional[int] = 12,
) -> pd.DataFrame:
    """
    Aggregates one-hot dummy importances back to their parent domain attributes
    (e.g., combines Mjob_teacher, Mjob_health, etc. under 'Mjob') to provide
    a high-level, human-interpretable importance ranking.

    Parameters:
        pipeline (Pipeline): Fitted Scikit-learn Pipeline.
        top_n (int, optional): Number of top attributes to return.

    Returns:
        pd.DataFrame: Columns ['attribute', 'display_name', 'importance', 'importance_pct'].
    """
    df_all = get_global_feature_importances(pipeline, top_n=None)

    # Determine parent attribute
    def _get_parent_attribute(feat: str) -> str:
        # Check known categorical features
        for parent in FEATURE_DISPLAY_NAMES:
            if feat == parent or feat.startswith(f"{parent}_"):
                return parent
        return feat

    df_all["parent_attribute"] = df_all["feature"].apply(_get_parent_attribute)
    df_agg = (
        df_all.groupby("parent_attribute")["importance"]
        .sum()
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    df_agg["display_name"] = df_agg["parent_attribute"].map(
        lambda x: FEATURE_DISPLAY_NAMES.get(x, x)
    )
    total_imp = df_agg["importance"].sum()
    df_agg["importance_pct"] = (df_agg["importance"] / total_imp) * 100.0

    if top_n is not None and top_n > 0:
        return df_agg.head(top_n).copy()
    return df_agg


def explain_prediction_tree_path(
    pipeline: Pipeline,
    student_profile: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Computes exact local prediction attribution for an individual student profile
    using tree path decomposition (Saabas algorithm) across the ensemble estimators.

    Mathematical Formulation:
        For each decision tree t in {1, ..., T}:
            Let node_0, node_1, ..., node_L be the sample's decision path from root to leaf.
            root_value = tree_t.value[0]
            For each step k:
                feature_j = tree_t.feature[node_k]
                delta_j = tree_t.value[node_{k+1}] - tree_t.value[node_k]
                tree_contrib_t[feature_j] += delta_j
        
        baseline = (1 / T) * sum_{t=1}^T (root_value_t)
        contribution_j = (1 / T) * sum_{t=1}^T (tree_contrib_{t, j})
        reconstructed_grade = baseline + sum_j (contribution_j)

    Guarantees:
        reconstructed_grade == pipeline.predict(student)[0] (within floating point precision).

    Parameters:
        pipeline (Pipeline): Fitted Scikit-learn Pipeline with a tree ensemble regressor.
        student_profile (dict): Key-value attributes for the student.
        top_k (int): Number of top positive and negative drivers to highlight.

    Returns:
        dict: Complete local explanation package including baseline, contributions,
              top positive drivers, top negative drivers, and verification status.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    regressor = pipeline.named_steps.get("regressor")

    if preprocessor is None or regressor is None:
        raise ValueError("Pipeline must contain both 'preprocessor' and 'regressor' steps.")

    # Validate and prepare student input
    df_clean = validate_and_prepare_input(student_profile)

    # Transform through pipeline ColumnTransformer
    X_trans = preprocessor.transform(df_clean)
    if issparse(X_trans):
        X_trans = X_trans.toarray()

    feature_names = get_pipeline_feature_names(pipeline)
    n_features = X_trans.shape[1]

    # Verify regressor supports tree path decomposition (RandomForest, ExtraTrees, etc.)
    if not hasattr(regressor, "estimators_"):
        # Graceful fallback for non-ensemble models (e.g., Ridge, LinearRegression)
        logger.warning(
            "Regressor %s does not contain 'estimators_'. Tree path decomposition unavailable.",
            type(regressor),
        )
        pred = float(regressor.predict(X_trans)[0])
        return {
            "supported": False,
            "method": "Linear Fallback / Unsupported",
            "predicted_grade": pred,
            "message": f"Tree decomposition is only applicable to tree ensemble models. Current model: {type(regressor).__name__}.",
        }

    estimators = regressor.estimators_
    n_trees = len(estimators)

    contributions = np.zeros(n_features, dtype=np.float64)
    biases = np.zeros(n_trees, dtype=np.float64)

    # Traverse each tree along the student's exact decision path
    for i, tree_est in enumerate(estimators):
        tree = tree_est.tree_
        indicator = tree_est.decision_path(X_trans)
        nodes = indicator.indices[indicator.indptr[0] : indicator.indptr[1]]

        # Root node value represents the tree's population expected value
        biases[i] = tree.value[0][0][0]

        for k in range(len(nodes) - 1):
            curr_n = nodes[k]
            next_n = nodes[k + 1]
            split_feat = tree.feature[curr_n]
            # Contribution is the shift in prediction induced by the split test
            step_delta = tree.value[next_n][0][0] - tree.value[curr_n][0][0]
            contributions[split_feat] += step_delta

    base_value = float(biases.mean())
    mean_contributions = contributions / n_trees
    raw_pred = float(regressor.predict(X_trans)[0])
    reconstructed_pred = float(base_value + mean_contributions.sum())
    is_exact = bool(np.isclose(raw_pred, reconstructed_pred, atol=1e-4))

    # Build DataFrame of all feature contributions
    df_contrib = pd.DataFrame({
        "feature": feature_names,
        "contribution": mean_contributions,
    })

    def _get_val_and_label(feat: str) -> Tuple[Any, str]:
        # Return student's actual value for this feature and display label
        if feat in df_clean.columns:
            val = df_clean[feat].iloc[0]
            label = FEATURE_DISPLAY_NAMES.get(feat, feat)
            return val, f"{label} = {val}"
        # Check one-hot feature
        for parent in df_clean.columns:
            if feat.startswith(f"{parent}_"):
                category = feat[len(parent) + 1 :]
                parent_val = df_clean[parent].iloc[0]
                is_active = (str(parent_val) == str(category))
                parent_display = FEATURE_DISPLAY_NAMES.get(parent, parent)
                val_repr = "Yes" if is_active else "No"
                return val_repr, f"{parent_display}: {category} ({val_repr})"
        return None, feat

    val_label_pairs = [_get_val_and_label(f) for f in df_contrib["feature"]]
    df_contrib["student_value"] = [p[0] for p in val_label_pairs]
    df_contrib["display_text"] = [p[1] for p in val_label_pairs]
    df_contrib["display_name"] = [
        FEATURE_DISPLAY_NAMES.get(f, f) for f in df_contrib["feature"]
    ]

    # Partition into positive and negative drivers
    df_pos = (
        df_contrib[df_contrib["contribution"] > 1e-4]
        .sort_values("contribution", ascending=False)
        .head(top_k)
    )
    df_neg = (
        df_contrib[df_contrib["contribution"] < -1e-4]
        .sort_values("contribution", ascending=True)
        .head(top_k)
    )

    positive_list = [
        {
            "feature": row["feature"],
            "display_name": row["display_name"],
            "display_text": row["display_text"],
            "student_value": row["student_value"],
            "contribution": float(row["contribution"]),
        }
        for _, row in df_pos.iterrows()
    ]

    negative_list = [
        {
            "feature": row["feature"],
            "display_name": row["display_name"],
            "display_text": row["display_text"],
            "student_value": row["student_value"],
            "contribution": float(row["contribution"]),
        }
        for _, row in df_neg.iterrows()
    ]

    return {
        "supported": True,
        "method": "Tree Path Contribution Decomposition (Saabas Algorithm)",
        "base_value": base_value,
        "predicted_grade": raw_pred,
        "reconstructed_grade": reconstructed_pred,
        "is_exact": is_exact,
        "discrepancy": abs(raw_pred - reconstructed_pred),
        "positive_contributions": positive_list,
        "negative_contributions": negative_list,
        "all_contributions": df_contrib.sort_values(
            "contribution", key=abs, ascending=False
        ).reset_index(drop=True),
        "disclaimer": (
            "Attributions reflect the internal split mechanics of the trained Random Forest "
            "ensemble relative to the population mean. They quantify statistical feature associations "
            "in this dataset and must NOT be interpreted as causal guarantees or physiological claims."
        ),
    }
