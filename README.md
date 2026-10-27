# Student Performance Prediction and Analytics System

> **Course:** Third Year B.Tech Information Technology  
> **Subject:** Data Science & Visualization Lab  
> **Dataset:** UCI Student Performance Dataset (Cortez & Silva, 2008)  
> **Target Variable:** `G3` — Final Mathematics Grade (0 – 20)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [Project Structure](#3-project-structure)
4. [Dataset Documentation](#4-dataset-documentation)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [Verified Model Results](#6-verified-model-results)
7. [Running the Dashboard](#7-running-the-dashboard)
8. [Running Tests](#8-running-tests)
9. [Limitations & Ethical Considerations](#9-limitations--ethical-considerations)
10. [Practical Lab Demonstration Guide](#10-practical-lab-demonstration-guide)
11. [Viva Examination Q&A](#11-viva-examination-qa)

---

## 1. Project Overview

This system predicts a secondary school student's final Mathematics grade (`G3`, scale 0–20) using demographic, academic history, and lifestyle features from the UCI benchmark dataset. It is built as a practical college lab project demonstrating the full data science workflow:

```
Raw CSV Data → Preprocessing Pipeline → Regression Model → Interactive Dashboard
```

**Key Objectives:**
- Identify academic factors most associated with final grade outcomes.
- Predict final marks as early as possible to enable targeted academic support.
- Present results in an accessible, student-friendly Streamlit dashboard.
- Demonstrate clean, tested, reproducible machine learning practices.

> **Important:** Predictions are statistical estimates based on historical patterns. They must not be used as the sole basis for any decision about a student's academic future.

---

## 2. Quick Start

### Prerequisites

- macOS (or Linux/Windows with minor path adjustments)
- Python 3.10 or later (tested on Python 3.13.7)
- No system-level dependencies required

### Step-by-Step Setup

**Open Terminal, navigate to the project folder:**

```bash
cd "/Users/apple/Documents/Projects/StudentAI 🔥"
```

**Create and activate the virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **What is a virtual environment?** It is an isolated Python sandbox for this project. Libraries installed here do not affect the rest of your Mac.

**Install all required packages:**

```bash
pip install -r requirements.txt
```

**Verify the installation (optional sanity check):**

```bash
python -c "import pandas, sklearn, streamlit, seaborn; print('All OK')"
```

**Run the full test suite:**

```bash
python -m pytest -v
```

**Launch the Streamlit dashboard:**

```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## 3. Project Structure

```
StudentAI/
├── app.py                         Streamlit web dashboard (4-tab interactive UI)
├── requirements.txt               Pinned project dependencies
├── README.md                      This file
├── .gitignore                     Git ignore rules
│
├── data/
│   ├── student_performance.csv    Primary dataset (395 rows × 33 cols)
│   ├── dataset_info.md            Data dictionary and provenance documentation
│   └── raw/                       Unmodified original UCI files (preserved)
│       ├── student-mat.csv        UCI Mathematics course (semicolon-separated)
│       ├── student-por.csv        UCI Portuguese course
│       └── student.txt            Original UCI feature descriptions
│
├── models/
│   ├── student_model.pkl          Serialized Scikit-learn Pipeline (best model)
│   └── evaluation_metrics.json   Verified test-set metrics for all 4 models
│
├── src/
│   ├── __init__.py
│   ├── logger.py                  Centralised logging (console + rotating file)
│   ├── data_loader.py             Dataset loading, quality checks, statistics
│   ├── preprocessor.py            ColumnTransformer pipeline (scaling + encoding)
│   ├── train.py                   Model training, evaluation, and artifact saving
│   └── predict.py                 Input validation, inference, grade categorization
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py           Automated unit and integration tests (28 tests)
│
└── outputs/
    ├── figures/                   Exported chart images (generated at runtime)
    └── logs/
        └── studentai.log          Application log file (generated at runtime)
```

---

## 4. Dataset Documentation

### Source

| Attribute | Value |
|:----------|:------|
| Name | Student Performance Data Set |
| Repository | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/320/student+performance |
| Authors | P. Cortez and A. Silva (2008) |
| License | Creative Commons Attribution 4.0 International |

**Citation:**
> Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. *Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008)*, pp. 5–12, Porto, Portugal.

### Key Facts

- **395 student records** from two Portuguese secondary schools (2005–2006)
- **33 attributes:** 15 numeric features, 17 binary/categorical features, 1 numeric target
- **Target:** `G3` (final Mathematics grade, 0–20 Portuguese scale)
- **Zero missing values;** zero duplicate rows
- **38 students (9.6%) scored 0** — this represents withdrawal, not failure to answer

### Important Note on G1, G2, and Target Dominance

`G2` (second period grade) has a correlation of **r = 0.905** with `G3`. This is **not data leakage** — the UCI study explicitly uses prior period grades as predictors for end-of-year forecasting. However, users should understand that the model's high R² is largely explained by this single feature. Behavioral indicators (study time, absences, family support) provide additional signal but have smaller individual effects.

---

## 5. Machine Learning Pipeline

### Architecture

```
CSV Load → Feature/Target Split → 80/20 Train/Test Split (seed=42)
    ↓
ColumnTransformer:
    Numeric (15 cols)    → SimpleImputer(median) → StandardScaler
    Categorical (17 cols) → SimpleImputer(mode)   → OneHotEncoder(handle_unknown='ignore')
    ↓
Regressor (RandomForestRegressor or GradientBoostingRegressor)
    ↓
Prediction Clamping: output = max(0.0, min(20.0, raw_pred))
```

### Data Leakage Prevention

The `ColumnTransformer` is **never** fitted on the full dataset. It is wrapped in a `sklearn.Pipeline` and fitted exclusively on `X_train` (316 samples). The scaling means (μ), standard deviations (σ), and one-hot vocabulary are derived from training data only. The 79 test samples are only transformed, never used to fit any parameter.

### Feature Groups

| Group | Features |
|:------|:---------|
| **Academic History** | `G1`, `G2`, `failures`, `absences` |
| **Study Habits** | `studytime`, `traveltime` |
| **Family & Support** | `Medu`, `Fedu`, `Mjob`, `Fjob`, `famsize`, `Pstatus`, `schoolsup`, `famsup`, `paid` |
| **Lifestyle** | `goout`, `Dalc`, `Walc`, `health`, `romantic`, `freetime` |
| **Demographics** | `school`, `sex`, `age`, `address`, `guardian`, `reason`, `nursery`, `activities`, `higher`, `internet` |

---

## 6. Verified Model Results

All metrics were computed on the **79 unseen test samples** (20% hold-out, `random_state=42`). These numbers were independently reproduced during the Phase 1 audit.

| Model | MAE | RMSE | R² |
|:------|:---:|:----:|:--:|
| Baseline — Dummy Mean | 3.646 | 4.550 | −0.010 |
| Linear Regression | 1.647 | 2.378 | 0.724 |
| Ridge Regression (α=1.0) | 1.639 | 2.372 | 0.726 |
| **Random Forest Regressor** *(saved)* | **1.186** | **1.971** | **0.811** |

**5-Fold Cross-Validation (Random Forest):**
- Mean R² = 0.830 ± 0.033 (stable across folds)
- Mean RMSE = 1.826 ± 0.336

The Random Forest reduces baseline RMSE by **57%** and explains **81.1%** of final-grade variance on unseen test data.

---

## 7. Running the Dashboard

```bash
# Activate virtual environment first
source .venv/bin/activate

# Launch Streamlit (opens browser automatically)
streamlit run app.py

# Or run without activating:
.venv/bin/python -m streamlit run app.py
```

The dashboard runs locally at `http://localhost:8501`. **No data is sent outside your machine.**

### Dashboard Tabs

| Tab | Contents |
|:----|:---------|
| 📋 Project Overview | Context, system architecture, dataset preview, key metrics |
| 📊 EDA | Grade KDE distributions, correlation heatmap, study time boxplot, absences trendline, statistics table |
| ⚙️ Model Evaluation | Benchmark comparison table, actual vs. predicted scatter, residual distribution |
| 🎯 Grade Predictor | Student input form, instant grade estimate, tier, risk advisory, disclaimer |

---

## 8. Running Tests

```bash
# Run all tests with verbose output
.venv/bin/python -m pytest -v

# Run a specific test by name
.venv/bin/python -m pytest -v -k "test_logger"

# Run tests and show coverage summary (requires pytest-cov if installed)
.venv/bin/python -m pytest -v --tb=short
```

**Current test suite:** 28 automated tests across 7 categories (data loading, preprocessing, model, prediction, tiers, logging, error paths).

---

## 9. Limitations & Ethical Considerations

1. **Context specificity:** Data originates from two Portuguese schools, 2005–2006. Results may not generalize to modern, international, or higher-education contexts.
2. **Self-reported features:** Alcohol consumption, family relationships, and study hours were collected via questionnaire and may contain social desirability bias.
3. **Small dataset:** 395 records is small by modern standards. Predictions carry meaningful uncertainty (RMSE ≈ 1.97 grade points on a 0–20 scale).
4. **Advisory use only:** The system flags statistical patterns, not individual student capability. Teachers, counselors, and parents must apply professional judgment alongside any model output.
5. **No causal claims:** Correlation between features and grades does not imply that changing one feature (e.g., reducing absences) will directly cause a grade improvement. External factors not captured in the data may be the true driving variables.

---

## 10. Practical Lab Demonstration Guide

Use this guide during your external practical examination.

### Recommended Demonstration Order (15 minutes)

1. **Open Terminal** → Show project folder structure using `ls`.
2. **Run tests** → `python -m pytest -v` → Point out all passing, explain 1 test category.
3. **Open dashboard** → `streamlit run app.py`.
4. **Tab 1 (Overview)** → Explain dataset origin, shape, target variable, and system goals.
5. **Tab 2 (EDA)** → Point out the correlation heatmap. Explain G1/G2 correlation with G3.
6. **Tab 3 (Evaluation)** → Show the metric table. Explain what MAE, RMSE, and R² mean. Point out the baseline comparison — this proves the model adds value.
7. **Tab 4 (Predictor)** → Enter two contrasting profiles:
   - *High achiever:* G1=17, G2=18, studytime=4, absences=0, failures=0.
   - *At-risk student:* G1=5, G2=6, studytime=1, absences=20, failures=2.
   - Read the advisory aloud and explain the disclaimer.

---

## 11. Viva Examination Q&A

### Q1. What is the target variable? Why did you choose regression and not classification?

**Answer:** The target variable `G3` is a continuous integer grade from 0 to 20. We chose **regression** because the output is a numerical value on a continuous scale, not a discrete category label. Regression allows us to predict the exact expected mark rather than just a coarse "pass/fail" bucket. We evaluate it using MAE, RMSE, and R².

---

### Q2. What is data leakage and how did you prevent it?

**Answer:** Data leakage occurs when information from the test set influences the training process — for example, if you compute scaling parameters (mean, standard deviation) across the entire dataset before splitting it. This would artificially inflate accuracy because the model has "seen" test data. We prevented it by wrapping the `ColumnTransformer` inside a `sklearn.Pipeline`. When `pipeline.fit(X_train, y_train)` is called, scaling parameters are derived **only** from the 316 training samples. The 79 test samples are transformed using those parameters but never used to fit them.

---

### Q3. What is the difference between MAE and RMSE? Which is more important here?

**Answer:**
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual grades. In plain English: "On average, we are off by X grade points."
- **RMSE (Root Mean Squared Error):** Like MAE but it squares the errors before averaging, then takes the square root. This **penalises large mistakes more heavily** than small ones.

For academic grading, RMSE is more important because a prediction that is off by 8 marks is far more harmful (a student may be incorrectly labelled as failing) than one that is off by 2 marks. We used RMSE as the model selection criterion.

---

### Q4. Why is G2 so strongly correlated with G3? Is that a problem?

**Answer:** G2 is the second-term evaluation; G3 is the final exam. Students who performed well in G2 typically also perform well in G3 — the same study habits, intelligence, and engagement drive both. The UCI study intentionally uses G1 and G2 as predictors because this mirrors how a real school counselor would use mid-year data to forecast final results. This is **not** data leakage. However, it does mean behavioral features (study time, absences) have smaller marginal impact once grade history is known.

---

### Q5. What is StandardScaler and why is it needed?

**Answer:** `StandardScaler` transforms each numeric feature to have **zero mean and unit standard deviation**: `z = (x − μ) / σ`. Without scaling, a feature like `absences` (0–93) would numerically dominate a feature like `Medu` (0–4), and distance-based or linear models would be biased toward features with larger numeric ranges. Random Forest does not require scaling mathematically, but keeping it in the pipeline ensures consistency when the pipeline is deployed or swapped with a different algorithm.

---

### Q6. What is OneHotEncoder and why did you use it instead of LabelEncoder?

**Answer:** `OneHotEncoder` converts a categorical variable with `k` categories into `k` binary (0/1) columns — one per category. `LabelEncoder` converts categories to integers (e.g., "yes"=1, "no"=0), which introduces an artificial ordinal relationship (implying "yes" is "greater than" "no"). For nominal categories like `Mjob` or `school`, no such ordering exists. `OneHotEncoder` avoids this false ordering. We set `handle_unknown='ignore'` so unseen categories during prediction do not cause a crash.

---

### Q7. What is cross-validation and why is it better than a single train/test split?

**Answer:** A single 80/20 split evaluates the model on just one random subset of 79 students. If that particular group happens to be easy or hard to predict, our metrics are misleading. **5-Fold Cross-Validation** splits the dataset into 5 groups; the model is trained 5 times, each time using 4 groups for training and 1 for testing. The final metric is the average across all 5 folds. Our Random Forest achieved CV R² = 0.830 ± 0.033, confirming stable performance across different data subsets — not just one lucky split.

---

### Q8. Why did you choose Random Forest over Linear Regression?

**Answer:** Linear Regression assumes a strictly linear relationship between features and the target. Real academic outcomes involve non-linear interactions — for example, high study time may only improve grades for students with low absences. Random Forest is an ensemble of many decision trees and naturally captures these non-linear interactions. On our test set, Random Forest achieved RMSE = 1.971 vs. Linear Regression's 2.378 — a 17% improvement in prediction accuracy.

---

### Q9. What is the DummyRegressor baseline and why is it important?

**Answer:** The `DummyRegressor` simply predicts the mean of all training grades for every student, ignoring all features. It achieved RMSE = 4.550 and R² = −0.010 on our test set. Comparing against this baseline is critical because it answers the question: "Does our machine learning model actually learn anything useful?" If our model performed similarly to the DummyRegressor, we would have no reason to trust it. Our Random Forest's RMSE of 1.971 — a 57% reduction — confirms the model learns genuine patterns.

---

### Q10. What are the project limitations? How could it be improved?

**Answer:** 
- **Small dataset:** 395 records limits generalization. More data would improve reliability.
- **Single context:** Data from 2 Portuguese schools, 2005–2006. Not directly applicable to Indian colleges or modern students.
- **G2 dominance:** The model's high R² relies heavily on period grades. Early-semester prediction (before any exams) is significantly less accurate.
- **No causal inference:** We can predict grades but cannot prove that changing a feature causes a grade improvement.

**Future improvements:**
- Collect larger, more diverse datasets.
- Add a behavioral-only model for early-semester use (before period exams).
- Implement SHAP (SHapley Additive exPlanations) for per-student explainability.
- Add confidence intervals to predictions to communicate uncertainty honestly.
- Incorporate longitudinal tracking: monitor actual vs. predicted grades over a semester.
