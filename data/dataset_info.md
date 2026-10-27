# UCI Student Performance Dataset Documentation

## 1. Dataset Provenance and Source

- **Dataset Name**: Student Performance Data Set
- **Source**: UCI Machine Learning Repository
- **URL**: [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Direct Archive**: `https://archive.ics.uci.edu/static/public/320/student+performance.zip`
- **Original Authors**: P. Cortez and A. Silva
- **Citation**:
  > Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. In *Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008)*, pp. 5-12, Porto, Portugal. ISBN: 978-9077381-39-7.

This dataset was downloaded directly from the official UCI repository and verified. No synthetic records have been generated.

---

## 2. Dataset Overview

- **Subject / Context**: Secondary education student achievement in Mathematics at two Portuguese public schools (*Gabriel Pereira* and *Mousinho da Silveira*).
- **Number of Observations (Rows)**: 395
- **Total Attributes (Columns)**: 33 (32 predictor features + 1 target variable)
- **Missing Values**: 0 (Cleaned and verified)
- **Duplicate Records**: 0

---

## 3. Target Variable Specification

- **Target Column**: `G3`
- **Definition**: Final Mathematics evaluation grade.
- **Data Type**: Integer (`int64`).
- **Scale**: Continuous numerical mark ranging from **0 to 20** (Portuguese secondary education grading standard):
  - `0 - 9`: Fail / Insufficient
  - `10 - 13`: Pass / Sufficient
  - `14 - 15`: Good
  - `16 - 20`: Excellent / Very Good
- **Machine Learning Task**: **Regression** (predicting the continuous score `G3` from $0$ to $20$).
- **Key Baseline**: Predicting the sample mean or median using `DummyRegressor`.

---

## 4. Complete Data Dictionary

### Academic History Features
| Column | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `G1` | Numeric (`int64`) | 0 to 20 | First period evaluation grade |
| `G2` | Numeric (`int64`) | 0 to 20 | Second period evaluation grade |
| `failures` | Numeric (`int64`) | 0 to 4 | Number of past academic class failures ($n$ if $1 \le n < 3$, else 4) |
| `absences` | Numeric (`int64`) | 0 to 93 | Number of school absences |

### Demographic & Family Features
| Column | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `school` | Categorical (`str`) | `GP` (Gabriel Pereira), `MS` (Mousinho da Silveira) | Student's school |
| `sex` | Categorical (`str`) | `F` (female), `M` (male) | Student's gender |
| `age` | Numeric (`int64`) | 15 to 22 | Student's age |
| `address` | Categorical (`str`) | `U` (urban), `R` (rural) | Home address location |
| `famsize` | Categorical (`str`) | `LE3` ($\le 3$), `GT3` ($> 3$) | Family size |
| `Pstatus` | Categorical (`str`) | `T` (together), `A` (apart) | Parent's cohabitation status |
| `Medu` | Numeric (`int64`) | 0 (none) to 4 (higher education) | Mother's education level |
| `Fedu` | Numeric (`int64`) | 0 (none) to 4 (higher education) | Father's education level |
| `Mjob` | Categorical (`str`) | `teacher`, `health`, `services`, `at_home`, `other` | Mother's occupation |
| `Fjob` | Categorical (`str`) | `teacher`, `health`, `services`, `at_home`, `other` | Father's occupation |
| `guardian` | Categorical (`str`) | `mother`, `father`, `other` | Primary guardian |

### Academic Support & Extra-Curricular Features
| Column | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `schoolsup` | Categorical (`str`) | `yes`, `no` | Extra school educational support |
| `famsup` | Categorical (`str`) | `yes`, `no` | Family educational support |
| `paid` | Categorical (`str`) | `yes`, `no` | Extra paid classes within Math |
| `activities` | Categorical (`str`) | `yes`, `no` | Extra-curricular activities |
| `nursery` | Categorical (`str`) | `yes`, `no` | Attended nursery school |
| `higher` | Categorical (`str`) | `yes`, `no` | Aspires to pursue higher education |
| `internet` | Categorical (`str`) | `yes`, `no` | Internet access at home |

### Lifestyle & Habits Features
| Column | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `studytime` | Numeric (`int64`) | 1 ($<2$h), 2 ($2-5$h), 3 ($5-10$h), 4 ($>10$h) | Weekly study time |
| `traveltime` | Numeric (`int64`) | 1 ($<15$m), 2 ($15-30$m), 3 ($30-60$m), 4 ($>60$m) | Commute time to school |
| `reason` | Categorical (`str`) | `home`, `reputation`, `course`, `other` | Reason for choosing school |
| `romantic` | Categorical (`str`) | `yes`, `no` | In a romantic relationship |
| `famrel` | Numeric (`int64`) | 1 (very bad) to 5 (excellent) | Quality of family relationship |
| `freetime` | Numeric (`int64`) | 1 (very low) to 5 (very high) | Free time after school |
| `goout` | Numeric (`int64`) | 1 (very low) to 5 (very high) | Frequency of going out with friends |
| `Dalc` | Numeric (`int64`) | 1 (very low) to 5 (very high) | Workday alcohol consumption |
| `Walc` | Numeric (`int64`) | 1 (very low) to 5 (very high) | Weekend alcohol consumption |
| `health` | Numeric (`int64`) | 1 (very bad) to 5 (very good) | Current health status |

---

## 5. Dataset Limitations & Ethical Considerations

1. **Context Specificity**: The dataset originates from two Portuguese schools between 2005 and 2006. Cultural and demographic patterns may not directly generalize to modern university or international collegiate systems.
2. **Self-Reported Survey Attributes**: Non-academic features (alcohol consumption, family relationships, study hours) were recorded via questionnaires and may contain social desirability bias.
3. **Period Grades Collinearity**: `G1` and `G2` exhibit very strong correlation with `G3` ($r > 0.8$). While excellent for final grade forecasting, a robust model must also demonstrate predictive capability using behavioral indicators when early exam grades are not yet available.
4. **Advisory Purpose**: Predictions generated by this system are statistical approximations intended for early academic intervention and advising, never for automated high-stakes disciplinary decisions.
