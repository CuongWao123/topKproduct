# Banking Product Recommendation & EDA System

## 📁 Project Structure

```
rec-sys/
│
├── data/
│   └── processed/
│       └── train.csv
│       └── test_*.csv
│
├── eda.ipynb
├── modeling.ipynb
├── labels.ipynb
├── ...
```

---

## 1️⃣ Data Preparation

- **File:** `labels.ipynb`
- **Tasks:**
  - Load raw monthly data.
  - Create product acquisition labels for each customer.
  - Merge features and labels.
  - Save processed train/test datasets to `data/processed/`.

---

## 2️⃣ Exploratory Data Analysis & Feature Engineering

- **File:** `eda.ipynb`
- **Tasks:**
  - **Memory Optimization:** Downcast numeric types for efficiency.
  - **Cleaning:** Fill missing values, create tenure features, convert dates, drop redundant columns, handle outliers.
  - **Univariate Analysis:** Distribution plots for features (age, gender, income, etc.), product ownership rates.
  - **Multivariate Analysis:** Correlation heatmaps, time-dependent product acquisition trends.
  - **Label Grouping:** Payment accounts, Deposits, Investments, Loans, Pensions, Others.
  - **Drift Detection:** PSI and KS statistics for feature stability over time.

---

## 3️⃣ Modeling & Recommendation

- **File:** `modeling.ipynb`
- **Tasks:**
  - **Multi-label Classification Models:** Random Forest, XGBoost, LightGBM (all via `MultiOutputClassifier`).
  - **Pipeline:** Preprocessing, scaling, encoding, feature selection, model training.
  - **Hyperparameter Tuning:** `RandomizedSearchCV` for all models.
  - **Evaluation:** Metrics (Accuracy, Hamming Loss, Jaccard, MAP@k, Precision@k), comprehensive evaluation across multiple test sets.
  - **Recommendation Algorithm:** Predict probability for each product, recommend top-k products.

---

## 4️⃣ Key Features

- **Target Columns:** All product labels (see `target_cols` in `eda.ipynb`).
- **Feature Engineering:** Tenure, age groups, income, segment, channel, etc.
- **Data Quality:** Outlier detection, missing value analysis, drift detection.

---

## 5️⃣ Usage Guide

### A. Data Preparation
1. Run `labels.ipynb` to create train/test datasets.

### B. EDA & Feature Engineering
2. Run `eda.ipynb` for data cleaning, analysis, and feature engineering.

### C. Modeling & Recommendation
3. Run `modeling.ipynb`:
   - Train models with hyperparameter tuning.
   - Evaluate and compare models.
   - Save best model for deployment.

### D. Recommendation
4. Use saved model to recommend products for new customers:
   - Load model (`joblib.load`).
   - Prepare customer features.
   - Predict top-k products.

---

## 6️⃣ How to Extend

- Add more models to `pipelines` and `param_grids`.
- Implement collaborative filtering if you have user-item interaction data.
- Add more advanced feature engineering or drift detection.

---

## 7️⃣ References

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/)
- [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/)
- [Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/)

---

## 8️⃣ Contact

For questions or improvements, please open an issue or contact the project owner.

---

**This project provides a full pipeline for banking product recommendation, from raw data to model deployment, with robust