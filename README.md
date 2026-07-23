# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)

---

## Project Description

This project identifies credit card customers likely to churn using a supervised machine learning pipeline. The raw customer data is processed through exploratory data analysis, feature engineering, and model training, producing two trained classifiers — a Random Forest and a Logistic Regression — along with evaluation artefacts (ROC curves, classification reports, and feature importance plots).

The pipeline is implemented following clean-code principles: modular functions, configuration via `.env` and YAML files, structured logging, and a pylint score of 10/10.

---

## Files and Data Description

### Main Files

- `src/churn_library.py`  
  Core library with all pipeline functions: data import, EDA, categorical encoding, feature engineering, model training, and reporting. Designed to be imported by other scripts.

- `main.py`  
  Entry point that orchestrates the full pipeline by calling functions from `churn_library.py`.

- `src/churn_script_logging_and_tests.py`  
  Test suite that exercises each function in `churn_library.py`, logs SUCCESS or ERROR messages to `logs/churn_library.log`, and re-raises exceptions on failure.

- `notebooks/churn_notebook.ipynb`  
  Exploratory notebook used as the original prototype. The production code in `churn_library.py` was refactored from this notebook.

- `src/param_config.yaml`  
  Hyperparameter configuration for the Random Forest grid search and the train/test split ratio. Decouples model parameters from code.

- `.env`  
  Environment-level constants: input data path and output directory paths. Loaded at runtime via `python-dotenv`.

---

### Data

- `data/bank_data.csv`  
  Credit card customer dataset with 10,127 rows and 22 columns. The target variable is derived from `Attrition_Flag` (binary: 0 = existing customer, 1 = churned). Features include demographics, account activity, and transaction history. No missing values.

---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Trained models → `models/`
- Logs → `logs/churn_library.log`

---

## Running the Files

### Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

### 1. Run the Pipeline

```bash
python main.py
```

Loads data from `data/bank_data.csv`, performs EDA, encodes categorical features, trains a Random Forest (with randomised hyperparameter search) and a Logistic Regression, and saves models, ROC curves, classification reports, and feature importance plots to their respective output directories.

### 2. Run Tests and Logging

```bash
python src/churn_script_logging_and_tests.py
```

Runs five tests — one per pipeline function — and writes structured log messages to `logs/churn_library.log`. Each test logs `SUCCESS` with relevant metrics on pass, or `ERROR` with the exception message on failure, then re-raises the exception.

---

## Expected Outputs

### Models

| File | Description |
|---|---|
| `models/rfc_model.pkl` | Best Random Forest estimator from randomised search |
| `models/logistic_model.pkl` | Logistic Regression pipeline (StandardScaler + LogisticRegression) |

### EDA Images (`images/eda/`)

| File | Description |
|---|---|
| `churn_distribution.png` | Histogram of the binary churn label |
| `customer_age_distribution.png` | Histogram of customer age |
| `marital_status_distribution.png` | Normalised bar chart of marital status |
| `total_trans_ct_distribution.png` | KDE density plot of total transaction count |
| `heatmap.png` | Correlation heatmap of all numeric features |

### Result Images (`images/results/`)

| File | Description |
|---|---|
| `roc_curve_result.png` | Overlaid ROC curves for both models |
| `rf_results.png` | Random Forest classification report (train and test) |
| `logistic_results.png` | Logistic Regression classification report (train and test) |
| `feature_importances.png` | Bar chart of Random Forest feature importances |

### Logs

- `logs/churn_library.log` — one log entry per test, overwritten on each run.

---

## Notes

- **Logging**: configured at module level in `churn_script_logging_and_tests.py` with `filemode="w"` (overwrite on each run). The `logs/` directory is created automatically if absent.
- **Configuration**: model hyperparameters and split ratios live in `param_config.yaml`; path constants live in `.env`. Neither file should be modified without also updating the other scripts that depend on them.
- **Dataset assumption**: the pipeline expects the `Attrition_Flag` column to exist and contain the values `"Existing Customer"` (non-churn) or `"Attrited Customer"` (churn).
- **Training time**: Random Forest training with `RandomizedSearchCV` (12 iterations, 3-fold CV) takes approximately 15–20 minutes on a standard CPU.
