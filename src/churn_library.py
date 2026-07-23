"""
Churn prediction library: EDA, feature engineering, model training, and reporting.

Author: Diego Hernández Jiménez
Date: 23/07/2026
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ["QT_QPA_PLATFORM"] = "offscreen"

load_dotenv()

EDA_DIR = os.getenv("EDA_DIR", "./images/eda")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./images/results")
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
DATA_PATH = os.getenv("DATA_PATH", "./data/bank_data.csv")

KEEP_COLS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    return pd.read_csv(pth)


def perform_eda(dataframe):
    """
    Perform EDA on dataframe and save figures to EDA_DIR.

    input:
            dataframe: pandas dataframe
    output:
            None
    """
    create_output_directories()

    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    plt.figure(figsize=(20, 10))
    dataframe["Churn"].hist()
    plt.savefig(os.path.join(EDA_DIR, "churn_distribution.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe["Customer_Age"].hist()
    plt.savefig(
        os.path.join(EDA_DIR, "customer_age_distribution.png"), bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.savefig(
        os.path.join(EDA_DIR, "marital_status_distribution.png"), bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(
        os.path.join(EDA_DIR, "total_trans_ct_distribution.png"), bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(20, 10))
    corr = dataframe.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(os.path.join(EDA_DIR, "heatmap.png"), bbox_inches="tight")
    plt.close()


def encoder_helper(dataframe, category_lst, response):
    """
    Encode categorical features as their mean churn rate per category.

    input:
            dataframe: pandas dataframe
            category_lst: list of categorical column names
            response: response column name
    output:
            dataframe: updated dataframe with encoded columns
    """
    for col in category_lst:
        dataframe[f"{col}_{response}"] = dataframe[col].map(
            dataframe.groupby(col)[response].mean()
        )
    return dataframe


def perform_feature_engineering(dataframe, response, config_path="src/param_config.yaml"):
    """
    Build feature matrix and split dataset into train and test sets.

    input:
              dataframe: pandas dataframe
              response: response column name
              config_path: path to the YAML config file
    output:
              x_train, x_test, y_train, y_test
    """
    with open(config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    split_params = config["train_test_split"]

    y_labels = dataframe[response]
    x_features = pd.DataFrame()
    x_features[KEEP_COLS] = dataframe[KEEP_COLS]

    return train_test_split(x_features, y_labels, **split_params)


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Save classification reports for RF and LR models as images.

    input:
            y_train: training labels
            y_test: test labels
            y_train_preds_lr: LR training predictions
            y_train_preds_rf: RF training predictions
            y_test_preds_lr: LR test predictions
            y_test_preds_rf: RF test predictions
    output:
            None
    """
    create_output_directories()

    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01, 1.25, "Random Forest Train", {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01, 0.6, "Random Forest Test", {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, "rf_results.png"), bbox_inches="tight")
    plt.close()

    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        "Logistic Regression Train",
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        "Logistic Regression Test",
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(
        os.path.join(RESULTS_DIR, "logistic_results.png"), bbox_inches="tight"
    )
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    Save a bar chart of feature importances for a fitted tree-based model.

    input:
            model: trained model with feature_importances_ attribute
            x_data: dataframe of feature columns
            output_pth: file path to save the figure
    output:
            None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=45, ha="right")
    plt.savefig(output_pth, bbox_inches="tight")
    plt.close()


def train_models(x_train, x_test, y_train, y_test, config_path="src/param_config.yaml"):
    """
    Train RF and LR models, then save models, ROC curve, classification reports,
    and feature importances.

    input:
            x_train: training features
            x_test: test features
            y_train: training labels
            y_test: test labels
            config_path: path to the YAML config file
    output:
            None
    """
    create_output_directories()

    with open(config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    rf_config = config["random_forest"]

    rfc = RandomForestClassifier(random_state=42)
    lrc = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])

    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=rf_config["param_dist"],
        **rf_config["search"],
        error_score="raise",
    )

    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    joblib.dump(cv_rfc.best_estimator_, os.path.join(MODELS_DIR, "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(MODELS_DIR, "logistic_model.pkl"))

    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, x_test, y_test, ax=axis)
    RocCurveDisplay.from_estimator(lrc, x_test, y_test, ax=axis)
    plt.savefig(
        os.path.join(RESULTS_DIR, "roc_curve_result.png"), bbox_inches="tight"
    )
    plt.close()

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_train,
        os.path.join(RESULTS_DIR, "feature_importances.png"),
    )
