"""
Tests and logging for the churn prediction library.

Each test function exercises one public function from churn_library, logs a
SUCCESS or ERROR message, and re-raises any exception so the caller can
detect failures.

Author: Diego Hernández Jiménez
Date: 23/07/2026
"""

import logging
import os

from dotenv import load_dotenv
import churn_library as cls

load_dotenv()

LOGS_DIR = os.getenv("LOGS_DIR", "./logs")
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")

CATEGORY_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def test_import(import_data):
    """Test that import_data returns a non-empty dataframe."""
    try:
        dataframe = import_data(cls.DATA_PATH)
        logging.info(
            "SUCCESS: import_data - shape=(%d, %d)", *dataframe.shape
        )
    except FileNotFoundError as err:
        logging.error("ERROR: import_data - file not found: %s", cls.DATA_PATH)
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("ERROR: import_data - dataframe is empty")
        raise err


def test_eda(perform_eda):
    """Test that perform_eda saves all expected image files."""
    expected_files = [
        "churn_distribution.png",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
        "total_trans_ct_distribution.png",
        "heatmap.png",
    ]
    try:
        dataframe = cls.import_data(cls.DATA_PATH)
        perform_eda(dataframe)
        for fname in expected_files:
            assert os.path.isfile(os.path.join(cls.EDA_DIR, fname))
        logging.info(
            "SUCCESS: perform_eda - %d images saved to %s",
            len(expected_files),
            cls.EDA_DIR,
        )
    except Exception as err:  # pylint: disable=broad-except
        logging.error("ERROR: perform_eda - %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    """Test that encoder_helper adds one encoded column per category."""
    try:
        dataframe = cls.import_data(cls.DATA_PATH)
        dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        dataframe = encoder_helper(dataframe, CATEGORY_COLUMNS, "Churn")
        expected_cols = [f"{col}_Churn" for col in CATEGORY_COLUMNS]
        for col in expected_cols:
            assert col in dataframe.columns
        logging.info(
            "SUCCESS: encoder_helper - %d encoded columns added",
            len(CATEGORY_COLUMNS),
        )
    except Exception as err:  # pylint: disable=broad-except
        logging.error("ERROR: encoder_helper - %s", err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test that perform_feature_engineering returns correctly sized splits."""
    try:
        dataframe = cls.import_data(cls.DATA_PATH)
        dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        dataframe = cls.encoder_helper(dataframe, CATEGORY_COLUMNS, "Churn")
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe, "Churn"
        )
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] == x_train.shape[0]
        assert y_test.shape[0] == x_test.shape[0]
        logging.info(
            "SUCCESS: perform_feature_engineering - train=%d, test=%d",
            x_train.shape[0],
            x_test.shape[0],
        )
    except Exception as err:  # pylint: disable=broad-except
        logging.error("ERROR: perform_feature_engineering - %s", err)
        raise err


def test_train_models(train_models):
    """Test that train_models saves model files and result images."""
    expected_models = [
        os.path.join(cls.MODELS_DIR, "rfc_model.pkl"),
        os.path.join(cls.MODELS_DIR, "logistic_model.pkl"),
    ]
    expected_images = [
        os.path.join(cls.RESULTS_DIR, "roc_curve_result.png"),
        os.path.join(cls.RESULTS_DIR, "rf_results.png"),
        os.path.join(cls.RESULTS_DIR, "logistic_results.png"),
        os.path.join(cls.RESULTS_DIR, "feature_importances.png"),
    ]
    try:
        dataframe = cls.import_data(cls.DATA_PATH)
        dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        dataframe = cls.encoder_helper(dataframe, CATEGORY_COLUMNS, "Churn")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            dataframe, "Churn"
        )
        train_models(x_train, x_test, y_train, y_test)
        for path in expected_models + expected_images:
            assert os.path.isfile(path)
        logging.info(
            "SUCCESS: train_models - %d models and %d images saved",
            len(expected_models),
            len(expected_images),
        )
    except Exception as err:  # pylint: disable=broad-except
        logging.error("ERROR: train_models - %s", err)
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
