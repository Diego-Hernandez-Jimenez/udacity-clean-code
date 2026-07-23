"""
Entry point for the churn prediction pipeline.

Author: Diego Hernández Jiménez
Date: 23/07/2026
"""

from src.churn_library import (
    DATA_PATH,
    create_output_directories,
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_models,
)

CATEGORY_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]


def main():
    """Run the full churn prediction pipeline."""
    create_output_directories()

    dataframe = import_data(DATA_PATH)

    perform_eda(dataframe)

    dataframe = encoder_helper(dataframe, CATEGORY_COLUMNS, "Churn")
    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe, "Churn")
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
