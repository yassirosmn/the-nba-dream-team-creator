# Imports standards
import numpy as np
import pandas as pd

# Import preprocessing
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from params import *


def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    def create_preprocessor() -> ColumnTransformer:
        """
            Scikit-learn pipeline that transforms a cleaned dataset of shape (_, XXXXX)
            into a preprocessed one of fixed shape (_, XXXX).
        """
        X_dropped = X.drop(columns=COLUMNS_TO_DROP)

        X_num = X_dropped.select_dtypes(include="number")

        robust_features = X_num.columns #Insérer noms de features...

        scalers = ColumnTransformer([
            ("rob", RobustScaler(), robust_features), # Robust
        ])

        numerical_pipeline = Pipeline([
            ("imputer", KNNImputer()),
            ("scalers", scalers)
        ])


        preprocessor = ColumnTransformer([
            ("num_pipeline", numerical_pipeline, make_column_selector(dtype_include="number")) # num_features
            # ("cat_pipeline", categorical_pipeline, make_column_selector(dtype_exclude="number")) # cat_features
        ]).set_output(transform="pandas")

        return preprocessor

    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed


# Tests
if __name__ == "__main__":
    from data import load_data, player_full_data_df
    dfs = load_data()
    X = player_full_data_df(dfs, 1997)
    X_preprocessed = preprocess_features(X)
