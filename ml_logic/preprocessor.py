# Imports standards
import numpy as np
import pandas as pd

# Import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from params import *
from ml_logic.registry import load_data_from_database, save_preprocessed_data


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
    Scikit-learn pipeline that transforms a cleaned dataset of shape (_, XXXXX)
    into a preprocessed one of fixed shape (_, XXXX).
    """
    print("⏳ Preprocessing in progress.. ⏳")
    X_dropped = X.drop(columns=COLUMNS_TO_DROP)
    X_dropped_season_drop = X_dropped.drop(columns=["season"])
    X_dropped_season_drop_num = X_dropped_season_drop.select_dtypes(include="number")


    # Imputing NaN values
    imputer = KNNImputer().set_output(transform='pandas')
    imputer.fit(X_dropped_season_drop_num)
    # Call the "transform" method on the object
    X_dropped_season_drop_imputed = imputer.transform(X_dropped_season_drop_num)


    # Scaling features
    robust_scaler = RobustScaler().set_output(transform='pandas')
    robust_scaler.fit(X_dropped_season_drop_imputed)
    X_num = robust_scaler.transform(X_dropped_season_drop_imputed)

    # Concatenation
    X_preprocessed = pd.concat([X_dropped[["season"]], X_dropped.select_dtypes(exclude="number"), X_num], axis=1)


    print("✅ Data preprocessed, with shape", X_preprocessed.shape)


    return X_preprocessed

# Tests
if __name__ == "__main__":
    X = load_data_from_database()
    X_processed = preprocess_features(X)
