# Imports standards
import numpy as np
import pandas as pd

# Import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from params import *
from ml_logic.registry import load_data_from_database, save_data


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
    Transforms a cleaned dataset of shape (_, XXXXX)
    into a preprocessed one of fixed shape (_, XXXX).
    """
    # Séparation (1997-2024) et 2025
    X_1997_2024 = X.query("season < 2025")
    X_2025 = X.query("season == 2025")

    print("⏳ Preprocessing in progress... ⏳")

    ### Preparing Data for Imputer and Scaler

    ## 1997-2024 :
    # Dropping useless columns
    X_1997_2024_dropped = X_1997_2024.drop(columns=COLUMNS_TO_DROP)
    # Dropping "season" because we don't want to scale that
    X_1997_2024_dropped_season_drop = X_1997_2024_dropped.drop(columns=["season"])
    # Selecting only numerical columns
    X_1997_2024_dropped_season_drop_num = X_1997_2024_dropped_season_drop.select_dtypes(include="number")

    ## 2025 :
    # Dropping useless columns
    X_2025_dropped = X_2025.drop(columns=COLUMNS_TO_DROP)
    # Dropping "season" because we don't want to scale that
    X_2025_dropped_season_drop = X_2025_dropped.drop(columns=["season"])
    # Selecting only numerical columns
    X_2025_dropped_season_drop_num = X_2025_dropped_season_drop.select_dtypes(include="number")

    # Imputing NaN values
    imputer = KNNImputer().set_output(transform='pandas')
    imputer.fit(X_1997_2024_dropped_season_drop_num)
    X_1997_2024_dropped_season_drop_imputed = imputer.transform(X_1997_2024_dropped_season_drop_num)
    X_2025_dropped_season_drop_imputed = imputer.transform(X_2025_dropped_season_drop_num)

    # Scaling features
    robust_scaler = RobustScaler().set_output(transform='pandas')
    robust_scaler.fit(X_1997_2024_dropped_season_drop_imputed)
    X_1997_2024_num = robust_scaler.transform(X_1997_2024_dropped_season_drop_imputed)
    X_2025_num = robust_scaler.transform(X_2025_dropped_season_drop_imputed)

    ### Preparing Data for one hot encoder

    ## 1997-2024 :
    X_1997_2024_pos = X_1997_2024[["pos"]]
    ## 2025 :
    X_2025_pos = X_2025[["pos"]]

    # One hot encode the positions
    ohe = OneHotEncoder(drop = "if_binary", # Doesn't create an extra column for binary features
                        sparse_output = False, # Returns full matrixes with zeros where need be instead of sparse matrixes
                        handle_unknown="ignore") # Useful to set everything to zero for unseen categories in the test set
    ohe.fit(X_1997_2024_pos)
    X_1997_2024_pos[ohe.get_feature_names_out()] = ohe.transform(X_1997_2024_pos)
    X_1997_2024_pos.drop(columns="pos", inplace=True)
    X_2025_pos[ohe.get_feature_names_out()] = ohe.transform(X_2025_pos)
    X_2025_pos.drop(columns="pos", inplace=True)

    # Concatenation to get back the non numerical columns + one hot encoded cat
    X_1997_2024_preprocessed = pd.concat([X_1997_2024_dropped[["season"]],
                                          X_1997_2024_dropped.select_dtypes(exclude="number"),
                                          X_1997_2024_pos,
                                          X_1997_2024_num],
                                         axis=1)
    X_2025_transformed = pd.concat([X_2025_dropped[["season"]],
                                    X_2025_dropped.select_dtypes(exclude="number"),
                                    X_2025_pos,
                                    X_2025_num],
                                   axis=1)

    print("✅ Data preprocessed (X from 1997 to 2024), with shape", X_1997_2024_preprocessed.shape)
    print("✅ Data transformed (2025), with shape", X_2025_transformed.shape)


    return X_1997_2024_preprocessed, X_2025_transformed

# Tests
if __name__ == "__main__":
    X = load_data_from_database()
    X_1997_2024_preprocessed, X_2025_transformed = preprocess_features(X)

