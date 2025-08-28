import numpy as np

from typing import Tuple

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping


def initialize_model(model_type, input_shape : tuple = None) -> Model:
    """
    Initialize the model
    """

    model = model_type

    return model


def fit_model(model, X, y) :
    model.fit(X,y)
    return model



def score_model(model, X, y) :
    return model.score(X,y)

if __name__ == "__main__":

    from data import load_data, player_full_data_df, y_creator
    from preprocessor import preprocess_features
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from xgboost import XGBRegressor
    import pandas as pd
    from sklearn.model_selection import cross_validate


    dfs = load_data()
    X = player_full_data_df(dfs, 1997)

    y = y_creator(1997)

    X_preprocessed = preprocess_features(X)

    model = initialize_model(model_type= XGBRegressor(), input_shape = None)

    model.fit(X_preprocessed,y)

    # model = compile_model(model=model, learning_rate=0.0005)

    # model = train_model(model=model,
    #                              X=X_preprocessed,
    #                              y=y,
    #                              batch_size=256,
    #                              patience=2,
    #                              validation_data=None,
    #                              validation_split=0.3)

    # metrics = evaluate_model(model = model,
    #                          X = X_preprocessed,
    #                          y = y,
    #                          batch_size=64)

    print(model.score(X_preprocessed,y))

    # cv_results = cross_validate(model, X_preprocessed, y, cv=5)

    # accuracy = cv_results["test_score"].mean()

    print("Test good (✅ pour Flavian)")
