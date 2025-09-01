import numpy as np
import pandas as pd

from params import *

# Import data
from ml_logic.data import load_data, player_full_data_df, new_y_creator
from ml_logic.model import initialize_model, fit_model, score_model,initialize_deep_dense_model, compile_deep_model, fit_deep_model, initialize_deep_cnn_model, initialize_deep_rnn_model
from ml_logic.from_player_to_team import get_all_seasons_all_teams_starters_stats
from ml_logic.registry import load_csvs_and_save_data_to_database, save_preprocessed_data, load_data_from_database, load_preprocessed_data_from_database

# Import preprocessing function
from ml_logic.preprocessor import preprocess_features

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor



def load_and_preprocess_and_save():
    """
        LoadPreprocess X and
    """
    # Load preprocessed data for
    load_csvs_and_save_data_to_database()
    X =load_data_from_database()

    # Process data
    X_preprocessed = preprocess_features(X)

    # Save preprocessed data to database
    save_preprocessed_data(X_preprocessed)

    return X_preprocessed


def get_X_y(X_preprocessed, y)-> pd.DataFrame:
    '''
        Returns a DataFrame which contains X and y (= X_preprocessed flattened)
        Only use for ML
    '''

    all_season_team_starters_stats_flattened, season_and_team_key = get_all_seasons_all_teams_starters_stats(X_preprocessed)
    df_preprocessed_teams_with_key = pd.concat(
        [pd.DataFrame(season_and_team_key, columns=["PM"]),
         pd.DataFrame(all_season_team_starters_stats_flattened)], axis = 1)

    df_preprocessed_teams_with_key_merged_y = df_preprocessed_teams_with_key.merge(y, how="left", on="PM")

    df_preprocessed_teams_with_key_merged_y_drop_key = df_preprocessed_teams_with_key_merged_y.drop(columns="PM")

    print("\n✅ got X (X_preprocessed flattened) and y \n")

    return df_preprocessed_teams_with_key_merged_y_drop_key



def train_ML(model_type, df_preprocessed_teams_with_key_merged_y_drop_key, split_ratio):
    """
        Trains model
        returns the model trained and the X_test_preproc and y_test as DFs
    """
    # Create (X_train_processed, y_train, X_val_processed, y_val, X_test_preprocessed, y_test)
    test_length = int(len(df_preprocessed_teams_with_key_merged_y_drop_key) * split_ratio)
    val_length = int((len(df_preprocessed_teams_with_key_merged_y_drop_key)-test_length) * split_ratio)
    train_length = len(df_preprocessed_teams_with_key_merged_y_drop_key) - val_length - test_length

    df_train_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[:train_length, :].sample(frac=1) # Shuffle datasets to improve training
    df_val_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[train_length: train_length + val_length, :].sample(frac=1)
    df_test_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[train_length+val_length:, :].sample(frac=1)

    # Create X's
    X_train_preprocessed = df_train_preprocessed.iloc[:, :-1]
    X_val_preprocessed = df_val_preprocessed.iloc[:, :-1]
    X_test_preprocessed = df_test_preprocessed.iloc[:, :-1]

    # Create y's
    y_train = pd.DataFrame(df_train_preprocessed.iloc[:, -1])
    y_val = pd.DataFrame(df_val_preprocessed.iloc[:, -1])
    y_test = pd.DataFrame(df_test_preprocessed.iloc[:, -1])

    # Initialize ML model
    model = initialize_model(model_type)

    # Train model
    model = fit_model(model, X_train_preprocessed, y_train)
    print("\n✅ train_ML() done \n")

    return model, X_test_preprocessed, y_test


def train_DL(model_type , X_teams_preprocessed, y, split_ratio=0.1):
    """
        Trains model, model type should be ['dense','rnn','cnn']
        returns the model trained and the X_test_preproc and y_test as DFs
    """
    # Create (X_train_processed, y_train, X_val_processed, y_val, X_test_preprocessed, y_test)
    test_length = int(len(X_teams_preprocessed) * split_ratio)
    val_length = int((len(X_teams_preprocessed) - test_length) * split_ratio)
    train_length = len(X_teams_preprocessed) - val_length - test_length


    # Create X's
    X_train_preprocessed = X_teams_preprocessed[:train_length]
    print("💛❤️", np.shape(X_train_preprocessed))
    X_val_preprocessed = X_teams_preprocessed[train_length: train_length + val_length]
    X_test_preprocessed = X_teams_preprocessed[train_length+val_length:]

    # Create y's
    y_train = y[:train_length]
    y_val = y[train_length: train_length + val_length]
    y_test = y[train_length+val_length:]

    # Initialize deep model :
    if model_type == "dense":
        model = initialize_deep_dense_model(X_train_preprocessed)
    elif model_type == "cnn":
        model = initialize_deep_cnn_model(X_train_preprocessed)
    else :
        model = initialize_deep_rnn_model(X_train_preprocessed)

    # Compile DL model
    model = compile_deep_model(model)

    # Train DL model
    history,model = fit_deep_model(model, X_train_preprocessed, y_train, validation_data=(X_val_preprocessed,y_val))
    print("\n✅ train_DL() done \n")

    return model, X_test_preprocessed, y_test


def evaluate_ML_model(model, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate the performance of the latest production model on processed data
    Return metrics as a DataFrame
    """

    df_score = model.score(X_test, y_test)

    print("\n✅ evaluate() done")
    print("\n💯 Score: ", df_score, "\n")

    return df_score

def evaluate_DL_model(model, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate the performance of the latest production model on processed data
    Return metrics as a DataFrame
    """

    df_score = model.evaluate(X_test, y_test)

    print("✅ evaluate() done \n")
    print("\n💯 Score: ", df_score, "\n")

    return df_score

def pred(model, X_new_preprocessed: pd.DataFrame=None):
    """
    Make a prediction using the latest trained model
    """
    # Predict
    y_pred = model.predict(X_new_preprocessed)

    # Print result
    print("✅ pred() done \n")
    print("🔮 Prediction: ", y_pred, "of shape : ", y_pred.shape, "\n")

    return y_pred



if __name__ == '__main__':

# ML tests
    # X_preprocessed = load_and_preprocess_and_save()
    # y_winrate, y = new_y_creator(1997)
    # df_for_model = get_X_y(X_preprocessed, y_winrate)
    # model, X_test_preprocessed, y_test = train_ML(LinearRegression(), df_for_model, 0.3)
    # score = evaluate_ML_model(model, X_test_preprocessed, y_test)
    # X_new = df_for_model.iloc[[25], :-1] # Test de pred d'une ligne au pif
    # y_pred = pred(model, X_new)

# DL tests
    # X_preprocessed = load_and_preprocess_and_save()
    X_preprocessed = load_preprocessed_data_from_database()
    y_winrate, y_df = new_y_creator(1997)
    y = y_df["global_score"]
    X, _, __ = get_all_seasons_all_teams_starters_stats(X_preprocessed, False)
    model, X_test_preprocessed, y_test = train_DL("dense", np.array(X), np.array(y), 0.1)
    score = evaluate_DL_model(model, X_test_preprocessed, y_test)

    X_new = X_test_preprocessed[15, :] # Test de pred d'une ligne au pif
    print("🎯", np.shape(X_new))

    y_pred = pred(model, X_new)
