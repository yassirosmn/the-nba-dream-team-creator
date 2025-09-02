import numpy as np
import pandas as pd

from params import *

# Import data
from ml_logic.data import load_data, player_full_data_df, new_y_creator
from ml_logic.model import initialize_model, fit_model, score_model,initialize_deep_dense_model, compile_deep_model, fit_deep_model, initialize_deep_cnn_model, initialize_deep_rnn_model
from ml_logic.from_player_to_team import get_all_seasons_all_teams_starters_stats
from ml_logic.registry import load_csvs_and_save_data_to_database, save_data, load_data_from_database, load_preprocessed_data_from_database

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
    X = load_data_from_database()
    # Process data
    X_preprocessed = preprocess_features(X)
    # Save preprocessed data to database
    save_data(X_preprocessed, "data_preprocessed")

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

    df_train_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[test_length + val_length:].sample(frac=1) # Shuffle datasets to improve training
    df_val_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[test_length:test_length + val_length].sample(frac=1)
    df_test_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[:test_length].sample(frac=1)

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


def train_DL(model_type , X_teams_preprocessed, y, split_ratio=0.1, verbose = "auto"):
    """
        Trains model, model type should be ['dense','rnn','cnn']
        returns the model trained and the X_test_preproc and y_test as numpy.arrays
    """
    # Create (X_train_processed, y_train, X_val_processed, y_val, X_test_preprocessed, y_test)
    test_length = int(len(X_teams_preprocessed) * split_ratio)
    val_length = int((len(X_teams_preprocessed) - test_length) * 3*split_ratio)
    train_length = len(X_teams_preprocessed) - val_length - test_length


    # Create X's
    X_train_preprocessed = X_teams_preprocessed[test_length + val_length:]
    X_val_preprocessed = X_teams_preprocessed[test_length:test_length + val_length]
    X_test_preprocessed = X_teams_preprocessed[:test_length]


    # Create y's
    y_train = y[test_length + val_length:]
    y_val = y[test_length:test_length + val_length]
    y_test = y[:test_length]


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
    print("▶️ train_DL() begin ")
    history,model = fit_deep_model(model, X_train_preprocessed, y_train, validation_data=(X_val_preprocessed,y_val), verbose=verbose)
    print("✅ train_DL() done ")

    return model, history, X_test_preprocessed, y_test


def evaluate_ML_model(model, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate the performance of the latest production model on processed data
    Return metrics as a DataFrame
    """
    print("▶️ evaluate_ML() begin")
    df_score = model.score(X_test, y_test)
    print("✅ evaluate_ML() done")
    print("💯 Score: ", df_score)
    return df_score

def evaluate_DL_model(model, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate the performance of the latest production model on processed data
    Return metrics as a DataFrame
    """
    print("▶️ evaluate_DL() begin")
    df_score = model.evaluate(X_test, y_test)
    print("✅ evaluate_DL() done")
    print("💯 Score: ", df_score)
    return df_score

def pred(model, X_new_preprocessed: pd.DataFrame=None):
    """
    Make a prediction using the latest trained model
    """
    # Predict
    print("▶️ pred() begin ")
    y_pred = model.predict(X_new_preprocessed)
    # Print result
    print("✅ pred() done ")
    print("🔮 Prediction: ", y_pred, "of shape : ", y_pred.shape)
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
    print("\n")
    # X_preprocessed = load_and_preprocess_and_save()
    X_preprocessed = load_preprocessed_data_from_database()
    print("\n")

    y_winrate, y_df = new_y_creator(1997)
    X, keys, __ = get_all_seasons_all_teams_starters_stats(X_preprocessed, False)
    y_df_classé = pd.DataFrame(keys, columns=["PM"]).merge(y_winrate,
                                                           how="left",
                                                           on="PM")
    # y = y_df_classé["global_score"]
    y = y_df_classé["winrate"]
    print("✅ X and y created")
    print("\n")

    model, history, X_test_preprocessed, y_test = train_DL("dense",
                                                           np.array(X),
                                                           np.array(y),
                                                           split_ratio=0.04,
                                                           verbose=0)
    print("🔎 loss : ", history.history["loss"][-1])
    print("🔎 mae : ", history.history["mae"][-1])
    print("🔎 val_loss : ", history.history["val_loss"][-1])
    print("🔎 val_mae : ", history.history["val_mae"][-1])

    print("\n")
    score = evaluate_DL_model(model, X_test_preprocessed, y_test)

    # # Prédiction de toute les lignes du X_test :
    print("\n")
    y_preds = []
    y_trues = []
    for row in range(len(X_test_preprocessed)):
        X_new = X_test_preprocessed[row:row+1, :, :] # Test de pred d'une ligne au pif
        y_preds.append(pred(model, X_new)[0][0])
        y_trues.append(y_test[row:row+1][0])
    df_trues_preds = pd.DataFrame()
    df_trues_preds["y_trues"] = y_trues
    df_trues_preds["y_preds"] = y_preds
    df_trues_preds["diff"] = df_trues_preds["y_trues"]-df_trues_preds["y_preds"]

    rmse = (np.mean((df_trues_preds["y_trues"]-df_trues_preds["y_preds"])**2))**0.5
    print("💯 RMSE = ", rmse)

    from sklearn.metrics import r2_score
    r2 = r2_score(df_trues_preds["y_trues"], df_trues_preds["y_preds"])
    print("💯 r2 = ", r2)

    # # Prédiction d'une seule ligne :
    # X_new = X_test_preprocessed[27:28, :, :] # Test de pred d'une ligne au pif
    # y_pred = pred(model, X_new)
    # print("🏀 y_true : ", y_test[27:28])
