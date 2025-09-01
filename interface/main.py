import numpy as np
import pandas as pd

from params import *

# Import data
from ml_logic.data import load_data, player_full_data_df, new_y_creator
from ml_logic.model import initialize_model, fit_model, score_model,initialize_deep_dense_model, compile_deep_model, fit_deep_model, initialize_deep_cnn_model, initialize_deep_rnn_model
from ml_logic.from_player_to_team import get_all_seasons_all_teams_starters_stats
from ml_logic.registry import load_csvs_and_save_data_to_database, save_preprocessed_data, load_data_from_database

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
        Returns a DataFrame which contains X and y
    '''

    all_season_team_starters_stats_flattened, season_and_team_key = get_all_seasons_all_teams_starters_stats(X_preprocessed)
    df_preprocessed_teams_with_key = pd.concat(
        [pd.DataFrame(season_and_team_key, columns=["PM"]),
         pd.DataFrame(all_season_team_starters_stats_flattened)], axis = 1)

    df_preprocessed_teams_with_key_merged_y = df_preprocessed_teams_with_key.merge(y, how="left", on="PM")

    df_preprocessed_teams_with_key_merged_y_drop_key = df_preprocessed_teams_with_key_merged_y.drop(columns="PM")

    return df_preprocessed_teams_with_key_merged_y_drop_key


def train(model_type, df_preprocessed_teams_with_key_merged_y_drop_key, split_ratio):
    """
        Trains model
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

    model = initialize_model(model_type)

    # Train model
    model = fit_model(model, X_train_preprocessed, y_train)
    print("✅ train() done \n")

    return model


def train_deep(model_type , X_preprocessed, y, split_ratio):
    """
        Trains model, model type should be ['dense','rnn','cnn']
    """
    all_season_team_starters_stats_flattened, season_and_team_key = get_all_seasons_all_teams_starters_stats(X_preprocessed)
    df_preprocessed_teams_with_key = pd.concat(
        [pd.DataFrame(season_and_team_key, columns=["PM"]),
         pd.DataFrame(all_season_team_starters_stats_flattened)], axis = 1)

    df_preprocessed_teams_with_key_merged_y = df_preprocessed_teams_with_key.merge(y, how="left", on="PM")

    df_preprocessed_teams_with_key_merged_y_drop_key = df_preprocessed_teams_with_key_merged_y.drop(columns="PM")

    # Create (X_train_processed, y_train, X_val_processed, y_val, X_test_preprocessed, y_test)
    test_length = int(len(df_preprocessed_teams_with_key_merged_y_drop_key) * split_ratio)
    val_length = int((len(df_preprocessed_teams_with_key_merged_y_drop_key)-test_length) * split_ratio)
    train_length = len(df_preprocessed_teams_with_key_merged_y_drop_key) - val_length - test_length

    df_train_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[:train_length, :].sample(frac=1) # Shuffle datasets to improve training
    df_val_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[train_length: train_length + val_length, :].sample(frac=1)
    df_test_preprocessed = df_preprocessed_teams_with_key_merged_y_drop_key.iloc[train_length+val_length:, :].sample(frac=1)

    X_train_preprocessed = df_train_preprocessed.iloc[:, :-1]
    X_val_preprocessed = df_val_preprocessed.iloc[:, :-1]
    X_test_preprocessed = df_test_preprocessed.iloc[:, :-1]

    y_train = pd.DataFrame(df_train_preprocessed.iloc[:, -1])
    y_val = pd.DataFrame(df_val_preprocessed.iloc[:, -1])
    y_test = pd.DataFrame(df_test_preprocessed.iloc[:, -1])

    print(X_train_preprocessed.shape)
    if model_type == "dense":
        model = initialize_deep_dense_model(X_train_preprocessed)
    elif model_type == "cnn":
        model = initialize_deep_cnn_model(X_train_preprocessed)
    else :
        model = initialize_deep_rnn_model(X_train_preprocessed)

    model = compile_deep_model(model)

    history,model = fit_deep_model(model, X_train_preprocessed, y_train, validation_data=(X_val_preprocessed,y_val))
    print("✅ train() done \n")
    eval = model.evaluate(X_test_preprocessed, y_test)

    return model, eval


def evaluate(model, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate the performance of the latest production model on processed data
    Return metrics as a DataFrame
    """
    # print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    # model = load_model(stage=stage)
    # assert model is not None

    # min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    # max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    # query = f"""
    #     SELECT * EXCEPT(_0)
    #     FROM `{GCP_PROJECT}`.{BQ_DATASET}.processed_{DATA_SIZE}
    #     WHERE _0 BETWEEN '{min_date}' AND '{max_date}'
    # """

    # data_processed_cache_path = Path(f"{LOCAL_DATA_PATH}/processed/processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    # data_processed = get_data_with_cache(
    #     gcp_project=GCP_PROJECT,
    #     query=query,
    #     cache_path=data_processed_cache_path,
    #     data_has_header=False
    # )

    # if data_processed.shape[0] == 0:
    #     print("❌ No data to evaluate on")
    #     return None

    # data_processed = data_processed.to_numpy()

    # X_new = data_processed[:, :-1]
    # y_new = data_processed[:, -1]

    # metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    # mae = metrics_dict["mae"]

    # params = dict(
    #     context="evaluate", # Package behavior
    #     training_set_size=DATA_SIZE,
    #     row_count=len(X_new)
    # )

    # save_results(params=params, metrics=metrics_dict)


    df_score = model.score(X_test,y_test)

    print("✅ evaluate() done \n")

    return df_score


def pred(model, X_new_preprocessed: pd.DataFrame=None):
    """
    Make a prediction using the latest trained model
    """

    # Display
    print("⭐️ Use case: predict\n")

    # Predict
    y_pred = model.predict(X_new_preprocessed)

    # Print result
    print("\n✅ Prediction: ", y_pred, "\n", "shape is: ", y_pred.shape, "\n")

    return y_pred



if __name__ == '__main__':

    # X_preprocessed = load_and_preprocess_and_save()
    # y_winrate, y = new_y_creator(1997)
    # model, score = train(LinearRegression(), X_preprocessed, y_winrate, 0.3)
    # # XGBRegressor(n_estimators=3, max_depth=5)
    # print(score)

    # # Test d'une ligne au pif
    # X_new = X_preprocessed.iloc[[5]]
    # pred(model, X_new)

# deep test
    X_preprocessed = load_and_preprocess_and_save()
    y_winrate,y = new_y_creator(1997)
    model, eval = train_deep("dense",X_preprocessed, y, 0.3)
    print(eval)
