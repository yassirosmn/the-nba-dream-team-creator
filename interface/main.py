import numpy as np
import pandas as pd

from params import *

# Import data
from ml_logic.data import load_data, player_full_data_df, y_creator
from ml_logic.model import initialize_model, fit_model, score_model
from ml_logic.from_player_to_team import get_all_seasons_all_teams_starters_stats

# Import preprocessing function
from ml_logic.preprocessor import preprocess_features
from xgboost import XGBRegressor


def preprocess():
    """
        Preprocess X
    """
    # Load preprocessed data for
    dfs = load_data()
    X = player_full_data_df(dfs, 1997)

    # Process data
    X_preprocessed = preprocess_features(X)

    return X_preprocessed

def train(model_type, split_ratio):
    """
        Trains model
    """
    X_preprocessed = preprocess()

    all_season_team_starters_stats_flattened, season_and_team_key = get_all_seasons_all_teams_starters_stats(X_preprocessed)
    df_preprocessed_teams = pd.concat(
        [pd.DataFrame(season_and_team_key, columns=["season_team"]),
         pd.DataFrame(all_season_team_starters_stats_flattened)], axis = 1)

    # Create (X_train_processed, y_train, X_val_processed, y_val, X_test_preprocessed, y_test)
    test_length = int(len(df_preprocessed_teams) * split_ratio)
    val_length = int((len(df_preprocessed_teams)-test_length) * split_ratio)
    train_length = len(df_preprocessed_teams) - val_length - test_length

    df_train_preprocessed = df_preprocessed_teams.iloc[:train_length, :].sample(frac=1) # Shuffle datasets to improve training
    df_val_preprocessed = df_preprocessed_teams.iloc[train_length: train_length + val_length, :].sample(frac=1)
    df_test_preprocessed = df_preprocessed_teams.iloc[train_length+val_length:, :].sample(frac=1)

    X_train_preprocessed = df_train_preprocessed.iloc[:, :-1]
    X_val_preprocessed = df_val_preprocessed.iloc[:, :-1]
    X_test_preprocessed = df_test_preprocessed.iloc[:, :-1]

    y_train = pd.DataFrame(df_train_preprocessed.iloc[:, -1])
    y_val = pd.DataFrame(df_val_preprocessed.iloc[:, -1])
    y_test = pd.DataFrame(df_test_preprocessed.iloc[:, -1])

    model = initialize_model(model_type)

    model = fit_model(model, X_train_preprocessed, y_train)
    print("✅ train() done \n")

    return model


# def evaluate(
#         min_date:str = '2014-01-01',
#         max_date:str = '2015-01-01',
#         stage: str = "Production"
#     ) -> float:
#     """
#     Evaluate the performance of the latest production model on processed data
#     Return MAE as a float
#     """
#     print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

#     model = load_model(stage=stage)
#     assert model is not None

#     min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
#     max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

#     # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
#     query = f"""
#         SELECT * EXCEPT(_0)
#         FROM `{GCP_PROJECT}`.{BQ_DATASET}.processed_{DATA_SIZE}
#         WHERE _0 BETWEEN '{min_date}' AND '{max_date}'
#     """

#     data_processed_cache_path = Path(f"{LOCAL_DATA_PATH}/processed/processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
#     data_processed = get_data_with_cache(
#         gcp_project=GCP_PROJECT,
#         query=query,
#         cache_path=data_processed_cache_path,
#         data_has_header=False
#     )

#     if data_processed.shape[0] == 0:
#         print("❌ No data to evaluate on")
#         return None

#     data_processed = data_processed.to_numpy()

#     X_new = data_processed[:, :-1]
#     y_new = data_processed[:, -1]

#     metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
#     mae = metrics_dict["mae"]

#     params = dict(
#         context="evaluate", # Package behavior
#         training_set_size=DATA_SIZE,
#         row_count=len(X_new)
#     )

#     save_results(params=params, metrics=metrics_dict)

#     print("✅ evaluate() done \n")

#     return mae


def pred(model, X_new: pd.DataFrame=None):
    """
    Make a prediction using the latest trained model
    """

    # Display
    print("⭐️ Use case: predict\n")

    #Process X_new
    X_processed = preprocess_features(X_new)

    # Initialize new model
    # model = load_model(model_type)
    # assert model is not None

    # Predict
    y_pred = model.predict(X_processed)

    # Print result
    print("\n✅ Prediction: ", y_pred, "\n", "shape is: ", y_pred.shape, "\n")

    return y_pred



if __name__ == '__main__':
    model = train(XGBRegressor(), 0.3)
    dfs = load_data()
    X = player_full_data_df(dfs, 1997)

    #Test d'une ligne au pif
    X_new = X.iloc[[5]]

    pred(model, X_new)
