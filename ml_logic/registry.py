import pickle
import pandas as pd
import time
import glob

from ml_logic.data import load_data, player_full_data_df
from params import *
from tensorflow import keras



def load_csvs_and_save_data_to_database() -> None:
    '''
        Saves the full database (after merging all DFs) in local
        Creates filtered DFs by position in 2025, and stores them to database
    '''
    print("⏳ Saving database locally... ⏳")
    df = load_data()
    X = player_full_data_df(df, 1997)
    # Create "models "folder if not existing
    all_data_folder = Path(DATABASE_PATH)
    all_data_folder.mkdir(parents=True, exist_ok=True)
    X.to_pickle(f"{DATABASE_PATH}player_full_database.pkl")
    print("✅ Database saved locally !")

    print("⏳ Saving 2025 filtered DFs locally... ⏳")
    X_2025_C = X.query("season == 2025 & pos == 'C'")
    X_2025_C.to_pickle(f"{DATABASE_PATH}X_2025_C.pkl")

    X_2025_SG = X.query("season == 2025 & pos == 'SG'")
    X_2025_SG.to_pickle(f"{DATABASE_PATH}X_2025_SG.pkl")

    X_2025_PF = X.query("season == 2025 & pos == 'PF'")
    X_2025_PF.to_pickle(f"{DATABASE_PATH}X_2025_PF.pkl")

    X_2025_PG = X.query("season == 2025 & pos == 'PG'")
    X_2025_PG.to_pickle(f"{DATABASE_PATH}X_2025_PG.pkl")

    X_2025_SF = X.query("season == 2025 & pos == 'SF'")
    X_2025_SF.to_pickle(f"{DATABASE_PATH}X_2025_SF.pkl")

    print("✅ 2025 filtered DFs saved locally")

def load_dfs_from_database() -> pd.DataFrame:
    '''
        Loads filtered DFs by position for 2025 from the database
    '''
    print("⏳ Loading locally saved 2025 filtered DFs.. ⏳")

    try:
        df_2025_C = pd.read_pickle(f"{DATABASE_PATH}X_2025_C.pkl")
        df_2025_SG = pd.read_pickle(f"{DATABASE_PATH}X_2025_SG.pkl")
        df_2025_PF = pd.read_pickle(f"{DATABASE_PATH}X_2025_PF.pkl")
        df_2025_PG = pd.read_pickle(f"{DATABASE_PATH}X_2025_PG.pkl")
        df_2025_SF = pd.read_pickle(f"{DATABASE_PATH}X_2025_SF.pkl")

        print("✅ 2025 filtered DFs loaded from local !")

    except:
            print(f"\n❌❌ No DFs found at path : {DATABASE_PATH}")
            pass

    return df_2025_C, df_2025_SG, df_2025_PF, df_2025_PG, df_2025_SF

def load_data_from_database() -> pd.DataFrame:
    '''
        Loads the full database
    '''
    print("⏳ Loading Database.. ⏳")

    try:
        df = pd.read_pickle(f"{DATABASE_PATH}player_full_database.pkl")
        print("✅ Database loaded from local !")

    except:
            print(f"\n❌❌ No database found at path : {DATABASE_PATH}")
            return None

    return df

def save_data(df: pd.DataFrame, name:str) -> None:
    '''
        Saves the preprocessed to the database
    '''
    print("⏳ Saving preprocessed data.. ⏳")
    df.to_pickle(f"{DATABASE_PATH}{name}.pkl")
    print("✅ Preprocessed data saved locally !")

def load_preprocessed_data_from_database() -> pd.DataFrame:
    '''
        Loads the preprocessed data from the database
    '''
    print("⏳ Loading preprocessed data.. ⏳")
    try:
        df = pd.read_pickle(f"{DATABASE_PATH}data_preprocessed.pkl")
        print("✅ Preprocessed data loaded from local !")

    except:
            print(f"\n❌❌ No preprocessed data found at path : {DATABASE_PATH}")
            return None

    return df

def save_model(model, model_type_is_deep: bool = True) -> None:
    """
    Persist trained model locally on the hard drive at f"{MODEL_PATH}_{timestamp}.h5"
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create "models "folder if not existing
    models_folder = Path(MODEL_PATH)
    models_folder.mkdir(parents=True, exist_ok=True)


    # Save model locally
    if model_type_is_deep:
        ml_folder = Path(MODEL_PATH) / "deep"
        ml_folder.mkdir(parents=True, exist_ok=True)  # Creates folder if not existing
        model.save(f"{MODEL_PATH}/deep/model_deep_{timestamp}.h5")
        print("✅ Model DL saved locally")

    else :
        ml_folder = Path(MODEL_PATH) / "ml"
        ml_folder.mkdir(parents=True, exist_ok=True)  # Creates folder if not existing
        filename = ml_folder / f"model_ml_{timestamp}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print("✅ Model ML saved locally")

    return None

def load_model(model_type_is_deep: bool = True) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    Return None (but do not Raise) if no model is found

    """

    if model_type_is_deep:

        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{MODEL_PATH}/deep/*")

        # Si aucun model trouvé
        if not local_model_paths:
            return None
        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        latest_model_deep = keras.models.load_model(most_recent_model_path_on_disk)
        print("✅ Model loaded from local disk")

        return latest_model_deep

    else:
            folder = Path(f"{MODEL_PATH}/ml")
            # Récupère tous les fichiers .pkl et trie par date de modification décroissante
            pkl_files = sorted(folder.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)

            # Prend le premier (le plus récent)
            latest_file = pkl_files[0]

            # Charge le modèle
            with open(latest_file, "rb") as f:
                latest_model_ml = pickle.load(f)

            print(f"✅ ML model loaded : {latest_file}")

            return latest_model_ml


if __name__ == "__main__":
    # from ml_logic.preprocessor import preprocess_features
    # # Save database
    # load_csvs_and_save_data_to_database()

    # # Load dfs
    # dfs,_,_,_,_ = load_dfs_from_database()
    # print(dfs.head())

    # # Load database from local
    # df = load_data_from_database()

    # # print(df.head())

    # #Preprocess data
    # X_prep = preprocess_features(df)

    # # Save preprocessed data
    # save_preprocessed_data(X_prep)

    # # Load preprocessed data
    # X_preprocessed = load_preprocessed_data_from_database()
    # print(f"\n ➡️ ➡️  Displaying first rows :\n{X_preprocessed.head()}")

    # from ml_logic.data import new_y_creator
    # from interface.main import get_X_y, train_ML
    # y_winrate, y = new_y_creator(1997)
    # df_for_model = get_X_y(X_preprocessed, y_winrate)
    # from sklearn.linear_model import LinearRegression
    # model, X_test_preprocessed, y_test = train_ML(LinearRegression(), df_for_model, 0.3)
    # save_model(model, False)

    # Test de load model
    model = load_model(False)
