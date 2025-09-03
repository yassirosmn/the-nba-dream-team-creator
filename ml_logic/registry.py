import pickle
import pandas as pd
import time
import glob
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = tout, 1 = warnings, 2 = erreurs, 3 = rien sauf erreurs graves

from ml_logic.data import load_data, player_full_data_df
from params import *
# import tensorflow as tf
# tf.get_logger().setLevel("ERROR")



def load_csvs_and_save_data_to_database() -> None:
    '''
        Saves the full database (after merging all DFs) in local
        Creates filtered DFs by position in 2025, and stores them to database
    '''
    print("\n⏳ Saving database locally... ⏳")
    df = load_data()
    X = player_full_data_df(df, 1997)
    # Create "models "folder if not existing
    all_data_folder = Path(DATABASE_PATH)
    all_data_folder.mkdir(parents=True, exist_ok=True)
    X.to_pickle(f"{DATABASE_PATH}player_full_database.pkl")
    print("\n✅ Database saved locally !")

    print("\n⏳ Saving 2025 filtered DFs locally... ⏳")
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

    print("\n✅ 2025 filtered DFs saved locally")

def load_dfs_from_database() -> pd.DataFrame:
    '''
        Loads filtered DFs by position for 2025 from the database
    '''
    print("\n⏳ Loading locally saved 2025 filtered DFs.. ⏳")

    try:
        df_2025_C = pd.read_pickle(f"{DATABASE_PATH}X_2025_C.pkl")
        df_2025_SG = pd.read_pickle(f"{DATABASE_PATH}X_2025_SG.pkl")
        df_2025_PF = pd.read_pickle(f"{DATABASE_PATH}X_2025_PF.pkl")
        df_2025_PG = pd.read_pickle(f"{DATABASE_PATH}X_2025_PG.pkl")
        df_2025_SF = pd.read_pickle(f"{DATABASE_PATH}X_2025_SF.pkl")

        print("\n✅ 2025 filtered DFs loaded from local !")

    except:
            print(f"\n❌❌ No DFs found at path : {DATABASE_PATH}")
            pass

    return df_2025_C, df_2025_SG, df_2025_PF, df_2025_PG, df_2025_SF

def load_data_from_database() -> pd.DataFrame:
    '''
        Loads the full database
    '''
    print("\n⏳ Loading Database.. ⏳")

    try:
        df = pd.read_pickle(f"{DATABASE_PATH}player_full_database.pkl")
        print("\n✅ Database loaded from local !")

    except:
            print(f"\n❌❌ No database found at path : {DATABASE_PATH}")
            return None

    return df

def save_data(df: pd.DataFrame, name:str) -> None:
    '''
        Saves the data to the database
    '''
    print("\n⏳ Saving data.. ⏳")
    df.to_pickle(f"{DATABASE_PATH}{name}.pkl")
    print("\n✅ Data saved locally !")

def load_preprocessed_data_from_database() -> pd.DataFrame:
    '''
        Loads the preprocessed data from the database
    '''
    print("\n⏳ Loading preprocessed data... ⏳")

    # Looks for all file of which name ends by "preprocessed.pkl"
    pickle_files = list(Path(DATABASE_PATH).glob("*preprocessed.pkl"))

    if not pickle_files:
        raise FileNotFoundError(f"\n❌❌ No preprocessed data found in {Path(DATABASE_PATH)}")

    # If there are many files, taking the most recent
    latest_preprocessed_data = max(pickle_files, key=lambda f: f.stat().st_mtime)

    print(f"\n✅ Loaded preprocessed data : {latest_preprocessed_data}")

    df = pd.read_pickle(latest_preprocessed_data)

    return df

def save_model(model, model_type_is_deep: bool = True) -> None:
    """
    Saves trained model locally on the hard drive at :
        - f"{MODEL_PATH}_{timestamp}.h5" if model is deep
        - f"model_ml_{timestamp}.pkl"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Create "models "folder if not existing
    models_folder = Path(MODEL_PATH)
    models_folder.mkdir(parents=True, exist_ok=True)


    # Save model locally
    if model_type_is_deep:
        ml_folder = Path(MODEL_PATH) / "deep"
        ml_folder.mkdir(parents=True, exist_ok=True)  # Creates folder if not existing

        model.save(f"{MODEL_PATH}/deep/model_deep_{timestamp}.keras")
        print("\n✅ Model DL saved locally")

    else :
        ml_folder = Path(MODEL_PATH) / "ml"
        ml_folder.mkdir(parents=True, exist_ok=True)  # Creates folder if not existing
        filename = ml_folder / f"model_ml_{timestamp}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print("\n✅ Model ML saved locally")

    return None

def load_model(model_type_is_deep: bool = True): # tf.keras.Model
    """
        Returns a locally saved model (latest one in alphabetical order)
        Returns None (but do not Raise) if no model is found
    """

    if model_type_is_deep:

        folder = Path(MODEL_PATH) / "deep"

        # Cherche les fichiers .keras et .h5
        model_files = list(folder.glob("model_deep_*.keras"))

        if not model_files:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {folder}")

        # Trie par date de modification (plus récent en premier)
        latest_file = max(model_files, key=lambda f: f.stat().st_mtime)

        print(f"\n✅ Deep Learning model loaded : {latest_file}")

        model_deep = tf.keras.models.load_model(latest_file)
        return model_deep


    else:
            folder = Path(f"{MODEL_PATH}/ml")
            # Récupère tous les fichiers .pkl et trie par date de modification décroissante
            pkl_files = sorted(folder.glob("*.pkl"), key=lambda f: f.stat().st_mtime, reverse=True)

            # Prend le premier (le plus récent)
            latest_file = pkl_files[0]

            # Charge le modèle
            with open(latest_file, "rb") as f:
                latest_model_ml = pickle.load(f)

            print(f"\n✅ Machine Learning model loaded : {latest_file}")

            return latest_model_ml


if __name__ == "__main__":
    from ml_logic.preprocessor import preprocess_features
    # # Save database
    load_csvs_and_save_data_to_database()

    # # # Load dfs
    dfs,_,_,_,_ = load_dfs_from_database()
    print(dfs.head())

    # # Load database from local
    df = load_data_from_database()

    # # print(df.head())

    # #Preprocess data
    X_1997_2024_preprocessed, X_2025_transformed = preprocess_features(df)

    # # Save preprocessed data
    save_data(X_1997_2024_preprocessed, "data_preprocessed")

    # # Load preprocessed data
    X_1997_2024_preprocessed = load_preprocessed_data_from_database()
    print(f"\n ➡️ ➡️  Displaying first rows :\n{X_1997_2024_preprocessed.head()}")

    from ml_logic.data import new_y_creator
    from interface.main import get_X_y, train_ML
    _,y2,y_winrate_2025, y = new_y_creator(1997)
    df_for_model = get_X_y(X_1997_2024_preprocessed, y2)
    from sklearn.linear_model import LinearRegression
    model = train_ML(LinearRegression(), df_for_model, 0.3)
    # model, X_test_preprocessed, y_test = train_DL(LinearRegression(), df_for_model, 0.3)
    save_model(model, False)

    # Test de load model
    model = load_model(False)
