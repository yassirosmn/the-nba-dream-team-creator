import pickle
import pandas as pd

from ml_logic.data import load_data, player_full_data_df


def load_csvs_and_save_data_to_database() -> None:
    '''
        Saves the full database (after merging all DFs) in local
    '''
    print("⏳ Saving to Database... ⏳")
    df = load_data()
    X = player_full_data_df(df, 1997)
    X.to_pickle("./database_folder/player_full_database.pkl")
    print("✅ Saved to database !")

    X_2025_C = X.query("season == 2025 & pos == 'C'")
    X_2025_C.to_pickle("./database_folder/X_2025_C.pkl")

    X_2025_SG = X.query("season == 2025 & pos == 'SG'")
    X_2025_SG.to_pickle("./database_folder/X_2025_SG.pkl")

    X_2025_PF = X.query("season == 2025 & pos == 'PF'")
    X_2025_PF.to_pickle("./database_folder/X_2025_PF.pkl")

    X_2025_PG = X.query("season == 2025 & pos == 'PG'")
    X_2025_PG.to_pickle("./database_folder/X_2025_PG.pkl")

    X_2025_SF = X.query("season == 2025 & pos == 'SF'")
    X_2025_SF.to_pickle("./database_folder/X_2025_SF.pkl")

    print("✅ Pickles for positions saved")


def load_dfs_from_database() -> pd.DataFrame:
    '''
        Saves the full database (after merging all DFs) in local
    '''
    print("⏳ Loading Database.. ⏳")

    try:
        df_2025_C = pd.read_pickle("./database_folder/X_2025_C.pkl")
        df_2025_SG = pd.read_pickle("./database_folder/X_2025_SG.pkl")
        df_2025_PF = pd.read_pickle("./database_folder/X_2025_PF.pkl")
        df_2025_PG = pd.read_pickle("./database_folder/X_2025_PG.pkl")
        df_2025_SF = pd.read_pickle("./database_folder/X_2025_SF.pkl")

        print("✅ Database loaded !")

    except:
            print(f"\n❌❌ No database found at path : ./database_folder/")
            pass

    return df_2025_C, df_2025_SG, df_2025_PF, df_2025_PG, df_2025_SF

def load_data_from_database() -> pd.DataFrame:
    '''
        Saves the full database (after merging all DFs) in local
    '''
    print("⏳ Loading Database.. ⏳")

    try:
        df = pd.read_pickle("./database_folder/player_full_database.pkl")
        print("✅ Database loaded !")

    except:
            print(f"\n❌❌ No database found at path : ./database_folder/")
            return None

    return df

def save_preprocessed_data(df: pd.DataFrame) -> None:
    '''
        Saves the full database (after merging all DFs) in local
    '''
    print("⏳ Saving preprocessed data.. ⏳")
    df.to_pickle("./database_folder/data_preprocessed.pkl")
    print("✅ Preprocessed data saved to database !")

def load_preprocessed_data_from_database() -> pd.DataFrame:
    '''
        Saves the full database (after merging all DFs) in local
    '''
    print("⏳ Loading preprocessed data.. ⏳")
    try:
        df = pd.read_pickle("./database_folder/data_preprocessed.pkl")
        print("✅ Preprocessed data loaded from database !")

    except:
            print(f"\n❌❌ No preprocessed data found at path : ./database_folder/")
            return None

    return df


# def save_model(model: keras.Model = None) -> None:
#     """
#     Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
#     - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
#     - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
#     """

#     timestamp = time.strftime("%Y%m%d-%H%M%S")

#     # Save model locally
#     model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
#     model.save(model_path)

#     print("✅ Model saved locally")

#     if MODEL_TARGET == "gcs":
#         # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

#         model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
#         client = storage.Client()
#         bucket = client.bucket(BUCKET_NAME)
#         blob = bucket.blob(f"models/{model_filename}")
#         blob.upload_from_filename(model_path)

#         print("✅ Model saved to GCS")

#         return None

#     if MODEL_TARGET == "mlflow":
#         mlflow.tensorflow.log_model(
#             model=model,
#             artifact_path="model",
#             registered_model_name=MLFLOW_MODEL_NAME
#         )

#         print("✅ Model saved to MLflow")

#         return None

#     return None


# def load_model(stage="Production") -> keras.Model:
#     """
#     Return a saved model:
#     - locally (latest one in alphabetical order)
#     - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
#     - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

#     Return None (but do not Raise) if no model is found

#     """

#     if MODEL_TARGET == "local":
#         print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

#         # Get the latest model version name by the timestamp on disk
#         local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
#         local_model_paths = glob.glob(f"{local_model_directory}/*")

#         if not local_model_paths:
#             return None

#         most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

#         print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

#         latest_model = keras.models.load_model(most_recent_model_path_on_disk)

#         print("✅ Model loaded from local disk")

#         return latest_model

#     elif MODEL_TARGET == "gcs":
#         # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
#         print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

#         client = storage.Client()
#         blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

#         try:
#             latest_blob = max(blobs, key=lambda x: x.updated)
#             latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
#             latest_blob.download_to_filename(latest_model_path_to_save)

#             latest_model = keras.models.load_model(latest_model_path_to_save)

#             print("✅ Latest model downloaded from cloud storage")

#             return latest_model
#         except:
#             print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

#             return None

#     elif MODEL_TARGET == "mlflow":
#         print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

#         # Load model from MLflow
#         model = None
#         mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#         client = MlflowClient()

#         try:
#             model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
#             model_uri = model_versions[0].source

#             assert model_uri is not None
#         except:
#             print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

#             return None

#         model = mlflow.tensorflow.load_model(model_uri=model_uri)

#         print("✅ Model loaded from MLflow")
#         return model
#     else:
#         return None


# def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
#     """
#     Transition the latest model from the `current_stage` to the
#     `new_stage` and archive the existing model in `new_stage`
#     """
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#     client = MlflowClient()

#     version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

#     if not version:
#         print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
#         return None

#     client.transition_model_version_stage(
#         name=MLFLOW_MODEL_NAME,
#         version=version[0].version,
#         stage=new_stage,
#         archive_existing_versions=True
#     )

#     print(f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

#     return None


# def mlflow_run(func):
#     """
#     Generic function to log params and results to MLflow along with TensorFlow auto-logging

#     Args:
#         - func (function): Function you want to run within the MLflow run
#         - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
#         - context (str, optional): Param describing the context of the run. Defaults to "Train".
#     """
#     def wrapper(*args, **kwargs):
#         mlflow.end_run()
#         mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#         mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

#         with mlflow.start_run():
#             mlflow.tensorflow.autolog()
#             results = func(*args, **kwargs)

#         print("✅ mlflow_run auto-log done")

#         return results
#     return wrapper


if __name__ == "__main__":
    from ml_logic.preprocessor import preprocess_features
    # Save database
    load_csvs_and_save_data_to_database()

    # Load database from local
    df = load_data_from_database()

    print(df.head())

    dfs,_,_,_,_ = load_dfs_from_database()

    print(dfs[["pos"]])

    '''# Preprocess data
    X_prep = preprocess_features(df)

    # Save preprocessed data
    save_preprocessed_data(X_prep)

    # Load preprocessed data
    X = load_preprocessed_data_from_database()
    print(f"\n ➡️ ➡️  Displaying first rows :\n{X.head()}")'''
