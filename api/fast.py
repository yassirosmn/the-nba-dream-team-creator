import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic import registry, from_player_to_team
from params import *

app = FastAPI()
app.state.model = registry.load_model(model_type_is_deep=True)



@app.get("/predict")
def predict(dream_team):
    """
        Make a nba score prediction
    """
    # Load 2025 scaled data
    try:
        X_2025_scaled = pd.read_pickle(f"{DATABASE_PATH}X_2025_transformed.pkl")
        print("✅ Scaled 2025 loaded from local !", X_2025_scaled.head())

    except:
            print(f"\n❌❌ No 2025 scaled data found at path : {DATABASE_PATH}")

    # # Load DFs from database
    df_2025_C, df_2025_SG, df_2025_PF, df_2025_PG, df_2025_SF = registry.load_dfs_from_database()

    # ###############                                                             #################
    # ######                      CHANGER LA PARTIE DATA => IMPORTER SCALER                   #####
    # ######                      TRANSFORM LE X_new                                          #####
    # ###############


    # Get players stats
    X_new_embedded, X_new_flattened = from_player_to_team.get_new_team_stats_per_season(dream_team, X_2025_scaled)

    ##################  Predict (contains the model loading)  ##################
    y_pred = app.state.model.predict(X_new_embedded)

    # Show prediction
    return {
            "Your team probability to win the NBA is" : round(float(y_pred[0]), 2)
            }


@app.get("/")
def root():
    return {"ok": True}


## To do after above code is done:
    # TERMINAL:
        # uvicorn fast:app --reload
        # CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
    # Build, tag, push to Artifact Registry, then deploy to Cloud Run (see steps in the course)
