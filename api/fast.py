import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic import registry, preprocessor, from_player_to_team

app = FastAPI()
app.state.model = registry.load_model(model_type_is_deep=False)

'''
https://www.notion.so/marcus-pernegger/Cr-ation-de-l-API-25d0f87e23cf8086a730e211a263c071?source=copy_link


# Used for taxifare, to check if needed
    # Comment taxifare: Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[“*”],  # Allows all origins
    allow_credentials=True,
    allow_methods=[“*”],  # Allows all methods
    allow_headers=[“*”],  # Allows all headers
)
'''

@app.get("/predict")
def predict(
        player_C_name: str,
        player_C_team: str,
        player_SG_name: str,
        player_SG_team: str,
        player_PF_name: str,
        player_PF_team: str,
        player_PG_name: str,
        player_PG_team: str,
        player_SF_name: str,
        player_SF_team: str,
    ):
    """
        Make a nba score prediction
    """


    ###############                                                             #################
    ######                      CHANGER LA PARTIE DATA => IMPORTER SCALER                   #####
    ######                      TRANSFORM LE X_new                                          #####
    ###############




    # Load preprocessed data
    X_preprocessed = registry.load_preprocessed_data_from_database()

    # Load DFs from database
    df_2025_C, df_2025_SG, df_2025_PF, df_2025_PG, df_2025_SF = registry.load_dfs_from_database()

    Dico_players_selected =  {
            player_C_name: player_C_team,       # To do:
            player_SG_name: player_SG_team,         # Create function that lets you select combination of player name and position -> player_C, player_SG, player_PF, player_PG, player_SF
            player_PF_name: player_PF_team,         # User selects the players that will be X_new
            player_PG_name: player_PG_team,
            player_SF_name: player_SF_team,
        }

    # Get players stats
    X_new_embedded, X_new_flattened = from_player_to_team.get_new_team_stats_per_season(Dico_players_selected, X_preprocessed)






    #Predict (contains the model loading)
    y_pred = app.state.model.predict(X_new_flattened)

    return {
            "Your team probability to win the NBA is:" : round(float(y_pred[0]), 2)
            }


@app.get("/")
def root():
    return {"ok": True}


## To do after above code is done:
    # TERMINAL:
        # uvicorn fast:app --reload
        # CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
    # Build, tag, push to Artifact Registry, then deploy to Cloud Run (see steps in the course)
