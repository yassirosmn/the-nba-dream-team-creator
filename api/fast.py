import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ml_logic import registry, preprocessor

app = FastAPI()
#app.state.model = registry.load_model()

################################ WIP ################################
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

@app.get(“/predict”)
def predict(player_C, player_SG, player_PF, player_PG, player_SF):

    X_new =  pd.DataFrame({
        "player_C": [player_C],       # To do:
        "player_SG": [player_SG],         # Create function that lets you select combination of player name and position -> player_C, player_SG, player_PF, player_PG, player_SF
        "player_PF": [player_PF],         # User selects the players that will be X_new
        "player_PG": [player_PG],
        "player_SF": [player_SF],
    })

    X_new_preprocessed = preprocess_features(X_new)

    y_pred = app.state.model.predict(X_new_preprocessed)

    return {"Your team probability to win the NBA is:" float(y_pred[0])}
'''

@app.get("/")
def root():
    return {"ok": True}


## To do after above code is done:
    # TERMINAL:
        # uvicorn fast:app --reload
        # CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
    # Build, tag, push to Artifact Registry, then deploy to Cloud Run (see steps in the course)
