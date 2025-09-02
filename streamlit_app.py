import streamlit as st
import pandas as pd
from params import DATABASE_PATH
import requests
from api.fast import predict

############################ HEADING ############################

def spinning_basketballs(n: int = 6, size_px: int = 52, spin_s: float = 1.2, bounce_s: float = 1.6):
    balls = "".join(
        f'<span class="ball-bounce" style="animation-delay:{i*0.12}s">'
        f'  <span class="ball-spin" style="font-size:{size_px}px">🏀</span>'
        f'</span>'
        for i in range(n)
    )
    st.markdown(
        f"""
        <style>
          .balls-wrap {{
            display:flex; gap:12px; justify-content:center; align-items:center;
            margin: 8px 0 24px;
          }}
          .ball-bounce {{
            display:inline-block;
            animation: bounce {bounce_s}s ease-in-out infinite;
          }}
          .ball-spin {{
            display:inline-block;
            animation: spin {spin_s}s linear infinite;
            filter: drop-shadow(0 2px 2px rgba(0,0,0,.15));
          }}
          @keyframes spin {{
            to {{ transform: rotate(360deg); }}
          }}
          @keyframes bounce {{
            0%,100% {{ transform: translateY(0); }}
            50%     {{ transform: translateY(-8px); }}
          }}
        </style>
        <div class="balls-wrap">{balls}</div>
        """,
        unsafe_allow_html=True,
    )

spinning_basketballs(n=1, size_px=100)

st.markdown("# What is your NBA dream team?")

############################ DROPDOWNS ############################

@st.cache_data
def load_pickles_to_df(pos):
    return pd.read_pickle(f"{DATABASE_PATH}X_2025_{pos}.pkl")

def team_player_picker(df: pd.DataFrame, pos_label: str, key_prefix: str):
    """Render two selectboxes side-by-side: Team (left) -> Player (right)."""
    teams = sorted(df["team"].dropna().unique())

    col_team, col_player = st.columns([1, 2], vertical_alignment="center")
    with col_team:
        team = st.selectbox(f"{pos_label} — Team", teams, key=f"{key_prefix}_team")

    # Players filtered by the selected team
    players = (
        df.loc[df["team"] == team, "player"]
          .dropna()
          .sort_values()
          .unique()
          .tolist()
    )

    # Key depends on team so the player box resets when team changes
    with col_player:
        player = st.selectbox(f"{pos_label} — Player", players, key=f"{key_prefix}_player_{team}")

    return player, team

def get_select_dream_team():
    df_2025_C  = load_pickles_to_df("C")
    df_2025_SG = load_pickles_to_df("SG")
    df_2025_PF = load_pickles_to_df("PF")
    df_2025_PG = load_pickles_to_df("PG")
    df_2025_SF = load_pickles_to_df("SF")

    player_C_name,  player_C_team  = team_player_picker(df_2025_C,  "Select your Center",         "C")
    player_SG_name, player_SG_team = team_player_picker(df_2025_SG, "Select your Shooting Guard", "SG")
    player_PF_name, player_PF_team = team_player_picker(df_2025_PF, "Select your Power Forward",  "PF")
    player_PG_name, player_PG_team = team_player_picker(df_2025_PG, "Select your Point Guard",    "PG")
    player_SF_name, player_SF_team = team_player_picker(df_2025_SF, "Select your Small Forward",  "SF")

    dream_team = {
        player_C_name:  player_C_team,
        player_SG_name: player_SG_team,
        player_PF_name: player_PF_team,
        player_PG_name: player_PG_team,
        player_SF_name: player_SF_team
    }
    return dream_team

selected_dream_team = get_select_dream_team()

############################ PREDICT BUTTON & CALL API ############################

# # BUTTON

if st.button('Will you win the next NBA season?'):

    #response = requests.get(url, params=params)

    response = predict(selected_dream_team)

    if response > 0.3:

        st.success(f"Super team - you are likely to win the season by {response}  🔥 🎉")

        st.balloons()

    else:
        st.error(f"Loser team - you are likely to win the season only by {response}  🫠")


############################ TEMPLATES ############################

# url = 'https://the-nba-dream-team.lewagon.ai/predict'               # To do: Update with correct URL


# if response.ok:
#     prediction = response.json()
#     #probability = prediction.get('fare', None)

#     st.success(f"Your team will win the season with a probability of: {prediction} %")

#     # Balloons
#     if prediction > 0.4:
#         st.balloons()
#     else:
#         st.error("Loser 😢")
