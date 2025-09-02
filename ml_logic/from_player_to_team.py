# For each season and team get the ID of players that are in starting 5

from ml_logic.data import player_full_data_df, load_data
from ml_logic.preprocessor import preprocess_features
from ml_logic.registry import load_preprocessed_data_from_database
from ml_logic.embedder import  player_embedder_transform
from params import *
import pandas as pd
import numpy as np




def get_player_stats_per_team_per_season(player: str,
                                           team: str,
                                           preprocessed_database_players_2025: pd.DataFrame,
                                           season: int = 2025)-> pd.DataFrame:
    """
    Get all the player's stats per season per team of last season (2025)
    """
    # take 1 player for 1 season from 1 team from full table
    stats_of_player_of_one_team_of_one_season = preprocessed_database_players_2025.loc[
        (preprocessed_database_players_2025["season"] == season) & (preprocessed_database_players_2025["team"] == team) & (preprocessed_database_players_2025["player"] == player)]
    return stats_of_player_of_one_team_of_one_season

# def create_dico_player_and_team(player: str, team:str)-> dict:

#     my_dico = {}

#     return my_dico

def flatten_df(df: pd.DataFrame)-> pd.DataFrame:
    df_flat = pd.DataFrame([df.values.flatten()],
                       columns=[f"{col}" for i in range(len(df)) for col in df.columns])
    return df_flat


def get_new_team_stats_per_season(dico_player_and_team: dict,
                                           data_preprocessed: pd.DataFrame,
                                           season: int = 2025):
    """
    Get new team player's stats per season per team of last season (2025)
    """
    the_n_players = pd.DataFrame()
    for player,team in dico_player_and_team.items():
        player_stats = get_player_stats_per_team_per_season(player, team, data_preprocessed)
        the_n_players = pd.concat([the_n_players, player_stats], ignore_index=True)

    #Embedding for Deep
    ## create list for embedding
    the_n_players = the_n_players[STATS_TO_KEEP]
    the_n_players_list = the_n_players.values.tolist()
    the_n_players_embedded = player_embedder_transform(the_n_players_list)
    # Flatten df for ML
    the_n_players_flattened = flatten_df(the_n_players)

    return the_n_players_embedded, the_n_players_flattened

#########################


def get_starters_stats_per_season_per_team(season: int,
                                           team: str,
                                           data_preprocessed: pd.DataFrame,
                                           starting_5: bool = False):
    """
    Get all the starters's stats per season per team
    Returns a DataFrame
    """
    # take only season and team from full table
    players_stats_of_one_team_of_one_season = data_preprocessed.loc[
        (data_preprocessed["season"] == season) & (data_preprocessed["team"] == team)]
    # take players only if starting_5 == 1 (the 5 players that played the most)
    starters_stats_of_one_season_of_one_team = players_stats_of_one_team_of_one_season.loc[players_stats_of_one_team_of_one_season["starting_5"] == 1]
    if starting_5 :
        return starters_stats_of_one_season_of_one_team
    else :
        return players_stats_of_one_team_of_one_season


def get_filtered_starters_stats_per_season_per_team(season: int,
                                                    team: str,
                                                    data_preprocessed: pd.DataFrame,
                                                    stats_filtered: list = STATS_TO_KEEP,
                                                    number_of_players: int = 5) -> list:
    """
    Get filtered starters's stats per season per team
    Returns a list
    """
    filtered_starters_stats_per_season_per_team = []
    starters_stats = get_starters_stats_per_season_per_team(season, team, data_preprocessed, True)
    filtered_starters_stats = starters_stats[stats_filtered]
    for _, player in filtered_starters_stats.iterrows():
        filtered_starters_stats_per_season_per_team.append(player.values)
    return filtered_starters_stats_per_season_per_team


def get_all_seasons_all_teams_starters_stats(X_preprocessed: pd.DataFrame, ML = True) :
    """
    Get filtered starters's stats for ALL season for ALL teams
    Returns 2 lists
    """
    all_season_team_starters_stats = []
    season_and_team_key = []
    all_season_team_starters_stats_flattened = []

    for season in X_preprocessed["season"].unique():
        for team in X_preprocessed["team"].unique():
            # Get the straters's stats for this season and this team :
            team_season_stats = get_filtered_starters_stats_per_season_per_team(season, team, X_preprocessed)

            # If there are stats for this season and team :
            if team_season_stats != [] :

                # Append it to a list :
                all_season_team_starters_stats.append(team_season_stats)

                # Create a Key SeasonTeam :
                key = str(season) + "_" + team
                season_and_team_key.append(key)

    if ML == False :
        all_season_team_starters_stats_embedded = player_embedder_transform(np.array(all_season_team_starters_stats))
        all_season_team_starters_stats_embedded_flattened = \
            [np.concatenate([x if isinstance(x, np.ndarray) else np.array([x]) \
            for x in row]) for row in all_season_team_starters_stats_embedded
            ]
        return all_season_team_starters_stats_embedded_flattened, season_and_team_key, all_season_team_starters_stats

    else :
        all_season_team_starters_stats_flattened = \
            [np.concatenate([x if isinstance(x, np.ndarray) else np.array([x]) \
            for x in row]) for row in all_season_team_starters_stats
            ]
        return all_season_team_starters_stats_flattened, season_and_team_key


# Tests
if __name__ == "__main__":
    from interface.main import load_and_preprocess_and_save, train_DL, train_ML, get_X_y, pred
    from sklearn.linear_model import LinearRegression
    from ml_logic.data import new_y_creator
    X = load_preprocessed_data_from_database()
    # Stat1 = get_player_stats_per_team_per_season("Bam Adebayo", "MIA", X, 2025)
    # print(Stat1)
    print("✅✅")
    dico = {
        "Precious Achiuwa":"NYK",
        "Ochai Agbaji":"TOR",
        "Santi Aldama":"MEM",
        "Patrick Baldwin Jr.":"LAC",
        "Armel Traoré":"LAL"
    }
    the_n_players_embedded, the_n_players_flattened = get_new_team_stats_per_season(dico, X, 2025)
    print(the_n_players_embedded, np.shape(the_n_players_embedded))
    # X_preprocessed = preprocess_features_and_save(X)
    # temp1, temp2 = get_all_seasons_all_teams_starters_stats(X_preprocessed)
    # print(pd.DataFrame(temp1))
    y_win_rate, y = new_y_creator(1997)
    df_X_y = get_X_y(X, y_win_rate)
    model, X_test_preproc, y_test = train_ML(LinearRegression(),df_X_y, 0.1)

    y_pred = pred(model, the_n_players_flattened)
    print("✅✅✅✅✅", y_pred)
    print("Test good (✅ pour Flavian)")
