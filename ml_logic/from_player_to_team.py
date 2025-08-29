# For each season and team get the ID of players that are in starting 5

from ml_logic.data import player_full_data_df, load_data
from ml_logic.preprocessor import preprocess_features
import pandas as pd
import numpy as np


def get_starters_stats_per_season_per_team(season: int,
                                           team: str,
                                           data_preprocessed: pd.DataFrame):
    """
    Get all the starters's stats per season per team
    Returns a DataFrame
    """
    # take only season and team from full table
    players_of_one_team_of_one_season = data_preprocessed.loc[
        (data_preprocessed["season"] == season) & (data_preprocessed["team"] == team)]
    # take players only if starting_5 == 1 (the 5 players that played the most)
    starters_stats_per_season_per_team = players_of_one_team_of_one_season.loc[players_of_one_team_of_one_season["starting_5"] == 1]
    return starters_stats_per_season_per_team



def get_filtered_starters_stats_per_season_per_team(season: int,
                                                    team: str,
                                                    data_preprocessed: pd.DataFrame,
                                                    stats_filtered: list =['ft', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pts'],
                                                                            #['season', 'age', 'g', 'gs', 'mp', 'fg', 'fga', 'x3p', 'x3pa', 'x2p',
                                                                            # 'x2pa', 'ft', 'fta', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pf',
                                                                            # 'pts', 'rate_rank', 'starting_5', 'experience', 'bad_pass_turnover',
                                                                            # 'lost_ball_turnover', 'shooting_foul_committed',
                                                                            # 'offensive_foul_committed', 'shooting_foul_drawn',
                                                                            # 'offensive_foul_drawn', 'avg_dist_fga', 'percent_fga_from_x2p_range',
                                                                            # 'percent_fga_from_x0_3_range', 'percent_fga_from_x3_10_range',
                                                                            # 'percent_fga_from_x10_16_range', 'percent_fga_from_x16_3p_range',
                                                                            # 'percent_fga_from_x3p_range', 'fg_percent_from_x2p_range',
                                                                            # 'fg_percent_from_x0_3_range', 'fg_percent_from_x3_10_range',
                                                                            # 'fg_percent_from_x10_16_range', 'fg_percent_from_x16_3p_range',
                                                                            # 'fg_percent_from_x3p_range', 'percent_dunks_of_fga', 'num_of_dunks',
                                                                            # 'percent_corner_3s_of_3pa', 'ht_in_in', 'wt'],
                                                    number_of_players: int = 5) -> list:
    """
    Get filtered starters's stats per season per team
    Returns a list
    """
    filtered_starters_stats_per_season_per_team = []
    starters_stats = get_starters_stats_per_season_per_team(season, team, data_preprocessed)
    filtered_starters_stats = starters_stats[stats_filtered]
    for _, player in filtered_starters_stats.iterrows():
        filtered_starters_stats_per_season_per_team.append(player.values)
    return filtered_starters_stats_per_season_per_team


def get_all_seasons_all_teams_starters_stats(X_preprocessed: pd.DataFrame) :
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

    all_season_team_starters_stats_flattened = \
        [np.concatenate([x if isinstance(x, np.ndarray) else np.array([x]) \
        for x in row]) for row in all_season_team_starters_stats
         ]

    return all_season_team_starters_stats_flattened, season_and_team_key

# Tests
if __name__ == "__main__":


    dfs = load_data()
    X = player_full_data_df(dfs, 1997)
    X_preprocessed = preprocess_features(X)
    temp1, temp2 = get_all_seasons_all_teams_starters_stats(X_preprocessed)
    print(pd.DataFrame(temp1))

    print("Test good (✅ pour Flavian)")
