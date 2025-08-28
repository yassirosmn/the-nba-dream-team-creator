# For each season and team get the ID of players that are in starting 5

from ml_logic.data import player_full_data_df, load_data
import pandas as pd

# Constants

list_dataframe = load_data()
players_full_data = player_full_data_df(list_dataframe, 1997)


# For season, team get 5 IDs

def get_season_team_players(season: int, team: str, number_of_players: int = 5):

    season_team_players_id = []

    # take only season and team from full table
    sub_table = players_full_data.loc[
        (players_full_data["season"] == season) & (players_full_data["team"] == team)]

    # take players only if starting_5 == 1
    starters = sub_table.loc[sub_table["starting_5"] == 1]


    return starters


# For ID get all player stats

def get_season_stats(ID: str, season: int):

    season_stats = players_full_data.loc[(players_full_data["ID"] == ID) &
                                      (players_full_data["season"] == season)]

    return season_stats


# Final team season stats

def get_team_season_stats(season: int, team: str, stats_filtered: list = ["pts", "ast", "stl", "blk", "trb", "tov"], number_of_players: int = 5):

    team_stats = []

    starters_stats = get_season_team_players(season, team, number_of_players)

    filtered_starters_stats = starters_stats[stats_filtered]

    for _, player in filtered_starters_stats.iterrows():
        team_stats.append(player.values)

    return team_stats


# Get all data per season and per team

def get_all_season_team_stats(full_dataset):

    X = []

    for season in full_dataset["season"].unique():

        for team in full_dataset["team"].unique():

            team_season_stats = get_team_season_stats(season, team)

            X.append(team_season_stats)

    return X


# Tests
if __name__ == "__main__":

    get_all_season_team_stats(players_full_data)

    print("Test good (✅ pour Flavian)")
