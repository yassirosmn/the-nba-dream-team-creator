# For each season and team get the ID of players that are in starting 5

def get_season_team_players(season: int, team: str, number_of_players: int = 5):

    season_team_players_id = []

    # take only season and team from full table
    sub_table = players_full_data.loc[
        (players_full_data["season"] == season) & (players_full_data["team"] == team)]

    # take players only if starting_5 == 1
    starters = sub_table.loc[sub_table["starting_5"] == 1]

    # add to intial list
    season_team_players_id = starters["ID"].tolist()

    return season_team_players_id

# For ID get all player stats

def get_season_stats(ID: str, season: int):

    sub_table = players_full_data.loc[(players_full_data["ID"] == ID) &
                                      (players_full_data["season"] == season)]

    return sub_table


################ WORKSPACE / WIP ################

# For a list of IDs get all the stats

def get_team_stats(players: List[PlayerID], season: int):
    XX 
