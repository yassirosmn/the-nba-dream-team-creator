from pathlib import Path
import pandas as pd


def load_data():
    path = Path("/./Users/marcuspernegger/code/yassirosmn/the-nba-dream-team-creator/raw_data")

    file_names = [
        "Player Totals.csv",
        "Player Play By Play.csv",
        "Player Season Info.csv",
        "Player Shooting.csv",
        "Player Career Info.csv"
    ]

    # Only read the 5 CSVs you care about, in that order
    file_paths = [path / name for name in file_names]

    dfs = []
    for p in file_paths:
        try:
            dfs.append(pd.read_csv(p, sep=",", encoding="utf-8"))
        except UnicodeDecodeError:
            # fallback for Windows-1252/Latin-1 files
            dfs.append(pd.read_csv(p, sep=",", encoding="latin1"))

    return dfs


## Merge


# Hector

def column_reduced(list_dataframe,year):
    list_new_df = []

    for df in list_dataframe:
        list_new_df.append(df.query(f"season >= {year}").copy(deep = True))

    return list_new_df

def primary_key_creator(list_dataframe): # ==> "season + team + player_id"
    list_df_with_PM = []
    for df in list_dataframe:
        list_df_with_PM.append(df.apply(lambda row : str(row.season) + '_' + row.team + '_'+ row.player_id , axis =1))
    return list_df_with_PM

def player_full_data_df(list_dataframe,year):
    #in order Total ; Play_by_play ; Season_info ; Shooting ; career_info

    list_dataframe_reduced = column_reduced(list_dataframe[:3],year)
    list_dataframe_reduced_PM = primary_key_creator(list_dataframe_reduced)

    Player_Totals_df_year = list_dataframe_reduced_PM[0].loc[:,['ID','season', 'lg', 'player', 'player_id', 'age', 'team', 'pos', 'g', 'gs',
       'mp', 'fg', 'fga', 'fg_percent', 'x3p', 'x3pa', 'x3p_percent', 'x2p',
       'x2pa', 'x2p_percent', 'e_fg_percent', 'ft', 'fta', 'ft_percent', 'orb',
       'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'trp_dbl']]

    Player_Play_By_Play_df_year = list_dataframe_reduced_PM[1].loc[:, ['ID', 'pg_percent', 'sg_percent', 'sf_percent', 'pf_percent',
       'c_percent', 'on_court_plus_minus_per_100_poss',
       'net_plus_minus_per_100_poss', 'bad_pass_turnover',
       'lost_ball_turnover', 'shooting_foul_committed',
       'offensive_foul_committed', 'shooting_foul_drawn',
       'offensive_foul_drawn', 'points_generated_by_assists', 'and1',
       'fga_blocked']]

    Player_Season_Info_df_year = list_dataframe_reduced_PM[2].loc[:, ['ID',
       'experience']]


    Player_Shooting_df_year = list_dataframe_reduced_PM[3].loc[:, ['ID','avg_dist_fga', 'percent_fga_from_x2p_range',
       'percent_fga_from_x0_3_range', 'percent_fga_from_x3_10_range',
       'percent_fga_from_x10_16_range', 'percent_fga_from_x16_3p_range',
       'percent_fga_from_x3p_range', 'fg_percent_from_x2p_range',
       'fg_percent_from_x0_3_range', 'fg_percent_from_x3_10_range',
       'fg_percent_from_x10_16_range', 'fg_percent_from_x16_3p_range',
       'fg_percent_from_x3p_range', 'percent_assisted_x2p_fg',
       'percent_assisted_x3p_fg', 'percent_dunks_of_fga', 'num_of_dunks',
       'percent_corner_3s_of_3pa', 'corner_3_point_percent',
       'num_heaves_attempted', 'num_heaves_made']]

    Player_Career_Info_Reduced_df= list_dataframe[4][['player_id','ht_in_in','wt']]

    Player_Totals_df_year['rate_rank'] = (Player_Totals_df_year.groupby(['season', 'team'])['gs'].rank(method='first', ascending=False))

    Player_Totals_df_year['starting_5'] = (Player_Totals_df_year['rate_rank'] <= 5).astype(int)

    daaaamboyyyy = Player_Totals_df_year.merge(Player_Season_Info_df_year,how= 'left',on='ID').merge(Player_Play_By_Play_df_year,how= 'left',on='ID').merge(Player_Shooting_df_year,how= 'left',on='ID').merge(Player_Career_Info_Reduced_df,how = 'left',on='player_id')

    return daaaamboyyyy


dfs = load_data()

player_full_data = player_full_data_df(dfs, 1997)
