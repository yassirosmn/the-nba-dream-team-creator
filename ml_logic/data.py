from pathlib import Path
import pandas as pd
import numpy as np
from params import *


##################  CONSTANTS  #####################

COLUMN_NAMES_RAW = ['ID', 'season', 'lg', 'player', 'player_id', 'age', 'team', 'pos', 'g',
       'gs', 'mp', 'fg', 'fga', 'fg_percent', 'x3p', 'x3pa', 'x3p_percent',
       'x2p', 'x2pa', 'x2p_percent', 'e_fg_percent', 'ft', 'fta', 'ft_percent',
       'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'trp_dbl',
       'rate_rank', 'starting_5', 'experience', 'pg_percent', 'sg_percent',
       'sf_percent', 'pf_percent', 'c_percent',
       'on_court_plus_minus_per_100_poss', 'net_plus_minus_per_100_poss',
       'bad_pass_turnover', 'lost_ball_turnover', 'shooting_foul_committed',
       'offensive_foul_committed', 'shooting_foul_drawn',
       'offensive_foul_drawn', 'points_generated_by_assists', 'and1',
       'fga_blocked', 'avg_dist_fga', 'percent_fga_from_x2p_range',
       'percent_fga_from_x0_3_range', 'percent_fga_from_x3_10_range',
       'percent_fga_from_x10_16_range', 'percent_fga_from_x16_3p_range',
       'percent_fga_from_x3p_range', 'fg_percent_from_x2p_range',
       'fg_percent_from_x0_3_range', 'fg_percent_from_x3_10_range',
       'fg_percent_from_x10_16_range', 'fg_percent_from_x16_3p_range',
       'fg_percent_from_x3p_range', 'percent_assisted_x2p_fg',
       'percent_assisted_x3p_fg', 'percent_dunks_of_fga', 'num_of_dunks',
       'percent_corner_3s_of_3pa', 'corner_3_point_percent',
       'num_heaves_attempted', 'num_heaves_made', 'ht_in_in', 'wt']

DTYPES_RAW = {'ID': 'object', 'season': 'int64', 'lg': 'object',
              'player': 'object', 'player_id': 'object', 'age': 'float64',
              'team': 'object', 'pos': 'object', 'g': 'int64', 'gs': 'float64',
              'mp': 'float64', 'fg': 'int64', 'fga': 'int64', 'fg_percent': 'float64',
              'x3p': 'float64', 'x3pa': 'float64', 'x3p_percent': 'float64',
              'x2p': 'float64', 'x2pa': 'float64', 'x2p_percent': 'float64',
              'e_fg_percent': 'float64', 'ft': 'int64', 'fta': 'int64', 'ft_percent': 'float64',
              'orb': 'float64', 'drb': 'float64', 'trb': 'float64', 'ast': 'int64',
              'stl': 'float64', 'blk': 'float64', 'tov': 'float64', 'pf': 'float64',
              'pts': 'int64', 'trp_dbl': 'float64', 'rate_rank': 'float64',
              'starting_5': 'int64', 'experience': 'int64', 'pg_percent': 'float64',
              'sg_percent': 'float64', 'sf_percent': 'float64', 'pf_percent': 'float64',
              'c_percent': 'float64', 'on_court_plus_minus_per_100_poss': 'float64',
              'net_plus_minus_per_100_poss': 'float64', 'bad_pass_turnover': 'int64',
              'lost_ball_turnover': 'int64', 'shooting_foul_committed': 'int64',
              'offensive_foul_committed': 'int64', 'shooting_foul_drawn': 'int64',
              'offensive_foul_drawn': 'float64', 'points_generated_by_assists': 'int64',
              'and1': 'int64', 'fga_blocked': 'int64', 'avg_dist_fga': 'float64',
              'percent_fga_from_x2p_range': 'float64', 'percent_fga_from_x0_3_range': 'float64',
              'percent_fga_from_x3_10_range': 'float64', 'percent_fga_from_x10_16_range': 'float64',
              'percent_fga_from_x16_3p_range': 'float64', 'percent_fga_from_x3p_range': 'float64',
              'fg_percent_from_x2p_range': 'float64', 'fg_percent_from_x0_3_range': 'float64',
              'fg_percent_from_x3_10_range': 'float64', 'fg_percent_from_x10_16_range': 'float64',
              'fg_percent_from_x16_3p_range': 'float64', 'fg_percent_from_x3p_range': 'float64',
              'percent_assisted_x2p_fg': 'float64', 'percent_assisted_x3p_fg': 'float64',
              'percent_dunks_of_fga': 'float64', 'num_of_dunks': 'int64', 'percent_corner_3s_of_3pa': 'float64',
              'corner_3_point_percent': 'float64', 'num_heaves_attempted': 'int64', 'num_heaves_made': 'int64',
              'ht_in_in': 'int64', 'wt': 'float64'}


##################  CODE  #####################

def load_data():

    path = FILE_PATH

    file_names = [
        "Player Totals.csv",
        "Player Play By Play.csv",
        "Player Season Info.csv",
        "Player Shooting.csv",
        "Player Career Info.csv"
    ]

    file_paths = [path / name for name in file_names]

    dfs = []
    for p in file_paths:
        try:
            dfs.append(pd.read_csv(p, sep=",", encoding="utf-8"))
        except UnicodeDecodeError: #we had errors, we added this to avoid errors
            dfs.append(pd.read_csv(p, sep=",", encoding="latin1"))

    return dfs

## Reduce columns, create player ID, merge dfs

def column_reduced(list_dataframe,year):
    list_new_df = []

    for df in list_dataframe:
        list_new_df.append(df.query(f"season >= {year}").copy(deep = True))

    return list_new_df

def primary_key_creator(list_dataframe): # ==> "season + team + player_id"
    list_df_with_PM = []

    for df in list_dataframe:
        df["ID"]= df.apply(lambda row : str(row.season) + '_' + row.team + '_'+ row.player_id , axis =1)
        list_df_with_PM.append(df)

    return list_df_with_PM

def player_full_data_df(list_dataframe,year):
    #in order Total ; Play_by_play ; Season_info ; Shooting ; career_info

    list_dataframe_reduced = column_reduced(list_dataframe[:4],year)
    list_dataframe_reduced_PM = primary_key_creator(list_dataframe_reduced)

    Player_Totals_df_year = list_dataframe_reduced_PM[0].loc[:,['ID','season', 'lg', 'player', 'player_id', 'age', 'team', 'pos', 'g', 'gs','mp', 'fg', 'fga', 'fg_percent', 'x3p', 'x3pa', 'x3p_percent', 'x2p',
    'x2pa', 'x2p_percent', 'e_fg_percent', 'ft', 'fta', 'ft_percent', 'orb',
    'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'trp_dbl']]

    Player_Play_By_Play_df_year = list_dataframe_reduced_PM[1].loc[:, ['ID', 'pg_percent', 'sg_percent', 'sf_percent', 'pf_percent','c_percent', 'on_court_plus_minus_per_100_poss',
    'net_plus_minus_per_100_poss', 'bad_pass_turnover',
    'lost_ball_turnover', 'shooting_foul_committed',
    'offensive_foul_committed', 'shooting_foul_drawn',
    'offensive_foul_drawn', 'points_generated_by_assists', 'and1',
    'fga_blocked']]

    Player_Season_Info_df_year = list_dataframe_reduced_PM[2].loc[:, ['ID','experience']]

    Player_Shooting_df_year = list_dataframe_reduced_PM[3].loc[:, ['ID','avg_dist_fga', 'percent_fga_from_x2p_range','percent_fga_from_x0_3_range', 'percent_fga_from_x3_10_range',
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

    merged_dfs = Player_Totals_df_year.merge(Player_Season_Info_df_year,how= 'left',on='ID').merge(Player_Play_By_Play_df_year,how= 'left',on='ID').merge(Player_Shooting_df_year,how= 'left',on='ID').merge(Player_Career_Info_Reduced_df,how = 'left',on='player_id')

    players_full_data = merged_dfs[(merged_dfs['team'] != "2TM") & (merged_dfs['team'] != "3TM") & (merged_dfs['team'] != "4TM")& (merged_dfs['team'] != "5TM")]

    return players_full_data


## Return filtered dataset on starting_5 from 1997

def player_starting_5_data(dfs, year):
    return player_full_data_df(dfs, year).query("starting_5 == 1")


# new y

# cette fonction prend pour entrée une année et renvoie un dataframe
# avec un pourcentage de la performance de l'équipe pour chaque joueur
# après l'année après celle entrée en parametre

# plus la valeur tend vers 1 plus l'équipe à été performance dans l'année
# (1 = premier de conférence + victoire aux playoffs)
def y_creator(year):
    #########################
    #### translate_dict #####
    #########################

    # creation of a translation dict : {'full team name' : 'abreviation'}
    translate= pd.read_csv('raw_data/Team Abbrev.csv')

    # drop unusefull columns
    translate = translate.drop(columns=['season','lg','playoffs']).set_index('team').to_dict()['abbreviation']


    #######################
    #### final_scores #####
    #######################

    final_scores = pd.read_csv('raw_data/Team_Playoffs_stats_raw.csv',sep = ';')

    # mise en forme du dataframe

    # la première ligne replace l'entête des colonnes (qui vide initialement), et suppression de la première ligne
    final_scores.columns = final_scores.iloc[0]
    final_scores.drop([0],inplace=True)

    # suppression des colonnes innutiles
    final_scores = final_scores.loc[:,['Yr', 'Series', 'Team']]
    final_scores = final_scores.iloc[:, :3] # suppression spécifiquement du deuxième label 'Team' (correspondant à l'équipe perdante du bracket)
    # keep only rows above 'year'
    final_scores['Yr'] = final_scores['Yr'].apply(lambda x : int(x)) # string ==> int
    final_scores =final_scores.query(f'Yr >= {year}')


    ## Extraction of data features into usable columns

    # Conf_ranking column creation, initaly kept last 3 caracters of Team column
    final_scores['Conf_ranking'] = final_scores['Team'].apply(lambda x : x[len(x)-2:len(x)-1])
    final_scores['Conf_ranking'] = final_scores['Conf_ranking'].apply(lambda x : 9 - int(x)) # scalling data : 1st --> 8 (point) ; 8 --> 1 (point)
    # clean Team column : del rank feature
    final_scores['Team'] = final_scores['Team'].apply(lambda x : x[:len(x)-4])
    # translate team name into abreviation
    final_scores['Team']=final_scores['Team'].apply(lambda team : translate[team.strip()])

    # Bracket point column
    dict_point = {'Eastern Conf First Round' : 1 , 'Western Conf First Round' : 1, 'Eastern Conf Semifinals' : 2, 'Western Conf Semifinals' : 2, 'Eastern Conf Finals' : 4 , 'Western Conf Finals': 4, 'Finals' : 8 }
    final_scores['bracket_points'] = final_scores['Series'].apply(lambda x : dict_point[x])
    # All_playoff_team_point column
    final_scores['All_playoff_team_point'] = final_scores.groupby(by = ['Yr','Team'])['bracket_points'].cumsum()
    # Global season score
    final_scores['global_season_score'] = final_scores.apply(lambda row : (int(row.Conf_ranking) + int(row.All_playoff_team_point))/23, axis =1)

    # keep only best bracket row for each team and each season
    final_scores = final_scores.groupby(by = ['Yr','Team']).agg(
        Conf_ranking=("Conf_ranking", "first"),
        bracket_points=("bracket_points", "max"),
        All_playoff_team_point=("All_playoff_team_point", "max"),
        global_season_score=("global_season_score", "max"),
    )
    # Primary key column creation
    final_scores = final_scores.reset_index() # rest multi-index
    final_scores["PM"] = final_scores["Yr"].astype(str) + final_scores["Team"] #use multi_index to create PM

    final_scores

    # #########################
    # ######## y_base #########
    # #########################

    Player_Season_Info_df = pd.read_csv('raw_data/Player Season Info.csv')

    # Creation of the base of y
    y_base = Player_Season_Info_df[['season','team']].query(f'season >= {year}').copy(deep = True)

    # PM (primary key) column creation for final_scores
    y_base['PM']=y_base.apply(lambda row : str(row['season']) + row['team'], axis = 1)

    # Merge final_scores & y-base
    y_base = y_base.merge(final_scores[['PM','global_season_score']], how = "left", on='PM')

    # replace Nan values of 'one' column (merge only add '1' to players that won the playoff this year, others have Nan values)
    y_base['global_season_score'] = y_base['global_season_score'].replace(np.nan, 0)

    # y_base.shape ==> 32606,1 | y.base.columns ==> 'one' | y_base.value_counts ==> 0.0 : 32213 ; 1.0 : 393
    y = y_base[['global_season_score']].copy(deep=True)

    return y


# Tests
if __name__ == "__main__":

    dfs = load_data()

    player_full_data = player_full_data_df(dfs, 1997)

    y = y_creator(1997)

    print("Test good (✅ pour Flavian)")
