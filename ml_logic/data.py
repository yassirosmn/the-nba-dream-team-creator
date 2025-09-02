import pandas as pd
import numpy as np
from params import *
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


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


def new_y_creator(year):

    # Chemin d'accès des fichiers de données
    path = FILE_PATH

    #########################
    #### translate_dict #####
    #########################

    # Chargement du fichier des abréviations d'équipes
    translate = pd.read_csv(f'{path}/Team Abbrev.csv')

    # Suppression des colonnes inutiles et création d'un dictionnaire {'nom complet': 'abréviation'}
    translate = translate.drop(columns=['season','lg','playoffs']).set_index('team').to_dict()['abbreviation']

    #######################
    #### conf_winrate #####
    #######################

    # Chargement du fichier contenant les résumés des équipes
    path = FILE_PATH
    conf_winrate = pd.read_csv(f'{path}/Team Summaries.csv')

    # Sélection des colonnes utiles (saison, équipe, victoires, défaites)
    conf_winrate = conf_winrate.loc[: , ['season', 'abbreviation', 'w', 'l',]]

    # Filtrage des saisons supérieures ou égales à 1997
    conf_winrate.query("season >= 1997", inplace=True)

    # Suppression des lignes contenant des valeurs manquantes
    conf_winrate = conf_winrate.dropna()

    # Transformation du nombre de victoires en pourcentage de victoires
    conf_winrate['w'] = conf_winrate.apply(lambda row : row['w']/(row['w'] + row['l']), axis = 1)

    # Création d'une clé primaire PM = saison + abréviation
    conf_winrate['PM'] = conf_winrate.apply(lambda row : str(row['season']) + '_' + str(row['abbreviation']), axis = 1)

    # Suppression des colonnes inutiles
    conf_winrate.drop(columns=['abbreviation','season','l'], inplace=True)

    # Conversion en dictionnaire {PM: taux de victoire}
    conf_winrate_dict = conf_winrate.set_index('PM').to_dict()['w']

    #######################
    #### final_scores #####
    #######################

    # Chargement du fichier brut des statistiques des playoffs
    final_scores = pd.read_csv('raw_data/Team_Playoffs_stats_raw.csv', sep = ';')

    # La première ligne contient les en-têtes de colonnes → remplacement et suppression de la première ligne
    final_scores.columns = final_scores.iloc[0]
    final_scores.drop([0], inplace=True)

    # Sélection des colonnes utiles (année, série, équipe)
    final_scores = final_scores.loc[:,['Yr', 'Series', 'Team']]

    # Suppression spécifique du doublon de colonne "Team"
    final_scores = final_scores.iloc[:, :3]

    # Conversion de la colonne année en entier
    final_scores['Yr'] = final_scores['Yr'].apply(lambda x : int(x))

    # Filtrage des saisons supérieures ou égales à l'année passée en paramètre
    final_scores = final_scores.query(f'Yr >= {year}')

    ## Extraction des caractéristiques utilisables

    # Nettoyage de la colonne Team : suppression du rang affiché en suffixe
    final_scores['Team'] = final_scores['Team'].apply(lambda x : x[:len(x)-4])

    # Traduction du nom d'équipe complet en abréviation
    final_scores['Team'] = final_scores['Team'].apply(lambda team : translate[team.strip()])

    # Attribution de points par tour de playoffs
    dict_point = {'Eastern Conf First Round' : 1 , 'Western Conf First Round' : 1,
                  'Eastern Conf Semifinals' : 2, 'Western Conf Semifinals' : 2,
                  'Eastern Conf Finals' : 4 , 'Western Conf Finals': 4,
                  'Finals' : 8 }
    final_scores['bracket_points'] = final_scores['Series'].apply(lambda x : dict_point[x])

    # Calcul cumulé des points de playoffs par équipe et saison
    final_scores['All_playoff_team_point'] = final_scores.groupby(by = ['Yr','Team'])['bracket_points'].cumsum()

    # Conservation uniquement du meilleur résultat par équipe et par saison
    final_scores = final_scores.groupby(by = ['Yr','Team']).agg(
        bracket_points=("bracket_points", "max"),
        All_playoff_team_point=("All_playoff_team_point", "max")
    )

    # Réinitialisation de l'index multi-niveaux
    final_scores = final_scores.reset_index()

    # Création de la clé primaire PM = année + abréviation
    final_scores["PM"] = final_scores["Yr"].astype(str) + '_' + final_scores["Team"]

    # Normalisation des points de playoffs sur une base de 15
    final_scores['All_playoff_team_point'] = final_scores['All_playoff_team_point'].apply(lambda val : val/15)

    # Conversion en dictionnaire {PM: score de playoffs}
    final_scores_dict = final_scores.set_index('PM').drop(columns=['Yr','bracket_points','Team']).to_dict()['All_playoff_team_point']

    #########################
    ######## y_base #########
    #########################

    # Chargement des informations des joueurs par saison
    Player_Season_Info_df = pd.read_csv('raw_data/Player Play By Play.csv')

    # Création de la base de données y
    y_base = Player_Season_Info_df[['season','team']].query(f'season >= {year}').copy(deep = True)

    # Suppression des équipes artificielles (joueurs ayant joué pour plusieurs équipes)
    y_base = y_base[y_base['team']!= '2TM']
    y_base = y_base[y_base['team']!= '3TM']
    y_base = y_base[y_base['team']!= '4TM']
    y_base = y_base[y_base['team']!= '5TM']

    # Création de la clé primaire PM = saison + abréviation
    y_base['PM'] = y_base.apply(lambda row : str(row['season']) + '_' + row['team'], axis = 1)

    # Ajout du taux de victoire par équipe
    y_base['winrate'] = y_base['PM'].apply(lambda PM : conf_winrate_dict[PM])

    # Ajout du score de playoffs par équipe (0 si absente)
    y_base['Playoff_score'] = y_base['PM'].apply(lambda PM : final_scores_dict[PM] if PM in final_scores_dict.keys() else 0.0)

    # Calcul d'un score global (moyenne entre taux de victoire et score de playoffs)
    y_base['global_score'] = y_base.apply(lambda row : (row.winrate + row.Playoff_score) / 2, axis = 1)

    # split y entre 1997 2024 et 2025
    y_base_2025 = y_base.query('season == 2025')
    y_base_1997_2024 = y_base.query('season < 2025')

    # Conservation uniquement de la colonne score global 2025
    y_2025 = y_base_2025[['global_score','PM']]
    y_2025.drop_duplicates(inplace=True)
    y_2025.reset_index(drop=True, inplace=True)
    y_winrate_2025 = y_base_2025[['PM','winrate']]
    y_winrate_2025.drop_duplicates(inplace=True)
    y_winrate_2025.reset_index(drop=True, inplace=True)

    # Conservation uniquement de la colonne score global 1997-2024
    y_1997_2024 = y_base_1997_2024[['global_score','PM']]
    y_1997_2024.drop_duplicates(inplace=True)
    y_1997_2024.reset_index(drop=True, inplace=True)
    y_winrate_1997_2024 = y_base_1997_2024[['PM','winrate']]
    y_winrate_1997_2024.drop_duplicates(inplace=True)
    y_winrate_1997_2024.reset_index(drop=True, inplace=True)
    # Renvoi du DataFrame final y
    return y_winrate_1997_2024, y_1997_2024, y_winrate_2025,y_2025

# Tests
if __name__ == "__main__":

    y_winrate_1997_2024, y_1997_2024, y_winrate_2025,y_2025 = new_y_creator(1997)

    print("Test good (✅ pour Flavian)")
