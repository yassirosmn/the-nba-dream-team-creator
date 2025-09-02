import os
import numpy as np
from pathlib import Path


##################  ENV VARIABLES  ##################

# Saving path for database
DATABASE_PATH = "./database_folder/all_data/"

# Saving path for trained models
MODEL_PATH = "./database_folder/models/"

FILE_PATH=Path("raw_data/")

# If user defined FILE_PATH - use it, otherwise use by default raw_data
FILE_PATH=Path(os.environ.get("FILE_PATH", "raw_data/"))

# Error message if file_path invalid
assert FILE_PATH.exists(), "file_path does not exist"

# to do once done
## api
## streamlit cloud

##################  VARIABLES  ##################
DATA_SIZE = "TBD"
CHUNK_SIZE = "TBD"

##################  Colonnes à drop  ##################
COLUMNS_TO_DROP = ["lg",
"fg_percent",
"x3p_percent",
"x2p_percent",
"e_fg_percent",
"ft_percent",
"trb",
"trp_dbl",
"pg_percent",
"sg_percent",
"sf_percent",
"pf_percent",
"c_percent",
"on_court_plus_minus_per_100_poss",
"net_plus_minus_per_100_poss",
"points_generated_by_assists",
"and1",
"fga_blocked",
"percent_assisted_x2p_fg",
"percent_assisted_x3p_fg",
"corner_3_point_percent",
"num_heaves_attempted",
"num_heaves_made"
]

STATS_TO_KEEP = ['ft', 'orb', 'drb', 'ast', 'stl', 'blk', 'tov', 'pts']
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
