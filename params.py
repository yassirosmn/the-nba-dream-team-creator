import os
import numpy as np
from pathlib import Path


##################  ENV VARIABLES  ##################


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
