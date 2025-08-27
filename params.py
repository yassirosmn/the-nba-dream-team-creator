import os
import numpy as np
from pathlib import Path


##################  ENV VARIABLES  ##################


FILE_PATH=Path("raw_data/")

# Error message if file_path invalid
assert FILE_PATH.exists(), "file_path does not exist"

# If user defined FILE_PATH - use it, otherwise use by default raw_data
FILE_PATH=Path(os.environ.get("FILE_PATH", "raw_data/"))

# to do once done
## api
## streamlit cloud

##################  VARIABLES  ##################
DATA_SIZE = "TBD"
CHUNK_SIZE = "TBD"
