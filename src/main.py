from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from data_handlers.codealltag_data_handler import CodealltagDataHandler
from utils.project_utils import ProjectUtils

import json


if __name__ == "__main__":
    project_root = ProjectUtils.get_project_root()
    data_handler = CodealltagDataHandler(project_root=project_root)
    email_files_info_10k_dataframe = data_handler.get_email_files_info_10k_dataframe()

    ### Print a random row from the dataframe as json
    random_row = email_files_info_10k_dataframe.sample(n=1).iloc[0]
    print(json.dumps(random_row.to_dict(), indent=4, ensure_ascii=False))