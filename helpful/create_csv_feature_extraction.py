import os
import numpy as np

import pandas as pd
from pathlib import Path

data_dir = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patches')
new_csv = pd.DataFrame(columns=['slide_id'])

for file in data_dir.iterdir():

    slide_id = file.stem

    new_csv.loc[len(new_csv)] = [slide_id]

new_csv.to_csv('/mnt/EncryptedDisk2/BreastData/Studies/CLAM/process_list_wo_ending.csv', index=False)

