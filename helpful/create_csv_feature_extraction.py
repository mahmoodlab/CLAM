import os
import numpy as np

import pandas as pd
from pathlib import Path

data_dir = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/test_patches_ute/masks')
new_csv = pd.DataFrame(columns=['slide_id'])

for file in data_dir.iterdir():

    slide_id = str(file.stem)

    new_csv.loc[len(new_csv)] = [slide_id]

new_csv.to_csv('/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/test_patches_ute/process_list_wo_ending.csv', index=False)

