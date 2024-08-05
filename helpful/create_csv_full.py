import os
import numpy as np

import pandas as pd
from pathlib import Path

biomarker_file = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/TMA_cores/csv/original/merged_dataset_with_biomarkers_281023.csv')
data_dir = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/test_patches_ute/features/h5_files')
new_csv = pd.DataFrame(columns=['case_id', 'slide_id', 'Ki67', 'ER','HER2','PR','Filepath'])

biomarker_df = pd.read_csv(biomarker_file)
no_information = []


for file in data_dir.iterdir():
    # case_id (patient), slide_id, label for the biomarkers, completepath
    case_id = file.stem
    slide_id = file.stem

    try:
        biomarker_info = biomarker_df[biomarker_df['ID']== int(case_id)]

        new_csv.loc[len(new_csv)] = [case_id, slide_id,biomarker_info['Ki67'].iloc[0],biomarker_info['ER'].iloc[0],biomarker_info['HER2'].iloc[0],biomarker_info['PR'].iloc[0], file]
    except:
        no_information.append(case_id)

    a=5

new_csv.to_csv('/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/test_patches_ute/train_csv.csv', index=False)
print(sorted(set(no_information)))