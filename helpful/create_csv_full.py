import os
import numpy as np

import pandas as pd
from pathlib import Path

biomarker_file = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/TMA_cores/csv/original/merged_dataset_with_biomarkers_281023.csv')
data_dir = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/TMA_cores/cores/level_0_magnification_40.0/HUNT0')
new_csv = pd.DataFrame(columns=['case_id', 'slide_id', 'Ki67', 'ER','HER2','PR','Filepath'])

biomarker_df = pd.read_csv(biomarker_file)
no_information = []


for file in data_dir.iterdir():
    # case_id (patient), slide_id, label for the biomarkers, completepath
    case_id = int(file.stem.split('_')[0])
    slide_id = file.stem

    try:
        biomarker_info = biomarker_df[biomarker_df['ID']== case_id]
        if biomarker_info['Ki67'].iloc[0] == '>= 15%':
            ki67 = '0'
        elif biomarker_info['Ki67'].iloc[0] == '< 15%':
            ki67 = '1'
        else:
            ki67 = None

        if biomarker_info['ER'].iloc[0] == '>= 1%':
            er = '0'
        elif biomarker_info['ER'].iloc[0] == '< 1%':
            er = '1'
        else:
            er = None

        if biomarker_info['HER2'].iloc[0] == 'Negative':
            her2 = '0'
        elif biomarker_info['HER2'].iloc[0] == 'Positive':
            her2 = '1'
        else:
            her2 = None

        if biomarker_info['PR'].iloc[0] == '>= 1%':
            pr = '0'
        elif biomarker_info['PR'].iloc[0] == '< 1%':
            pr = '1'
        else:
            pr = None


        new_csv.loc[len(new_csv)] = [case_id, slide_id,biomarker_info['Ki67'].iloc[0],biomarker_info['ER'].iloc[0],biomarker_info['HER2'].iloc[0],biomarker_info['PR'].iloc[0], file]
    except:
        no_information.append(case_id)

    a=5

new_csv.to_csv('/mnt/EncryptedDisk2/BreastData/Studies/CLAM/csv/CLAM_HUNT0_level_0_magnification_40.0.csv', index=False)
print(sorted(set(no_information)))