import synapseclient, os
from pathlib import Path
import pandas as pd

import synapseutils

biomarker_file = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/csv/CLAM_HUNT0_level_0_magnification_40.0.csv')

biomarker_df = pd.read_csv(biomarker_file)

feature_file = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/csv/process_list_wo_ending.csv')
feature_df = pd.read_csv(feature_file)

feature_df['case_id'] = feature_df['slide_id'].str.split('_').str[0]

biomarker_df['case_id'] = biomarker_df['case_id'].astype(str)
feature_df['case_id'] = feature_df['case_id'].astype(str)

# Merge the DataFrames on the ID column
result_df = biomarker_df.merge(feature_df, on='slide_id', how='inner')

result_df = result_df.drop(columns=['case_id_y'])
result_df = result_df.rename(columns={'case_id_x': 'case_id'})



result_df.to_csv('/mnt/EncryptedDisk2/BreastData/Studies/CLAM/csv_for_training.csv', index=False)
