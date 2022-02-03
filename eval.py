from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *


#%%

# Training settings
## If train with different hyperparameters, or different loss functions, (e.g., --B, --no_inst_cluster)
## it does not affect how the model is evaluated. Tested with different B values here, no changes in validation results.
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='root of data directory')
parser.add_argument('--data_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
                    help='the directory to save eval results relative to project root (default: ./eval_results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, for external validation (default: None, to use the same splits as training)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
### macro is preferred over micro as the former gives equal importance to each class 
### whereas the later gives equal importance to each sample
### the class High is rare, but it's way important, 'macro' should be a better choice because it treats each class equally.
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC') 
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622'])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
## Parameters for test
#parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
#args = parser.parse_args()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#args.drop_out = True
#args.k = 10 
#args.k_start = -1
#args.k_end = 2 #10
#args.fold = -1
#args.label_frac = 1 
#args.data_dir = "./results/features_mondor_tumor_detection_masked"
#args.results_dir = "./results/training_gene_signatures_tumor_masked"
#args.eval_dir = "./eval_results_349_tumor_detection_masked"
#args.splits_dir = "./splits/mondor_hcc_139_6G_Interferon_Gamma_cv_highvsrest_00X_100"
#args.split = 'test'
#args.models_exp_code = "tcga_hcc_tumor-masked_349_6G_Interferon_Gamma_cv_highvsrest_622_CLAM_50_s1"
#args.save_exp_code = "mondor_hcc_tumor-detect-masked_139_6G_Interferon_Gamma_cv_highvsrest_00X_CLAM_50_s1_cv"
#args.task = "mondor_hcc_139_6G_Interferon_Gamma_cv_highvsrest_00X"
#args.model_type = 'clam_sb'
#args.model_size = 'small'

#%%

encoding_size = 1024

args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}


with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_6G_Interferon_Gamma_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Gajewski_13G_Inflammatory_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'tcga_hcc_349_Inflammatory_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Inflammatory_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Interferon_Gamma_Biology_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_T-cell_Exhaustion_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))