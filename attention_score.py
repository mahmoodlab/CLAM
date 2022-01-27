"""
@author: Q Zeng
"""

from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import time
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

#%%
# Training settings
parser = argparse.ArgumentParser(description='CLAM Attention Score Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--data_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
					help='directory to save eval results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
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
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['tcga_hcc_349_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622',
                                                 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622',
                                                  
                                                 'mondor_hcc_139_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Gajewski_13G_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_139_6G_Interferon_Gamma_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Interferon_Gamma_Biology_cv_highvsrest_00X',
                                                 'mondor_hcc_139_T-cell_Exhaustion_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X'])
parser.add_argument('--feature_bags', type=str, nargs='+', default=None, 
                    help='names of patched feature files (ends with .pt) for visualization (default: [])')
args = parser.parse_args()

#%%
# Parameters for test
#parser = argparse.ArgumentParser(description='CLAM Attention Score Script')
#args = parser.parse_args()
#
#args.drop_out = True
#args.n_classes = 2
#args.model_size = 'small'
#args.model_type = 'clam_sb'
#
#ckpt_path = '../results/training_gene_signatures/tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622_CLAM_50_s1/s_0_checkpoint.pt'

#%%

encoding_size = 1024

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

#if args.bestk >= 0:
#    bestk = args.bestk
    
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
#ckpt_path = os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold))
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
#print("ckpt_path:")
#print(ckpt_path)

#args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code), "attention_scores_" + str(bestk))
args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))
#os.makedirs(args.save_dir, exist_ok=True)

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

print(settings)
if args.task == 'tcga_hcc_349_Inflammatory_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Inflammatory_highvsrest.csv',
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
    
elif args.task == 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_6G_Interferon_Gamma_highvsrest.csv',
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
    
elif args.task == 'mondor_hcc_139_Inflammatory_cv_highvsrest_00X':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Inflammatory_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'mondor_hcc_139_Gajewski_13G_Inflammatory_cv_highvsrest_00X':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Gajewski_13G_Inflammatory_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'mondor_hcc_139_6G_Interferon_Gamma_cv_highvsrest_00X':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_6G_Interferon_Gamma_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'mondor_hcc_139_Interferon_Gamma_Biology_cv_highvsrest_00X':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Interferon_Gamma_Biology_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    
elif args.task == 'mondor_hcc_139_T-cell_Exhaustion_cv_highvsrest_00X':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_T-cell_Exhaustion_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'mondor_hcc_139_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

else:
    raise NotImplementedError

if __name__ == "__main__":
     print("dataset:")
     print(dataset)
     print("data_dir:")
     print(args.data_dir)
     
     for ckpt_idx in range(len(ckpt_paths)):
         save_dir = args.save_dir
         save_dir = os.path.join(save_dir, "attention_scores_" + str(folds[ckpt_idx]))
         os.makedirs(save_dir, exist_ok=True)
         
         model = initiate_model(args, ckpt_paths[ckpt_idx])
         
         if args.feature_bags is not None:
             feature_bags = args.feature_bags
         else:
             feature_bags = sorted(os.listdir(args.data_dir))
             feature_bags = [features for features in feature_bags if features.endswith(".pt")]
         
         total = len(feature_bags)
         times = 0.
         
         for i in range(total): 
             print("\n\nprogress: {:.2f}, {}/{} in current model. {} out of {} models".format(i/total, i, total, ckpt_idx, len(ckpt_paths)))
             print('processing {}'.format(feature_bags[i]))
    
             bag_features = torch.load(os.path.join(args.data_dir, feature_bags[i]), map_location=lambda storage, 
                                       loc: storage.cuda(0))
             # torch.Size([22857, 1024])
             
             time_elapsed = -1
             start_time = time.time()
             logits, Y_prob, Y_hat, A, _ = model(bag_features) # A is the matrix of attention scores (n_classes x n_patches)
             time_elapsed = time.time() - start_time
             times += time_elapsed
             
             print("logits")
             print(logits)
             print("Y_prob")
             print(Y_prob)
             print("Y_hat")
             print(Y_hat)
             print("A")
             print(A.size()) # torch.Size([1, 22857])
             print("Max:")
             print(torch.max(A))
             print("Min:")
             print(torch.min(A))
             torch.save(A, os.path.join(save_dir, "attention_score_" + feature_bags[i]))
    
         times /= total
    
         print("average time in s per slide: {}".format(times))
