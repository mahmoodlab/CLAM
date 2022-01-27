#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 11:50:45 2021

For fast pipeline
Used with conda env 'deepai'

Evaluate shufflenet on patch level without annotations (non fast pipeline)
Can comment out the part of slide-level aggregation, so as to parallel each fold
Then use eval_customed_models_slide_aggregation.py
@author: Q Zeng
"""


#%% 
from __future__ import print_function

import argparse
import pdb
import os
import math

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
###*********************************
#Modified by Qinghe for the fast pipeline
#from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits, Dataset_from_Split
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits, Dataset_from_Split_FP
###*********************************
from utils.core_utils import train
from datasets.dataset_h5 import Whole_Slide_Bag

import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, precision_score
from sklearn.metrics import auc as calc_auc
import h5py
import time
# for the fast pipeline
import openslide

import matplotlib.pyplot as plt

from tqdm import tqdm
# for Spyder
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import statistics

#%%
# Generic validation settings
parser = argparse.ArgumentParser(description='Configurations for WSI Validation')
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
parser.add_argument('--trnsfrms', type=str, choices=['imagenet', None], default=None,  # None really did nothing!
                    help='transforms applied to images, default no preprocessing')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, first fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, last fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--split', type=str, choices=['val', 'test', 'all'], default='test')
parser.add_argument('--splits_dir', type=str, default=None, 
                    help='manually specify the set of  to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--model_type', type=str, choices=['shufflenet'], default='shufflenet', 
                    help='type of model (default: shufflenet)')
parser.add_argument('--task', type=str, choices=['tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622'])

###*********************************
#Modified by Qinghe for the fast pipeline
parser.add_argument('--data_slide_dir', type=str, default=None, help='path to slides (ndpi and svs accepted by default)')
parser.add_argument('--custom_downsample', type=int, default=1, help='overruled by target_patch_size')
parser.add_argument('--target_patch_size', type=int, default=-1, help='overrule custom_downsample')
###*********************************
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# Parameters for test
# parser = argparse.ArgumentParser(description='Configurations for WSI Training')
# args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# args.data_dir = './results/patches_tumor_masked'
# args.results_dir = "./results/training_custom_tumor_masked"
# args.eval_dir = "./eval_results_349_custom"
# args.save_exp_code = "tcga_hcc_tumor-masked_349_6G_Interferon_Gamma_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv_test"
# args.models_exp_code = "tcga_hcc_tumor-masked_349_6G_Interferon_Gamma_cv_highvsrest_622_shufflenet_frz3_imagenet_s1"
# args.split = 'test'
# args.fold = -1
# args.batch_size = 512
# args.seed = 1
# args.k = 10 
# args.k_start = -1
# args.k_end = 10 #10
# args.trnsfrms = 'imagenet'
# args.splits_dir = None
# args.task = "tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622" 
# args.model_type = 'shufflenet'
# args.data_slide_dir = './data/data_tcga_hcc'
# args.custom_downsample = 1
# args.target_patch_size = 256
# args.testing = False

#%% 
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#%% 
class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
        data_dir, 
        ###*********************************
        #Modified by Qinghe for the fast pipeline
        data_slide_dir=None,
        custom_downsample=1,
        target_patch_size=-1, # at actual magnification
        ###*********************************
        **kwargs):
    
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        ###*********************************
        #Modified by Qinghe for the fast pipeline
        self.data_slide_dir = data_slide_dir
        self.custom_downsample = custom_downsample
        self.target_patch_size = target_patch_size
        ###*********************************

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        
        full_path = os.path.join(self.data_dir,'{}.h5'.format(slide_id))
        with h5py.File(full_path,'r') as hdf5_file:
        ###*********************************
        #Modified by Qinghe for the fast pipeline
        #For the 4 variables: patch_level, patch_size, target_patch_size, custom_downsample
        #If in the original - dataset_h5.Whole_Slide_Bag_FP, there is 'self.' before, remove it, otherwise add it.
            #img = hdf5_file['imgs'][:]
            coords = hdf5_file['coords'][:]
            patch_level = f['coords'].attrs['patch_level']
            patch_size = f['coords'].attrs['patch_size']
            
        #unlike in __init__, here we shouldn't modify self.target_patch_size or self.custom_downsample, should keep the pass in values
        if patch_size == self.target_patch_size:
            target_patch_size = None
        elif self.target_patch_size > 0:
            target_patch_size = (self.target_patch_size, ) * 2
        elif self.custom_downsample > 1:
            target_patch_size = (patch_size // self.custom_downsample, ) * 2
        else:
            target_patch_size = None
                
        ###********* improve to remove the input --slide_ext
        #slide_file_path = os.path.join(args.data_slide_dir, bag_name.replace('.h5', args.slide_ext))
        # priority: NOT, AND, OR!!
        slide_file_path = os.path.join(self.data_slide_dir, [sli for sli in os.listdir(self.data_slide_dir) if (sli.endswith('.ndpi') or
                                                             sli.endswith('.svs')) and sli.startswith(slide_id)][0])
        ###*********
        with openslide.open_slide(slide_file_path) as wsi:
            for ind_coord in range(len(coords)):
                img = wsi.read_region(coords[ind_coord], patch_level, (patch_size, patch_size)).convert('RGB')
                if target_patch_size is not None:
                    img = img.resize(target_patch_size) # (256, 256, 3)
                    if ind_coord == 0:
                        imgs = torch.from_numpy(img).unsqueeze(0) # (1, 256, 256, 3)
                    else:
                        imgs = torch.cat((imgs, torch.from_numpy(img).unsqueeze(0)), 0) 

        #img = torch.from_numpy(img)
        #return img, label, coords
        return imgs, label, coords
        ###*********************************
		
#%% 
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hats, Ys):
        Y_hats = np.array(Y_hats.cpu()).astype(int)
        Ys = np.array(Ys.cpu()).astype(int)
        for label_class in np.unique(Ys):
            cls_masks = Ys == label_class # if belongs to this class
            self.data[label_class]["count"] += cls_masks.sum()
            self.data[label_class]["correct"] += (Y_hats[cls_masks] == Ys[cls_masks]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

#%% 
def collate_MIL(batch):
    imgs = torch.cat([item[0] for item in batch], dim = 0)
     # imgs = torch.stack([item[0] for item in batch], dim = 0)
    labels = torch.LongTensor([item[1] for item in batch])
    IDs = [item[3] for item in batch]
    return [imgs, labels, IDs]

###*********************************
#Modified by Qinghe for the fast pipeline
#def get_dataset_loader(split_dataset, training = False, weighted = False, batch_size = 64, trnsfrms = None):
def get_dataset_loader(split_dataset, testing = False, batch_size = 64, trnsfrms = None, data_slide_dir=None, custom_downsample=1, 
					   target_patch_size=-1):
###*********************************
    """
        return the test loader
    """
	# kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == "cuda" else {}
	# 'pin_memory': False is faster for my GTX 1080 in testing mode, ncalls=1482,time=1091->ncalls=222,time=936.4
    kwargs = {'num_workers': 2, 'pin_memory': False} if device.type == "cuda" else {}
    
    atts = {att : getattr(split_dataset,att) for att in ["slide_data", "data_dir", "num_classes"]}
    
    tile_IDs = []
    labels = []
    for i in range(len(split_dataset.slide_data['slide_id'])):
        with h5py.File(os.path.join(split_dataset.data_dir,split_dataset.slide_data['slide_id'][i])+'.h5','r') as hdf5_file:
            tile_IDs.extend([split_dataset.slide_data['slide_id'][i] +":" + str(s) for s in range(hdf5_file['coords'].shape[0])]) # all tiles from each slides
            labels.extend([split_dataset.slide_data['label'][i]] * hdf5_file['coords'].shape[0]) # split_dataset.slide_data['label']: 1,1,1,...,1,0,0,0,...,0
    
	# convert split_dataset into patch level
    ###*********************************
    #Modified by Qinghe for the fast pipeline
    #split_dataset = Dataset_from_Split(tile_IDs, labels, trnsfrms, **atts)
    split_dataset = Dataset_from_Split_FP(tile_IDs, labels, trnsfrms, data_slide_dir, custom_downsample, target_patch_size, **atts)
    ###*********************************
		
    if not testing: # here testing mean only use a very small amount of data
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
    
    else:
        ###*********************************
        ids = np.concatenate((np.arange(int(len(split_dataset)*0.001)), np.arange(len(split_dataset)-1-int(len(split_dataset)*0.001),len(split_dataset)-1))) #Cluster High and Cluster Median/Low
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs)
        
    return loader

#%% 
def initiate_model(args, ckpt_path):
    print('\nInit Model...', end=' ') 
        
    if args.model_type == 'shufflenet':
        # Load ShuffleNet pretrained on ImageNet
#        model = torch.hub.load('./models/pytorch_vision_v0.6.0', 'shufflenet_v2_x1_0', source='local') # jean zay could not download to cache, so first copy shufflenet to ./models
        model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True) # force_reload=True for the first time 
        # print(model.cuda())
        # summary(model.cuda(), (3, 256, 256)) # similar as Keras, 167 layers
        
        # Reshape the output layer for 2 classes
        model.fc = nn.Linear(1024, args.n_classes)
        # print(model.cuda())
        
#        # count layers
#        count = 0
#        for child in model.children():
#            count+=1
#        print(count) # 7 blocks
       
        # summary(model.cuda(), (3, 256, 256)) #  Non-trainable params: 696

    else:
        raise NotImplementedError
    
    # relocate
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     device_ids = list(range(torch.cuda.device_count()))
    #     model = nn.DataParallel(model, device_ids=device_ids).to('cuda:0')
    # else:
    #     model.to(device)


    
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt, strict=True)

    model.to(device)
    model.eval()
	 
    print('Done!')
    #print_network(model)
	
    return model

#%%%
def evaluate(dataset, args, cur):
    """   
        evaluate a single fold
    """

    print('\nInit {} split...'.format(args.split), end=' ')
    dataset.load_from_h5(True)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    ###*********************************
    #Modified by Qinghe for the fast pipeline
    loader = get_dataset_loader(dataset, testing = args.testing, batch_size = args.batch_size, trnsfrms = args.trnsfrms, 
								data_slide_dir = args.data_slide_dir, custom_downsample = args.custom_downsample, 
								target_patch_size = args.target_patch_size)
    ###*********************************
    print('Done!')

    
    results_dict, test_error, test_auc, acc_logger = summary(model, loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    return results_dict, test_auc, 1-test_error

#%% 
def summary(model, loader, n_classes): # should have tile level and slide level
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = []
    all_labels = []
    
    patient_results = {}

    for batch_idx, (data, labels, IDs) in tqdm(enumerate(loader)):
        # data = data.permute(0, 3, 1, 2).float()
        # data, labels = data.to(device), labels.to(device)
        data, labels = data.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(data) # n x 2
            Y_hats = torch.max(outputs, 1)[1]
            y_probs = F.softmax(outputs, dim = 1)

        acc_logger.log_batch(Y_hats, labels)
        all_probs.extend(y_probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        patient_results.update({batch_idx: {'batch_size': data.size(0), 'tile_ids': IDs, 'probs': all_probs, 'labels': np.asarray(all_labels)}})
        error = calculate_error(Y_hats, labels)
        test_error += error * data.size(0)

    test_error /= len(all_labels)
    
    all_probs = np.stack(all_probs, axis=0)
    all_labels = np.asarray(all_labels)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    # else:
        # aucs = []
        # binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        # for class_idx in range(n_classes):
        #     if class_idx in all_labels:
        #         fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        #         aucs.append(calc_auc(fpr, tpr))
        #     else:
        #         aucs.append(float('nan'))

        # auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
   
   
#%% 
# depends on function roc_curve()
def Find_Optimal_Cutoff(target, probs):
    fpr, tpr, thresholds = roc_curve(target, probs)
    
    listy = tpr - fpr
    optimal_idx = np.argwhere(listy == np.amax(listy))
    optimal_idx = optimal_idx.flatten().tolist()
    optimal_thresholds = [thresholds[optimal_id] for optimal_id in optimal_idx]
    print("optimal_threshold(s): " +str(optimal_thresholds))
    if len(optimal_thresholds) > 1: # If multiple optimal cutoffs, the one closer to the median was chosen.
#         raise Exception("Multiple optimal cutoffs!")
        optimal_thresholds = [min(optimal_thresholds, key=lambda x:abs(x - statistics.median(thresholds)))]
        print("Multiple optimal cutoffs! {} is chosen.".format(str(optimal_thresholds)))

    return optimal_thresholds

#%% 
# #############################################################################
# ROC analysis
def draw_mean_roc(labels_for_roc, probs_for_roc, k = 10):
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # print(X[test].shape, y[test].shape)

    fig, ax = plt.subplots()
#     counter = 0
    for i in range(k):
        
        fpr, tpr, thresholds = roc_curve(labels_for_roc[i], probs_for_roc[i])
#         print(counter,counter+nslides[i])
        ax.plot(fpr, tpr, alpha=0.3, lw=1)
    #     ax.plot(viz.fpr, viz.tpr, label='ROC fold {}'.format(i), alpha=0.3, lw=1)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        counter_auc = calc_auc(fpr, tpr)
        aucs.append(counter_auc) # not real auc (plot), but the interpolated auc for mean calculation

#         counter = counter + nslides[i]

    print('\nAUCs: {}'.format(aucs))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            alpha=.8)
    #         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = calc_auc(mean_fpr, mean_tpr) #####
    std_auc = np.std(aucs) #####
    # can set marker to check the prob distribution, here just disable for simple mean curves
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=str(k) + "-fold Receiver Operating Characteristic")
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.xlabel('1 - Specificity (False Positive Rate)')
    ax.legend(loc="lower right")
    # ax.legend(loc="best")
    return aucs, mean_auc, std_auc, np.max(aucs), np.argwhere(aucs == np.max(aucs))[0]


#%% 
if __name__ == "__main__":
    
    seed_torch(args.seed)
    
    args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))
    args.results_dir = os.path.join(args.results_dir, args.model_type)
    args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.splits_dir is None:
        args.splits_dir = args.models_dir
    
    assert os.path.isdir(args.models_dir)
    assert os.path.isdir(args.splits_dir)
    
    settings = {'task': args.task,
                'split': args.split,
                'data_dir':args.data_dir,
                'save_dir': args.save_dir, 
                'models_dir': args.models_dir,
                'model_type': args.model_type,
                'trnsfrms': args.trnsfrms,
                ###*********************************
                #Modified by Qinghe or the fast pipeline
                'data_slide_dir': args.data_slide_dir,
                'target_patch_size': args.target_patch_size,
                'custom_downsample': args.custom_downsample,
                ###*********************************
				'testing': args.testing}
    
    
    with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
        print(settings, file=f)
    f.close()
    
    print(settings)

    if args.task == 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_6G_Interferon_Gamma_highvsrest.csv',
                                data_dir= args.data_dir,
                                ###*********************************
                                #Modified by Qinghe for the fast pipeline
                                data_slide_dir = args.data_slide_dir,
                                target_patch_size=args.target_patch_size,
                                custom_downsample=args.custom_downsample,
                                ###*********************************
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                                label_col = 'cluster',
                                patient_strat= True,
                                ignore=[])
        
    elif args.task == 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Gajewski_13G_Inflammatory_highvsrest.csv',
                                data_dir= args.data_dir,
                                ###*********************************
                                #Modified by Qinghe for the fast pipeline
                                data_slide_dir = args.data_slide_dir,
                                target_patch_size=args.target_patch_size,
                                custom_downsample=args.custom_downsample,
                                ###*********************************
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                                label_col = 'cluster',
                                patient_strat= True,
                                ignore=[])
        
    elif args.task == 'tcga_hcc_349_Inflammatory_cv_highvsrest_622':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Inflammatory_highvsrest.csv',
                                data_dir= args.data_dir,
                                ###*********************************
                                #Modified by Qinghe for the fast pipeline
                                data_slide_dir = args.data_slide_dir,
                                target_patch_size=args.target_patch_size,
                                custom_downsample=args.custom_downsample,
                                ###*********************************
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                                label_col = 'cluster',
                                patient_strat= True,
                                ignore=[])
        
    elif args.task == 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Interferon_Gamma_Biology_highvsrest.csv',
                                data_dir= args.data_dir,
                                ###*********************************
                                #Modified by Qinghe for the fast pipeline
                                data_slide_dir = args.data_slide_dir,
                                target_patch_size=args.target_patch_size,
                                custom_downsample=args.custom_downsample,
                                ###*********************************
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                                label_col = 'cluster',
                                patient_strat= True,
                                ignore=[])
        
    elif args.task == 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                                data_dir= args.data_dir,
                                ###*********************************
                                #Modified by Qinghe for the fast pipeline
                                data_slide_dir = args.data_slide_dir,
                                target_patch_size=args.target_patch_size,
                                custom_downsample=args.custom_downsample,
                                ###*********************************
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                                label_col = 'cluster',
                                patient_strat= True,
                                ignore=[])
        
    elif args.task == 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_T-cell_Exhaustion_highvsrest.csv',
                                data_dir= args.data_dir,
                                ###*********************************
                                #Modified by Qinghe for the fast pipeline
                                data_slide_dir = args.data_slide_dir,
                                target_patch_size=args.target_patch_size,
                                custom_downsample=args.custom_downsample,
                                ###*********************************
                                shuffle = False, 
                                seed = args.seed, 
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
    datasets_id = {'val': 1, 'test': 2, 'all': -1}
    

    all_results = []
    all_auc = []
    all_acc = []
	
    labels_for_roc = []
    probs_for_roc = []
    opt_thresholds = []

    fig = plt.figure(figsize=(15,15))

    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
            
        model = initiate_model(args, ckpt_paths[ckpt_idx])
        patient_results, test_error, auc = evaluate(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
		# save patch-level prediction as csv (for vis) and also pkl *******
        df = pd.DataFrame.from_dict(patient_results).T
        
        # 3 advantages: 0- fast, 1- much smaller size, 2- could be more accurate %.9g
        since = time.time()
        save_pkl(os.path.join(args.save_dir, 'split_{}_results.pkl'.format(folds[ckpt_idx])), patient_results)
        time_elapsed = time.time() - since
        print('Saving to pkl in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
#        # super long to save as csv (str) and can only have %.8g
#        since = time.time()
#        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=True)
#        time_elapsed = time.time() - since
#        print('Saving to csv in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

		# *****************************************************************
    
#		# slide-level aggregation # ***************************************
#        tile_ids = []
#        for i in range(len(patient_results.keys())):
#	        tile_ids.extend(patient_results[i]['tile_ids'])
#	        
#        probs = []
#        labels = patient_results[len(patient_results.keys())-1]['labels']
#        for j in range(len(patient_results[len(patient_results.keys())-1]['probs'])):
#	        probs.append(patient_results[len(patient_results.keys())-1]['probs'][j][1])
#	        
#	    # plot histogram for patch-level probabilities 
#        fig.add_subplot(4, 3, ckpt_idx+1)
#        plt.hist(np.array(probs), density=False, bins=10)  # density=False would make counts
#        plt.ylabel('Probability')
#        plt.xlabel('Data');
#        plt.xlim((0, 1))
#	        
#        threshold = Find_Optimal_Cutoff(labels, probs)[0]
#        print('Patch-level threshold of fold {}: {}'.format(ckpt_idx, threshold))
#        opt_thresholds.append(threshold)
#	    
#	    
#        slide_id = tile_ids[0].split(":")[0]
#	
#        n_patches = 0
#        n_positives = 0
#        slide_probs = {}
#        slide_labels = {}
#	
#        for i in range(len(tile_ids)):
#	        if slide_id != tile_ids[i].split(":")[0]:
#	            slide_probs[slide_id] = n_positives / n_patches
#	            slide_labels[slide_id] = labels[i-1]
#	            slide_id = tile_ids[i].split(":")[0]
#	            n_patches = 0
#	            n_positives = 0
#	        if probs[i] >= threshold:
#	            n_positives = n_positives + 1
#	        n_patches = n_patches + 1
#	
#        labels_for_roc.append(list(slide_labels.values()))
#        probs_for_roc.append(list(slide_probs.values()))
#        
#        df_slide = pd.DataFrame({'slide_id': list(slide_labels.keys()), 'Y': list(slide_labels.values()), 'p_1': list(slide_probs.values())})
#        df_slide.to_csv(os.path.join(args.save_dir, 'slide_aggregation_fold_{}.csv'.format(folds[ckpt_idx])))
#		
#    fig.savefig(os.path.join(args.save_dir, "prob_distribution.png"))
#	
#    aucs, mean_auc, std_auc, max_auc, max_idx = draw_mean_roc(k = args.k, labels_for_roc = labels_for_roc, probs_for_roc = probs_for_roc)
#    plt.savefig(os.path.join(args.save_dir, "roc.png"))
#    plt.show()
#    print("Mean AUC: {:.3f}".format(mean_auc))
#    print("AUC sd: {:.3f}".format(std_auc))
#    print("Max AUC: {:.3f}".format(max_auc))
#    print("Max index: {}".format(max_idx))
#    auc_summary = pd.DataFrame([aucs]).T
#    auc_summary.loc['Mean AUC'] = mean_auc
#    auc_summary.loc['AUC sd'] = std_auc
#    auc_summary.loc['Max AUC'] = max_auc
#    auc_summary.loc['Max index'] = max_idx
        
#    auc_summary.index.names = ['folds']
#    auc_summary.rename(columns={0:'test_auc'}, inplace=True)
#    auc_summary.to_csv(os.path.join(args.save_dir, 'slide_aggregation_summary.csv'))
#    pd.DataFrame(opt_thresholds).to_csv(os.path.join(args.save_dir,, 'cutoffs.csv'), header=False)
#		# *****************************************************************
#
#
#    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
#    if len(folds) != args.k:
#        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
#    else:
#        save_name = 'summary.csv'
#    final_df.to_csv(os.path.join(args.save_dir, save_name))
	
	
    




