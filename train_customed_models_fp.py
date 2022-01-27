#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 00:21:11 2021

For fast pipeline
Used with conda env 'deepai'

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
#Modified by Qinghe 15/05/2021, for the fast pipeline
#from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits, Dataset_from_Split
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits, Dataset_from_Split_FP
###*********************************
from utils.core_utils import train
from datasets.dataset_h5 import Whole_Slide_Bag

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import h5py
import time
# 15/05/2021, for the fast pipelien
import openslide

#%%
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='root of data directory')
parser.add_argument('--data_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--trnsfrms', type=str, choices=['imagenet', None], default=None,  # None really did nothing!
                    help='transforms applied to images, default no preprocessing')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--patience', type=int, default=20,
                    help='how long to wait after last time validation loss improved (default: 20)')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--min_epochs', type=int, default=50,
                    help='minimum number of epochs to stop training (default: 50)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, first fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, last fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--splits_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
# parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                      help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['shufflenet'], default='shufflenet', 
                    help='type of model (default: shufflenet)')
parser.add_argument('--freeze', type=int, default=0, 
                    help='freeze the first n blocks (/layers) of the model during training')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
# unlike CLAM which weighting on slide level, here is the undersampling on patch level dataset (after 500 from each slide - a must)
parser.add_argument('--train_weighting', action='store_true', default=False, 
                    help='enable undersampling (Cluster Low) to balance classes during training')
# disable to make the distribution (class imbalance on slide-level) more similar to the test set
parser.add_argument('--val_weighting', action='store_true', default=False, 
                    help='enable undersampling (Cluster Low) to balance classes during validation')
# doesn't matter since never used in real test, but obviously more reasonable to enable it (val and test split should be processed in the same way)
# undersampling may be the worst option, should try weighting (like CLAM to keep the length) or focal loss
parser.add_argument('--test_weighting', action='store_true', default=False, 
                    help='enable undersampling (Cluster Low) to balance classes during test')
parser.add_argument('--task', type=str, choices=['tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622'])
###*********************************
#Modified by Qinghe 15/05/2021, for the fast pipeline
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

# args.early_stopping = True
# args.patience = 5 #20
# args.max_epochs = 30 #50
# args.min_epoch = 10
# args.lr = 5e-5
# args.reg = 1e-5
# args.opt = 'adam'
# args.batch_size = 32
# args.seed = 1
# args.k = 10 
# args.k_start = -1
# args.k_end = 2 #10
# args.label_frac = 1 
# args.data_dir = "./results/patches_tumor"
# args.trnsfrms = None
# args.results_dir = "./results/training_custom" 
# args.splits_dir = None
# args.train_weighting = True
# args.val_weighting = True
# args.test_weighting = True
# args.bag_loss = 'ce' 
# args.task = "tcga_hcc_349_v1_cv_highvsrest_622" 
# args.model_type = 'shufflenet'
# args.freeze = 3
# args.log_data = True
# args.testing = True
# args.exp_code = "tcga_hcc_tumor_349_v1_cv_highvsrest_622_shufflenet-test" 

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
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

#%% 
def collate_MIL(batch):
    imgs = torch.cat([item[0] for item in batch], dim = 0)
    # imgs = torch.stack([item[0] for item in batch], dim = 0)
    labels = torch.LongTensor([item[1] for item in batch])
    IDs = [item[3] for item in batch]
    return [imgs, labels, IDs]

###*********************************
#Modified by Qinghe 15/05/2021, for the fast pipeline
#def get_dataset_loader(split_dataset, training = False, testing = False, weighted = False, batch_size = 64, trnsfrms = None):
def get_dataset_loader(split_dataset, training = False, testing = False, weighted = False, batch_size = 64, trnsfrms = None,
                       data_slide_dir=None, custom_downsample=1, target_patch_size=-1):
###*********************************
    """
        return either the validation loader or training loader 
    """
    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == "cuda" else {}
    
    atts = {att : getattr(split_dataset,att) for att in ["slide_data", "data_dir", "num_classes"]}
    
    tile_IDs = []
    labels = []
    for i in range(len(split_dataset.slide_data['slide_id'])):
        with h5py.File(os.path.join(split_dataset.data_dir,split_dataset.slide_data['slide_id'][i])+'.h5','r') as hdf5_file:
            tile_IDs.extend([split_dataset.slide_data['slide_id'][i] +":" + str(s) 
                       ###*********************************
                       #Modified by Qinghe 15/05/2021, for the fast pipeline
                       #for s in np.random.randint(0, hdf5_file['imgs'].shape[0], 500)]) # collect up 500 patches each slide
                       for s in np.random.randint(0, hdf5_file['coords'].shape[0], 500)]) # collect up 500 patches each slide
                       ###*********************************
        labels.extend([split_dataset.slide_data['label'][i]]*500) # split_dataset.slide_data['label']: 1,1,1,...,1,0,0,0,...,0
        
    if not testing: # here testing mean only use a very small amount of data
        if weighted: # undersampling on patch level
            idx = np.random.randint(low=labels.count(1), high=len(labels), size=labels.count(1)) # High is label 1
            idx = np.concatenate((range(labels.count(1)), idx)) # all the idx for High and the selected Low idx
            tile_IDs = np.asarray(tile_IDs)[idx].tolist()
            labels = np.asarray(labels)[idx].tolist()
        ###*********************************
        #Modified by Qinghe 15/05/2021, for the fast pipeline
        #split_dataset = Dataset_from_Split(tile_IDs, labels, trnsfrms, **atts)
        split_dataset = Dataset_from_Split_FP(tile_IDs, labels, trnsfrms, data_slide_dir, custom_downsample, target_patch_size, **atts)
        ###*********************************
        
        if training:
            # without replacement, then sample from a shuffled dataset
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
    else:
        ###*********************************
        #Modified by Qinghe 15/05/2021, for the fast pipeline
        #split_dataset = Dataset_from_Split_FP(tile_IDs, labels, trnsfrms, **atts)
        split_dataset = Dataset_from_Split_FP(tile_IDs, labels, trnsfrms, data_slide_dir, custom_downsample, target_patch_size, **atts)
        ###*********************************
        ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.001), replace = False)
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs)
        
    return loader

#%%%
def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    for dset_split in [train_split, val_split, test_split]:
        dset_split.load_from_h5(True)
        # dset_split.data_dir = dset_split.data_dir.replace('features', 'patches')
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    
    print('\nInit Loaders...', end=' ')
    ###*********************************
    #Modified by Qinghe 15/05/2021, for the fast pipeline
    train_loader = get_dataset_loader(train_split, training=True, testing = args.testing, weighted = args.train_weighting, 
                                      #batch_size = args.batch_size, trnsfrms = args.trnsfrms)
                                      batch_size = args.batch_size, trnsfrms = args.trnsfrms, data_slide_dir = args.data_slide_dir, 
                                      custom_downsample = args.custom_downsample, target_patch_size = args.target_patch_size)
    val_loader = get_dataset_loader(val_split, testing = args.testing, weighted = args.val_weighting, 
                                    #batch_size = args.batch_size, trnsfrms = args.trnsfrms)
                                    batch_size = args.batch_size, trnsfrms = args.trnsfrms, data_slide_dir = args.data_slide_dir, 
                                    custom_downsample = args.custom_downsample, target_patch_size = args.target_patch_size)
    test_loader = get_dataset_loader(test_split, testing = args.testing, weighted = args.test_weighting, 
                                     #batch_size = args.batch_size, trnsfrms = args.trnsfrms)
                                     batch_size = args.batch_size, trnsfrms = args.trnsfrms, data_slide_dir = args.data_slide_dir, 
                                     custom_downsample = args.custom_downsample, target_patch_size = args.target_patch_size)
    ###*********************************
    print('Done!')

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ') 
        
    if args.model_type == 'shufflenet':
        # Load ShuffleNet pretrained on ImageNet
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
        
        count = 0
        for child in model.children():
          if count < args.freeze:
            for param in child.parameters():
                param.requires_grad = False 
          count+=1 # 7 for shufflenet
       
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
    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch = args.min_epochs, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
        early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    # For training, the test will also be done. The summary.csv will be the same as the running the eval.py, except there are also eval results!
    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 

#%% 
def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    size_dataset = 0
    print('\n')
    for batch_idx, (data, labels, patch_id) in enumerate(loader):
    
        # From: [batch_size, height, width, channels]
        # To: [batch_size, channels, height, width]
        # print(data.shape)
        # data = data.permute(0, 3, 1, 2).float() # cast Input type (torch.cuda.ByteTensor) to match weight type (torch.cuda.FloatTensor)

        # data = torch.narrow(data, 1, torch.randint(0, data.shape[0] - 500 + 1), (500,)) # random sample 500 patches: random start with consecutive fixed n
        # data = torch.index_select(data, 0, torch.as_tensor(torch.randint(0, data.shape[0], (500,))))
        # print(data.shape)
        # print()
        
        data, labels = data.to(device), labels.to(device)

        outputs = model(data) # n x 2

        Y_hats = torch.max(outputs, 1)[1]
        y_probs = F.softmax(outputs, dim = 1)
        
        acc_logger.log_batch(Y_hats, labels)
        loss = loss_fn(outputs, labels)
        
        train_loss += loss.item() * data.size(0)
        print('batch {}, loss: {:.4f}, batch_size: {}'.format(batch_idx, loss.item(), data.size(0))) # should use tqdm with these info
           
        error = calculate_error(Y_hats, labels)
        train_error += error * data.size(0)
        
        size_dataset += data.size(0)
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= size_dataset
    train_error /= size_dataset

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

#%% 
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, labels, patch_id) in enumerate(loader):
            # data = data.permute(0, 3, 1, 2).float()
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(data) # n x 2

            Y_hats = torch.max(outputs, 1)[1]
            y_probs = F.softmax(outputs, dim = 1)

            acc_logger.log_batch(Y_hats, labels)
            
            loss = loss_fn(outputs, labels)

            all_probs.extend(y_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            val_loss += loss.item() * data.size(0)
            error = calculate_error(Y_hats, labels)
            val_error += error * data.size(0)
            

    val_error /= len(all_labels)
    val_loss /= len(all_labels) # the monitor is the mean validation loss
    
    all_probs = np.stack(all_probs, axis=0)
    all_labels = np.asarray(all_labels)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

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

    for batch_idx, (data, labels, IDs) in enumerate(loader):
        # data = data.permute(0, 3, 1, 2).float()
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
        #Modified by Qinghe 15/05/2021, for the fast pipeline
        data_slide_dir=None,
        custom_downsample=1,
        target_patch_size=-1, # at actual magnification
        ###*********************************
        **kwargs):
    
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        ###*********************************
        #Modified by Qinghe 15/05/2021, for the fast pipeline
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
        #Modified by Qinghe 15/05/2021, for the fast pipeline
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
                
        ###********* 15/05/2021, improve to remove the input --slide_ext
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
def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.splits_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)
        # *********************************************************
#         # Here is the way to load the pickled file using pandas:
#         unpickled_dict = pd.read_pickle(filename)
#         unpickled_df[0].keys() # o is the batch id
#         # output: dict_keys(['batch_size', 'tile_ids', 'probs', 'labels'])
        # *********************************************************

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

#%% 
if __name__ == "__main__":
    
    since = time.time()
    
    seed_torch(args.seed)
    
    # add model type to the results_dir
    args.results_dir = os.path.join(args.results_dir, args.model_type)
    
     # encoding_size = 1024
    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'trnsfrms': args.trnsfrms,
                'early_stopping': args.early_stopping,
                'patience': args.patience,
                'max_epochs': args.max_epochs,
                'min_epochs': args.min_epochs,
                'batch_size': args.batch_size,
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'model_type': args.model_type,
                'freeze': args.freeze,
                'train_weighting': args.train_weighting,
                'val_weighting': args.val_weighting,
                'test_weighting': args.test_weighting,
                'opt': args.opt,
                ###*********************************
                #Modified by Qinghe 15/05/2021, for the fast pipeline
                #'testing': args.testing
                'testing': args.testing,
                'data_dir':args.data_dir,
                'data_slide_dir': args.data_slide_dir,
                'target_patch_size': args.target_patch_size,
                'custom_downsample': args.custom_downsample
                ###*********************************
                }
    
    print('\nLoad Dataset')
    
    if args.task == 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_6G_Interferon_Gamma_highvsrest.csv',
                                data_dir= args.data_dir,
                                ###*********************************
                                #Modified by Qinghe 15/05/2021, for the fast pipeline
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
                                #Modified by Qinghe 15/05/2021, for the fast pipeline
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
                                #Modified by Qinghe 15/05/2021, for the fast pipeline
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
                                #Modified by Qinghe 15/05/2021, for the fast pipeline
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
                                #Modified by Qinghe 15/05/2021, for the fast pipeline
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
                                #Modified by Qinghe 15/05/2021, for the fast pipeline
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
        
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    
    if args.splits_dir is None:
        args.splits_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        args.splits_dir = os.path.join('splits', args.splits_dir)
    assert os.path.isdir(args.splits_dir)
    
    settings.update({'splits_dir': args.splits_dir})
    
    
    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))   
    
    results = main(args)
    print("finished!")
    print("end script")
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



