#%%

from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from models.resnet_custom import resnet50_baseline, resnet152_torchvision, densenet121_torchvision
import argparse
from utils.utils import print_network, collate_features
from PIL import Image
import h5py

#%%
# Modifed by Qinghe
# Move the arg parsing part to the top
# Add cpu arg
parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--csv_path', type=str)
parser.add_argument('--feat_dir', type=str)
parser.add_argument('--weight', type=str)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
###*********************************
#Modified by Qinghe 2020
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--trnsfrms', type=str, choices=['imagenet', None], default='imagenet', 
                    help='transforms applied to images, default imagenet standardization as preprocessing')
parser.add_argument('--cpu', default=False, action='store_true', help='force to use cpu') # if gpu not available, use cpu automatically
#Modified by Qinghe 01/05/2021
parser.add_argument('--target_patch_size', type=int, default=-1,
                    help='the desired size of patches for optional scaling before feature embedding')
###*********************************
#Modified by Qinghe 11/11/2021, for data augmentation
parser.add_argument('--train_augm', default=False, action='store_true', help='data augmentation')
args = parser.parse_args()    

#%%
# Parameters for test
# parser = argparse.ArgumentParser(description='Feature Extraction')
# args = parser.parse_args()

# args.data_dir = "./results/patches_mondor_tumor"
# args.csv_path = "./dataset_csv/mondor_hcc_feature_139.csv" 
# args.feat_dir = "./results/features_trained_custom/tcga_hcc_tumor_349_v1_cv_highvsrest_622_shufflenet-frz3_imagenet_s1"
# args.model = "shufflenet"
# args.weight = "./results/training_custom/tcga_hcc_tumor_349_v1_cv_highvsrest_622_shufflenet-frz3_imagenet_s1/s_6_checkpoint.pt"
# args.trnsfrms = 'imagenet'
# args.n_classes = 2
# args.batch_size = 256
# args.no_auto_skip = False 
# args.cpu = False 

#%%
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%%
def save_hdf5(output_dir, asset_dict, mode='a'):
    file = h5py.File(output_dir, mode)

    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val  

    file.close()
    return output_dir

#%%
###*********************************
#Modified by Qinghe 23/04/2021
#def compute_w_loader(file_path, output_path, model, batch_size = 8, verbose = 0, print_every=20, pretrained=True):
def compute_w_loader(file_path, output_path, model, batch_size = 8, verbose = 0, print_every=20, trnsfrms = 'imagenet', pretrained=True,
                     #Modified by Qinghe 11/11/2021, for data augmentation
                     #target_patch_size=-1):
                     target_patch_size=-1, train_augm=False):
###*********************************

    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet. Actually here means whether to use ImageNet transform or not
    """
    ###*********************************
    #Modified by Qinghe
    if trnsfrms == 'imagenet': # finally to preprocess (standardize) patches, with the imagenet mean and std of RGB
        pretrained = True
        custom_transforms=None
    else: # finally to preprocess (standardize) patches, with the mean and std of RGB both equal to (0.5, 0.5, 0.5).
        pretrained = False
        custom_transforms = trnsfrms # so far only None

    #dataset = Whole_Slide_Bag(file_path=file_path, pretrained=pretrained)
    #Modified by Qinghe 11/11/2021, for data augmentation
    #dataset = Whole_Slide_Bag(file_path=file_path, pretrained=pretrained, target_patch_size=target_patch_size, custom_transforms=custom_transforms)
    dataset = Whole_Slide_Bag(file_path=file_path, pretrained=pretrained, target_patch_size=target_patch_size, 
                           custom_transforms=custom_transforms, train_augm = train_augm)
    ###*********************************
    
    x, y = dataset[0]
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():    
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)
            mini_bs = coords.shape[0]
            
            features = model(batch)
            
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, mode=mode)
            mode = 'a'
    
    return output_path

#%%
if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    ###*********************************
    #Modified by Qinghe 01/05/2021
    if csv_path is None:
        raise NotImplementedError
    ###*********************************
    bags_dataset = Dataset_All_Bags(args.data_dir, csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    dest_files = os.listdir(args.feat_dir)

    print('loading model checkpoint')
    ###*********************************
    #Modified by Qinghe 2020
    # model = resnet50_baseline(pretrained=True)
    if args.model == "resnet152":
        model = resnet152_torchvision(pretrained=True)
    elif args.model == "densenet121":
        model = densenet121_torchvision(pretrained=True) 
    elif args.model == "resnet50":
        model = resnet50_baseline(pretrained=True) # default use by clam
    elif args.model == "shufflenet":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=False)
        # Reshape the output layer for 2 classes so that can match the weights
        model.fc = nn.Linear(1024, args.n_classes)
        checkpoint = torch.load("./results/training_custom/tcga_hcc_tumor_349_v1_cv_highvsrest_622_shufflenet-frz3_imagenet_s1/s_6_checkpoint.pt")
        model.load_state_dict(checkpoint)
        
        # replace the fc layer with a placeholder identity
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
                
            def forward(self, x):
                return x
        model.fc = Identity()
        
    else:
        raise NotImplementedError
    ###*********************************
    model = model.to(device)
    
    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        bag_candidate = bags_dataset[bag_candidate_idx]
        bag_name = os.path.basename(os.path.normpath(bag_candidate))

        if '.h5' in bag_candidate:

            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(bag_name)
            if not args.no_auto_skip and bag_name in dest_files:
                print('skipped {}'.format(bag_name))
                continue

            output_path = os.path.join(args.feat_dir, bag_name)
            file_path = bag_candidate
            time_start = time.time()
            output_file_path = compute_w_loader(file_path, output_path, 
            ###*********************************
            #Modified by Qinghe 2020
            #model = model, batch_size = args.batch_size, verbose = 1, print_every = 20)
            model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
            target_patch_size=args.target_patch_size, trnsfrms = args.trnsfrms,
            #Modified by Qinghe 11/11/2021, for data augmentation
            train_augm = args.train_augm)
            ###*********************************
            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
            file = h5py.File(output_file_path, "r")

            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'))


