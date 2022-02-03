import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from PIL import Image
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_hdf5(output_path, asset_dict, mode='a'):
    file = h5py.File(output_path, mode)

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
    return output_path


def compute_w_loader(file_path, output_path, wsi, model,
     batch_size = 8, verbose = 0, print_every=20, pretrained=True,
     ###*********************************
     ###Modified by Qinghe 26/10/2021, add color unmixing
#    custom_downsample=1, target_patch_size=-1):
     custom_downsample=1, target_patch_size=-1,
     unmixing = False, separate_stains_method = None, file_bgr = None, bgr = None, delete_third_stain = False, 
     convert_to_rgb = False,
     ###Modified by Qinghe 29/10/2021, add color normalization
     color_norm = False, color_norm_method = None,
     ###*********************************
     ###Modified by Qinghe 02/11/2021, add saving (normalized) patches to h5 file
     save_images_to_h5 = False, image_h5_dir = None):
    ###*********************************
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet. Actually here means whether to use ImageNet transform or not
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding (overruled by target_patch_size)
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
        ###*********************************
        ###Modified by Qinghe 26/10/2021, add color unmixing
        #custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        custom_downsample=custom_downsample, target_patch_size=target_patch_size,
        unmixing = unmixing, separate_stains_method = separate_stains_method, file_bgr = file_bgr, bgr = bgr, 
        delete_third_stain = delete_third_stain, convert_to_rgb = convert_to_rgb,
        ###Modified by Qinghe 29/10/2021, add color normalization
        color_norm = color_norm, color_norm_method = color_norm_method,
        ###*********************************
        ###Modified by Qinghe 02/11/2021, add saving (normalized) patches to h5 file
        save_images_to_h5 = save_images_to_h5, image_h5_dir = image_h5_dir)
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


parser = argparse.ArgumentParser(description='Feature Extraction')
###*********************************
#Modified by Qinghe 01/05/2021, make it compatible with the old version of the code
# parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_dir', type=str, help='path to h5 files')
###*********************************
parser.add_argument('--data_slide_dir', type=str, default=None, help='path to slides (ndpi and svs accepted by default)')
###********* 15/05/2021, improve to remove the input --slide_ext	
#parser.add_argument('--slide_ext', type=str, default= '.svs')	
###********* 02/02/2021, reuse to enable custom ext	
parser.add_argument('--slide_ext', nargs="+", default= ['.svs', '.ndpi', '.tiff'], help='slide extensions to be recognized, svs/ndpi/tiff by default')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
###*********************************
#Modified by Qinghe 01/05/2021
#parser.add_argument('--custom_downsample', type=int, default=1)
#parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--custom_downsample', type=int, default=1, help='overruled by target_patch_size')
parser.add_argument('--target_patch_size', type=int, default=-1, help='overrule custom_downsample')
###*********************************
###*********************************
#Modified by Qinghe 16/05/2021, for trained shufflenet
parser.add_argument('--model', type=str, default="resnet50", help='feautre extractor')
parser.add_argument('--weights_path', type=str, default=None)
parser.add_argument('--n_classes', type=int, default=2)
###*********************************
###*********************************
###Modified by Qinghe 26/10/2021, add color unmixing
parser.add_argument('--unmixing', default=False, action='store_true', help='apply color unmixing')
parser.add_argument('--separate_stains_method', default=None, help='method to separate stains')
parser.add_argument('--file_bgr', type=str, default=None, help='overruled by bgr')
parser.add_argument('--bgr', type=str, default=None, help='overrule file_bgr')
parser.add_argument('--delete_third_stain', default=False, action='store_true', help='replace the third stain with slide background')
parser.add_argument('--convert_to_rgb', default=False, action='store_true', help='use rgb instead of stain image')
###*********************************
###*********************************
###Modified by Qinghe 29/10/2021, add color normalization
parser.add_argument('--color_norm', default=False, action='store_true', help='apply color normalization')
parser.add_argument('--color_norm_method', default=None, help='method for color normalization')
###*********************************
###Modified by Qinghe 02/11/2021, add saving (normalized) patches to h5 file
parser.add_argument('--save_images_to_h5', default=False, action='store_true', help='save patch image to h5 in a new path')
parser.add_argument('--image_h5_dir', default=None, help='path to save new h5 with patch images added')
###*********************************
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    ###*********************************
    #Modified by Qinghe 01/05/2021, make it compatible with the old version of the code
    #bags_dataset = Dataset_All_Bags(csv_path)
    bags_dataset = Dataset_All_Bags(args.data_dir, csv_path)
    ###*********************************
    
    os.makedirs(args.feat_dir, exist_ok=True)
    ###*********************************
    #Modified by Qinghe 01/05/2021, make it compatible with the old version of the code
    #os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    #os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    #dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    dest_files = os.listdir(args.feat_dir)
    ###*********************************
    

    print('loading model checkpoint')
    ###*********************************
    #Modified by Qinghe 16/05/2021, for trained shufflenet
#    # model = resnet50_baseline(pretrained=True)
#    if args.model == "resnet152":
#        model = resnet152_torchvision(pretrained=True)
#    elif args.model == "densenet121":
#        model = densenet121_torchvision(pretrained=True) 
#    elif args.model == "resnet50":
    if args.model == "resnet50":
        model = resnet50_baseline(pretrained=True) # default use by clam
    elif args.model == "shufflenet":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=False)
        # Reshape the output layer for 2 classes so that can match the weights
        model.fc = nn.Linear(1024, args.n_classes)
        checkpoint = torch.load(args.weights_path)
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
        ###*********************************
        #Modified by Qinghe 01/05/2021, make it compatible with the old version of the code
        #slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        #bag_name = slide_id+'.h5'
        #h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        bag_candidate = bags_dataset[bag_candidate_idx]    # full path + full name (ext/.h5 included)
        bag_name = os.path.basename(os.path.normpath(bag_candidate)) # full name (ext/.h5 included)
        h5_file_path = bag_candidate
        
        #slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
        ###********* 15/05/2021, improve to remove the input --slide_ext
        #slide_file_path = os.path.join(args.data_slide_dir, bag_name.replace('.h5', args.slide_ext))
        # priority: NOT, AND, OR!!
        slide_file_path = os.path.join(args.data_slide_dir, [sli for sli in os.listdir(args.data_slide_dir) if sli.endswith(tuple(args.slide_ext)) and 
                           sli.startswith(os.path.splitext(bag_name)[0])][0])
        ###*********
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(bag_name)

        #if not args.no_auto_skip and slide_id+'.pt' in dest_files:
        #    print('skipped {}'.format(slide_id))
        if not args.no_auto_skip and bag_name in dest_files:    
            print('skipped {}'.format(bag_name))
            continue 

        #output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        output_path = os.path.join(args.feat_dir, bag_name)
        time_start = time.time()
        with openslide.open_slide(slide_file_path) as wsi:
            output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
                                                model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
                                                ###*********************************
                                                ###Modified by Qinghe 26/10/2021, add color unmixing
                                                #custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
                                                custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
                                                unmixing = args.unmixing, separate_stains_method = args.separate_stains_method,
                                                file_bgr = args.file_bgr, bgr = args.bgr, delete_third_stain = args.delete_third_stain, 
                                                convert_to_rgb = args.convert_to_rgb,
                                                ###Modified by Qinghe 29/10/2021, add color normalization
                                                color_norm = args.color_norm, color_norm_method = args.color_norm_method,
                                                ###*********************************
                                                ###Modified by Qinghe 02/11/2021, add saving (normalized) patches to h5 file
                                                save_images_to_h5 = args.save_images_to_h5, image_h5_dir = args.image_h5_dir)
                                                ###*********************************

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        #torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
        ##********
        # 19/05/2021 for training clam with trained shufflenet
        # torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'))
        # The 1.6 release of PyTorch switched torch.save to use a new zipfile-based file format.
        try: # in deepai conda env (torch.__version__='1.7.0') when using shufflenet, otherwise training the clam, will throw an error complaining that it is a zip file.
            torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'),_use_new_zipfile_serialization=False)
        except TypeError: # in clam conda env (torch version '1.3.1')
            torch.save(features, os.path.join(args.feat_dir, bag_base+'.pt'))
        ##********
        ###*********************************


