import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np
import torch.distributed as dist

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():     
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'
    
    return output_path

def main(args, rank, world_size):
    global device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        print('initializing dataset')
    
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    if rank == 0:
        os.makedirs(args.feat_dir, exist_ok=True)
        os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    
    dist.barrier() 

    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
            
    _ = model.eval()
    model = model.to(device)
    total = len(bags_dataset)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for bag_candidate_idx in tqdm(range(total), disable=(rank != 0)):
        
        if bag_candidate_idx % world_size != rank:
            continue
        
        slide_id = bags_dataset[bag_candidate_idx]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
        
        if rank == 0 and bag_candidate_idx % 20 == 0:
             print(f'\nRank 0 progress: {bag_candidate_idx}/{total}')

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            continue 

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        
        try:
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                        wsi=wsi, 
                                        img_transforms=img_transforms)

            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
            output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 0)
        
        except Exception as e:
            print(f"Error processing {slide_id} on rank {rank}: {e}")
            continue

        time_elapsed = time.time() - time_start
        
        try:
            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
        
        except Exception as e:
            print(f"Error saving features for {slide_id} on rank {rank}: {e}")
            continue
if __name__ == '__main__':
    
    if "LOCAL_RANK" not in os.environ:
        print("Error: Please use 'torchrun' to launch this script.")
        exit(1)
        
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, default='/data/wsi/BRACS-process/patch-results')
    parser.add_argument('--data_slide_dir', type=str, default='/data/wsi/BRACS-process/wsi-soft-link')
    parser.add_argument('--slide_ext', type=str, default= '.svs')
    parser.add_argument('--csv_path', type=str, default='/data/wsi/BRACS-process/labels/all.csv')
    
    parser.add_argument('--feat_dir', type=str, default='/data/wsi/BRACS-process/uni-features')
    parser.add_argument('--model_name', type=str, default='uni_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=224)
    args = parser.parse_args()

    main(args, rank, world_size)
    
    dist.destroy_process_group()