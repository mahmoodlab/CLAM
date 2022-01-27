#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 23:54:41 2021

@author: Q Zeng
"""

import argparse
import pandas as pd
import os
import h5py
import openslide
import torch
from torchvision import transforms
import numpy as np
import time

#%%
parser = argparse.ArgumentParser(description='Data augmentation')
parser.add_argument('--data_dir', type=str, help='path to h5 files')
parser.add_argument('--result_dir', type=str, help='path to augmented h5 files')
parser.add_argument('--data_slide_dir', type=str, default=None, help='path to slides (ndpi and svs accepted by default)')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--target_patch_size', type=int, default=-1, help='obligatory parameter')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
args = parser.parse_args()

#data_dir = 'results/patches_tumor_masked'
#result_dir = './results/patches-augm-8-flips-rots_tumor_masked'
#data_slide_dir = './data/data_tcga_hcc'
#csv_path = 'dataset_csv/tcga_hcc_feature_354.csv'
#target_patch_size = 256

#%% Define flip and rotation functions, modified from https://github.com/msahasrabudhe/lymphoMIL
#   Tensors here are defined as B x C x H x W

def AugTransform_flipX(T):
    # Flip the images in tensor T along X.
    return torch.flip(T, (3,))

def AugTransform_flipY(T):
    # Flip the images in tensor T along Y.
    return torch.flip(T, (2,))

def AugTransform_transpose(T):
    # Transpose the image. 
    return T.permute(0,1,3,2)

def AugTransform_rot90(T):
    # Rotate the image 90 degrees clockwise. 
    return AugTransform_transpose(AugTransform_flipX(T))

def AugTransform_rot180(T):
    # Rotate the image 180 degrees. 
    return AugTransform_flipX(AugTransform_flipY(T))

def AugTransform_rot270(T):
    # Rotate the image 270 degrees. 
    return AugTransform_transpose(AugTransform_flipY(T))

#%%
def write_img(image, num):
    if augm_file['coords'].attrs['patch_size'] != args.target_patch_size:
        image = image.resize((args.target_patch_size, ) * 2)        
    augm_file['imgs'][idx_coord*8+num] = np.asarray(image)

#%%
if __name__ == '__main__':
    os.makedirs(args.result_dir, exist_ok=True)
    dest_files = os.listdir(args.result_dir)
    
    df = pd.read_csv(args.csv_path)
    print("{} slides to process".format(len(df)))
    total_time = 0.
    nslides = 0
    
    for idx in range(len(df)):
        slide_id = df['slide_id'][idx]
        
        if not args.no_auto_skip and slide_id+'.h5' in dest_files:    
            print('skipped {}'.format(slide_id+'.h5'))
            continue 
        
        print('Processing {}...'.format(slide_id+'.h5'))
        nslides = nslides + 1
        start_time = time.time()
        
        with h5py.File(os.path.join(args.result_dir, slide_id + '.h5'), 'w') as augm_file:
            with h5py.File(os.path.join(args.data_dir, slide_id + '.h5'), 'r') as file:
                # the dataset 'coords' is the same as the original h5 file. That to say 1 coord for all augmented copies
                augm_file.create_dataset('coords', data=file['coords'][()])
                for key, value in dict(file['coords'].attrs.items()).items():
                    augm_file['coords'].attrs[key] = value
            
            # create a resizable dataset (None means unlimited)
            dset = augm_file.create_dataset('imgs', shape=(augm_file['coords'].shape[0]*8, args.target_patch_size, args.target_patch_size, 3), 
                                       maxshape=(None, args.target_patch_size, args.target_patch_size, 3), 
                                       chunks=(1, args.target_patch_size, args.target_patch_size, 3), dtype='uint8')
            
            slide_file_path = os.path.join(args.data_slide_dir, [sli for sli in os.listdir(args.data_slide_dir) if (sli.endswith('.ndpi') or sli.endswith('.svs')) and 
                           sli.startswith(slide_id)][0])
            with openslide.open_slide(slide_file_path) as wsi:
                for idx_coord in range(len(augm_file['coords'])):
                    coord = augm_file['coords'][idx_coord]
                    img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#                    print(np.asarray(img).shape) # (512, 512, 3)
                    
                    pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#                    print(pil_to_tensor.shape) # torch.Size([1, 3, 512, 512])
                    
                    # 8 versions for each patch, to achieve rotational invariance of the models
                    write_img(img, 0)
                    
                    im = transforms.ToPILImage()(AugTransform_flipX(pil_to_tensor).squeeze_(0)).convert("RGB")
                    write_img(im, 1)
#                    print(np.asarray(im).shape) # (256, 256, 3)
                    
                    im = transforms.ToPILImage()(AugTransform_flipY(pil_to_tensor).squeeze_(0)).convert("RGB")
                    write_img(im, 2)
#                    print(np.asarray(im).shape) # (256, 256, 3)
                    
                    im = transforms.ToPILImage()(AugTransform_rot90(pil_to_tensor).squeeze_(0)).convert("RGB")
                    write_img(im, 3)
#                    print(np.asarray(im).shape) # (256, 256, 3)

                    im = transforms.ToPILImage()(AugTransform_rot180(pil_to_tensor).squeeze_(0)).convert("RGB")
                    write_img(im, 4)
#                    print(np.asarray(im).shape) # (256, 256, 3)

                    im = transforms.ToPILImage()(AugTransform_rot270(pil_to_tensor).squeeze_(0)).convert("RGB")
                    write_img(im, 5)
#                    print(np.asarray(im).shape) # (256, 256, 3)

                    im = transforms.ToPILImage()(AugTransform_flipX(AugTransform_rot90(pil_to_tensor)).squeeze_(0)).convert("RGB")
                    write_img(im, 6)
#                    print(np.asarray(im).shape) # (256, 256, 3)

                    im = transforms.ToPILImage()(AugTransform_flipY(AugTransform_rot90(pil_to_tensor)).squeeze_(0)).convert("RGB")
                    write_img(im, 7)
#                    print(np.asarray(im).shape) # (256, 256, 3)
                    
        processing_time = time.time() - start_time
        print("Processing time: {} s.".format(processing_time))
        total_time = total_time + processing_time
        print('Done.')
        
    print("Average processing time: {} s.".format(total_time/nslides))


#%% Parallel processing seems to be much longer... no matter either putting the subprocess inside the coords iteration or not
#import multiprocessing as mp
#from multiprocessing import Process
#print("Max worker: {}".format(mp.cpu_count()))
#    
#def work1():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        write_img(img, 0)
#       # display(img)
#        #print(np.asarray(img).shape) # (256, 256, 3)
#    
#def work2():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        
#        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#        im = transforms.ToPILImage()(AugTransform_flipX(pil_to_tensor).squeeze_(0)).convert("RGB")
#        write_img(im, 1)
#        #display(im)
#       # print(np.asarray(im).shape) # (256, 256, 3)
#    
#def work3():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        
#        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#        im = transforms.ToPILImage()(AugTransform_flipY(pil_to_tensor).squeeze_(0)).convert("RGB")
#        write_img(im, 2)
#        #display(im)
#        #print(np.asarray(im).shape) # (256, 256, 3)
#    
#def work4():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        
#        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#        im = transforms.ToPILImage()(AugTransform_rot90(pil_to_tensor).squeeze_(0)).convert("RGB")
#        write_img(im, 3)
#        #display(im)
#        #print(np.asarray(im).shape) # (256, 256, 3)
#
#def work5():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        
#        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#        im = transforms.ToPILImage()(AugTransform_rot180(pil_to_tensor).squeeze_(0)).convert("RGB")
#        write_img(im, 4)
#        #display(im)
#        #print(np.asarray(im).shape) # (256, 256, 3)
#
#def work6():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        
#        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#        im = transforms.ToPILImage()(AugTransform_rot270(pil_to_tensor).squeeze_(0)).convert("RGB")
#        write_img(im, 5)
#        #display(im)
#        #print(np.asarray(im).shape) # (256, 256, 3)   
#    
#def work7():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        
#        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#        im = transforms.ToPILImage()(AugTransform_flipX(AugTransform_rot90(pil_to_tensor)).squeeze_(0)).convert("RGB")
#        write_img(im, 6)
#        #display(im)
#        #print(np.asarray(im).shape) # (256, 256, 3)
#        
#def work8():
#    """thread worker function"""
#    for idx_coord in range(len(augm_file['coords'])):
#        coord = augm_file['coords'][idx_coord]
#        img = wsi.read_region(coord, augm_file['coords'].attrs['patch_level'], (augm_file['coords'].attrs['patch_size'], augm_file['coords'].attrs['patch_size'])).convert('RGB')
#    #                    print(np.asarray(img).shape) # (512, 512, 3)
#        
#        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
#        im = transforms.ToPILImage()(AugTransform_flipY(AugTransform_rot90(pil_to_tensor)).squeeze_(0)).convert("RGB")
#        write_img(im, 7)
#        #display(im)
#        #print(np.asarray(im).shape) # (256, 256, 3)
#
#
#print('Processing {}...'.format(slide_id+'.h5'))
#start_time = time.time()
#
#with h5py.File(os.path.join(args.result_dir, slide_id + '.h5'), 'w') as augm_file:
#    with h5py.File(os.path.join(args.data_dir, slide_id + '.h5'), 'r') as file:
#        augm_file.create_dataset('coords', data=file['coords'][()])
#        for key, value in dict(file['coords'].attrs.items()).items():
#            augm_file['coords'].attrs[key] = value
#    
#    # create a resizable dataset (None means unlimited)
#    dset = augm_file.create_dataset('imgs', shape=(augm_file['coords'].shape[0]*8, args.target_patch_size, args.target_patch_size, 3), 
#                               maxshape=(None, args.target_patch_size, args.target_patch_size, 3), 
#                               chunks=(1, args.target_patch_size, args.target_patch_size, 3), dtype='uint8')
#    
#    slide_file_path = os.path.join(args.data_slide_dir, [sli for sli in os.listdir(args.data_slide_dir) if (sli.endswith('.ndpi') or sli.endswith('.svs')) and 
#                   sli.startswith(slide_id)][0])
#    with openslide.open_slide(slide_file_path) as wsi:
#        p1 = Process(target = work1)
#        p1.start()
#        p2 = Process(target = work2)
#        p2.start()
#        p3 = Process(target = work3)
#        p3.start()
#        p4 = Process(target = work4)
#        p4.start()
#        p5 = Process(target = work5)
#        p5.start()
#        p6 = Process(target = work6)
#        p6.start()
#        p7 = Process(target = work7)
#        p7.start()
#        p8 = Process(target = work8)
#        p8.start()
#        # This is where I had to add the join() function.
#        p1.join()
#        p2.join()
#        p3.join()
#        p4.join()
#        p5.join()
#        p6.join()
#        p7.join()
#        p8.join()
#        print(time.time() - start_time)
#        
#with h5py.File(os.path.join(args.result_dir, slide_id + '.h5'), 'r') as augm_file:
#    with h5py.File(os.path.join(args.result_dir[:-1], slide_id + '.h5'), 'r') as file:
#        print((augm_file['coords'][()]==file['coords'][()]).all())
#        print((augm_file['imgs'][()]==file['imgs'][()]).all())
