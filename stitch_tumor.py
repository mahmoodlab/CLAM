#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:57:36 2020

@author: Q Zeng
"""

from PIL import Image
import os
import time
import numpy as np
import h5py
from scipy import stats
from wsi_core.WholeSlideImage import DrawMap, StitchPatches
from tqdm import tqdm

path_tumor = "./results/patches_tumor_masked"
path_stitch = "./results/stitches_tumor_masked"

os.makedirs(path_stitch, exist_ok=True)

for root, dirs, files in os.walk(path_tumor):
    patch_bags = files
    
for i in tqdm(range(len(files))):
    
    print("\nProcessing " + patch_bags[i])
    
    if os.path.isfile(os.path.join(path_stitch, patch_bags[i].replace(".h5", ".png"))):
            print("Already existed. Source folder is " + path_tumor.split("/")[-1] + ".")
    else:
        with h5py.File(os.path.join(path_tumor, patch_bags[i]), mode='r') as f:
            dset = f["imgs"]
            
            downscale=64
            
            draw_grid=False
            bg_color=(0,0,0)
            alpha=-1
    
            if 'downsampled_level_dim' in f["imgs"].attrs.keys():
                w, h = f["imgs"].attrs['downsampled_level_dim']
            else:
                w, h = f["imgs"].attrs['level_dim']
            print('original size: {} x {}'.format(w, h)) # the patching level size
    
            w = w // downscale
            h = h //downscale # download 64 of the patching level
            # f["coords"] = (f["coords"] / downscale).astype(np.int32)
            print('downscaled size for stiching: {} x {}'.format(w, h))
            print('number of patches: {}'.format(len(f["imgs"])))
            img_shape = f["imgs"][0].shape # 22857
            print('patch shape: {}'.format(img_shape))
    
            downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)
            
            if w*h > Image.MAX_IMAGE_PIXELS: 
                raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
            
            if alpha < 0 or alpha == -1:
                heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
            else:
                heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
            
            heatmap = np.array(heatmap)
            
            heatmap = DrawMap(heatmap, dset, (f["coords"][:] / downscale).astype(np.int32), downscaled_shape, indices=None, draw_grid=draw_grid)
        
        heatmap.save(os.path.join(path_stitch, patch_bags[i].replace(".h5", ".png")))
        
    #    with h5py.File(os.path.join(path_seg, patch_bags[i]), mode='r') as f:
    #        snapshot = StitchPatches(f, downscale=64, bg_color=(255,255,255), alpha=-1, draw_grid=False)
    #    
    #    Image.blend(snapshot, heatmap, 0.5).show()
