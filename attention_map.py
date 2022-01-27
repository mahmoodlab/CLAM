# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Q Zeng
"""

import argparse
from PIL import Image
import torch
import os
import time
import numpy as np
import h5py
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import random
import string
from wsi_core.WholeSlideImage import DrawMap, StitchPatches
from PIL import ImageDraw
import math
# 12/05/2021
import pandas as pd

#%%
parser = argparse.ArgumentParser(description='CLAM Attention Map Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
					help='directory to save eval results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
#parser.add_argument('--bestk', type=int, default=10, help='the fold of the best model (default: 10)')
parser.add_argument('--k', type=int, default=10, help='number of total trained folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--downscale', type=int, default=64, help='downsample ratio to the splitting magnification (default: 64)')
parser.add_argument('--snapshot', action='store_true', default=False, help='export snapshot')
parser.add_argument('--grayscale', action='store_true', default=False, help='export grayscale heatmap')
parser.add_argument('--colormap', action='store_true', default=False, help='export colored heatmap')
parser.add_argument('--blended', action='store_true', default=False, help='export blended image')
parser.add_argument('--patch_bags', type=str, nargs='+', default=None, 
                    help='names of patch files (ends with .h5) for visualization (default: None)')
parser.add_argument('--B', type=int, default=-1, help='save the B positive and B negative patches used for slide-level decision (default: -1, not saving)')
## 12/05/2021
parser.add_argument('--tp', action='store_true', default=False, help='only process true positive slides')
args = parser.parse_args()

#%%
## Parameters for test
#parser = argparse.ArgumentParser(description='CLAM Attention Map Script')
#args = parser.parse_args()
#
#args.data_root_dir = "../results/patches_mondor_tumor" #'./results/patches_tumor'
#args.save_exp_code = 'mondor_hcc_tumor_258_Gajewski_13G_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv'
#args.k = 10
#args.k_start = -1
#args.k_end = 10
#args.fold = 8
#args.downscale = 8
#args.snapshot = True
#args.grayscale = True
#args.colormap = True
#args.blended = True
#args.patch_bags = ["HMNT0124 - 2017-06-05 04.23.20.h5"]
#args.B = 8

#%%

#if args.bestk >= 0:
#    bestk = args.bestk

#args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))

#os.makedirs(os.path.join(args.save_dir, "attention_maps_" + str(bestk)), exist_ok=True)

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
    

if __name__ == "__main__":
    ###*********************************
    # 12/05/2021, for only true positive
    
#    if args.patch_bags is not None:
#        patch_bags = args.patch_bags
#    else:
#        patch_bags = sorted(os.listdir(args.data_root_dir))
#        patch_bags = [patches for patches in patch_bags if os.path.isfile(os.path.join(args.data_root_dir, patches))]
        
    if not args.tp:
        if args.patch_bags is not None:
            patch_bags = args.patch_bags
        else:
            patch_bags = sorted(os.listdir(args.data_root_dir))
            patch_bags = [patches for patches in patch_bags if os.path.isfile(os.path.join(args.data_root_dir, patches))]
            
    ###*********************************

     
    for fold in folds:
        ###*********************************
        # 12/05/2021, for only true positive
        if args.tp:
            patch_bags = []
            
            tp_file = "fold_"+str(fold)+"_optimal_tcga.csv"
            print('Load the file indicating the true prositive slides: {}.'.format(tp_file))
            tp_file = os.path.join(args.save_dir, tp_file)
            
            tp_df = pd.read_csv(tp_file)
    
            # classify predictions by the columns "Y" and "true_prediction"
            for i in range(tp_df.shape[0]):
                if (tp_df.iloc[i, 1] == 1.0) and tp_df.iloc[i, 7]: #TP
                    patch_bags.append(tp_df.iloc[i, 0]+".h5")
            
        ###*********************************    
        
        save_dir = args.save_dir
        save_dir = os.path.join(save_dir, "attention_maps_" + str(fold) + "_" + 
                                     str(args.downscale))
        os.makedirs(save_dir, exist_ok=True)
        
        total = len(patch_bags)
        times = 0.
         
        for i in range(total): 
            print("\n\nprogress: {:.2f}, {}/{} in current model. {} out of {} models".format(i/total, i, total, folds.index(fold), len(folds)))
            print('processing {}'.format(patch_bags[i]))
            
            time_elapsed = -1
            start_time = time.time()
             
            fpatch = os.path.join(args.data_root_dir, patch_bags[i])
            f = h5py.File(fpatch, mode='r')
            
            # Studying the structure of the file by printing what HDF5 groups are present
            for key in f.keys():
                print(key) # Names of the groups in HDF5 file.
            
            #file_feature = "../results/colorectal/patches/TCGA-A6-6137-01A-01-TS1.58247feb-5bdc-41e1-9075-908a65c40273.h5" # test on colorectal
            #f = h5py.File(file_feature, mode='r')
            #
            ## Studying the structure of the file by printing what HDF5 groups are present
            #for key in f.keys():
            #    print(key) #Names of the groups in HDF5 file.
            #    
            ##Extracting the data
            #coords_feature = list(f["coords"])
            #
            ##After you are done
            #f.close()
            
            if args.snapshot:
                snapshot = StitchPatches(fpatch, downscale=args.downscale, bg_color=(255,255,255), alpha=-1, draw_grid=False)
                snapshot.save(os.path.join(save_dir, patch_bags[i].replace(".h5", "_snapshot.png")))
                #snapshot.show()
            
            att = torch.load(os.path.join(args.save_dir, "attention_scores_"+str(fold), "attention_score_"+
                                          patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage.cuda(0))
            print(att.shape)
            # torch.Size([1, 22857])
            list_att = att.data.tolist()[0]
            del(att)
            
            percentile = []
            for j in range(len(list_att)):
                percentile.append(stats.percentileofscore(list_att, list_att[j])) # the rank in ascending order
            print(len(percentile)) # 22857
            del(list_att)
            
            nor = [(x - min(percentile)) / (max(percentile) - min(percentile)) for x in percentile] # scale to [0, 1], 1 is the most attended
            del(percentile)
            
            # for the B highest and B lowest, save the original patch named with the attention score and coords
            if args.B > 0:
                patch_dir = os.path.join(args.save_dir, 'repres_patches_'+str(fold))
                os.makedirs(patch_dir, exist_ok=True)
                inds = list(range(args.B))
                inds.extend(list(range(-args.B,0)))
                sort_index = np.argsort(-np.array(nor)) # descending
                for n in inds:
                    ind = sort_index[n]
                    im = Image.fromarray(f["imgs"][ind])
                    im.save(os.path.join(patch_dir, patch_bags[i].replace(".h5", '_'+str(n)+'_'+str(nor[ind])+'_['+
                                         str(f["coords"][ind][0])+','+str(f["coords"][ind][1])+'].tif')))
                    
            
            cmap = matplotlib.cm.get_cmap('RdBu_r')
            heatmap = (cmap(nor)*255)[:,:3] # (N, 3)
            # del(nor)
            
            dset = f["imgs"][()] # convert dataset to array, to stop the pointer
            a = np.ones((256,256,3)) # to further enlarge 1 pixel to 256*256 patch
            for h in range(len(heatmap)):
                dset[h] = a * heatmap[h][np.newaxis, np.newaxis,:] # dset is the heatmap, in the format for stitching
            del(heatmap, a, cmap)
            
            # Add random letters to make sure they wouldn't affect each other with multiplethreading
            letters = string.ascii_lowercase
            result_str = "".join(random.choice(letters) for n in range(5))
            
            filename = "garbage_"+result_str+".h5"
            file = h5py.File(filename, mode='w')
            dset = file.create_dataset("imgs", data=dset) # convert array back to dataset
            for k in range(len(f["imgs"].attrs.values())): # "coords" have no attributes
                dset.attrs[list(f["imgs"].attrs.keys())[k]] = list(f["imgs"].attrs.values())[k]
            
            downscale=args.downscale
            draw_grid=False
            bg_color=(255,255,255)
            alpha=-1
    
            if 'downsampled_level_dim' in f["imgs"].attrs.keys():
                w, h = f["imgs"].attrs['downsampled_level_dim']
            else:
                w, h = f["imgs"].attrs['level_dim']
            print('original size: {} x {}'.format(w, h)) # the patching level size
    
            w = w // downscale
            h = h //downscale # downsample 64 of the patching level
            # f["coords"] = (f["coords"] / downscale).astype(np.int32)
            print('downscaled size for stiching: {} x {}'.format(w, h))
            print('number of patches: {}'.format(len(f["imgs"]))) # 22857
            img_shape = f["imgs"][0].shape # (256, 256, 3)
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
            
            print(patch_bags[i])
            # heatmap.show()
            if args.colormap:
                heatmap.save(os.path.join(save_dir, patch_bags[i].replace(".h5", "_heatmap.png")))
            
            if args.grayscale:
                grayscale = Image.new('L', (w, h))
                draw = ImageDraw.Draw(grayscale)
                
                for n in range(len(nor)):
                    xcoord = (f["coords"][n][0] / downscale).astype(np.int32)
                    ycoord = (f["coords"][n][1] / downscale).astype(np.int32)
                    draw.rectangle([(xcoord, ycoord), (xcoord+downscaled_shape[0], ycoord+downscaled_shape[1])], fill= math.floor(nor[n] * 255))
                grayscale.save(os.path.join(save_dir, patch_bags[i].replace(".h5", "_grayscale.png")))
            
            #After you are done
            file.close()
            # Delete this temporary file
            os.remove(filename)
            f.close()
            
            # Save blended image
            if args.blended:
                Image.blend(snapshot, heatmap, 0.3).save(os.path.join(save_dir, patch_bags[i].replace(".h5", "_blended.png")))
            
            time_elapsed = time.time() - start_time
            times += time_elapsed
            
        times /= total
        print("average time in s per slide: {}".format(times))
        
        # os.remove("./garbage_"+result_str+".h5")
