#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:43:36 2021
Key workflow:
1. Downscale from the splitting magnification (20x) ['downscale']
2. Load in the dim of 20x ['downsampled_level_dim'] from the slide (extracted using openslide by checking the highest magnfication, so no exception, one file for all the 3 mondor slide cases)
3. Calculate dim on acutal stitching magnification ['w', 'h'] and the downsampled level ['vis_level'] in slide
4. Stitching at the downsampled level (existed): atually no problem for snapshot, colormap, grayscale
5. Reisze the stitched images: so blended (here checked work for tcga and mondor with downscale<256)

Mondor slides are often too large for the 'dset' part, would block our workstation and finally be killed. 
So we run this script for all the mondor tp slides on the VM UNIV.
@author: Q Zeng
"""
#%%
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
from wsi_core.WholeSlideImage import DrawMap, StitchCoords, WholeSlideImage
from PIL import ImageDraw
import math
# 12/05/2021
import pandas as pd
###*********************************
#Modified by Qinghe 25/05/2021, for the fast pipeline
import openslide
###*********************************

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
                    help='names of patch files (ends with .h5) for visualization (default: None), overruled by tp')
parser.add_argument('--B', type=int, default=-1, help='save the B positive and B negative patches used for slide-level decision (default: -1, not saving)')
###*********************************
## 12/05/2021
parser.add_argument('--tp', action='store_true', default=False, help='only process true positive slides, overrule patch_bags')
# for fast pipeline
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--custom_downsample', type=int, default=1, help='overruled by target_patch_size')
parser.add_argument('--target_patch_size', type=int, default=-1, help='overrule custom_downsample')
# 27/05/2021, 27/05/2021, add cpu mode for VM NIUV, and add auto_skip
parser.add_argument('--cpu', default=False, action='store_true', help='force to use cpu') # if gpu not available, use cpu automatically
parser.add_argument('--auto_skip', default=False, action='store_true', help='auto skip checking if snapshot file exists')
###********* 03/02/2021, enable custom ext
parser.add_argument('--slide_ext', nargs="+", default= ['.svs', '.ndpi', '.tiff'], help='slide extensions to be recognized, svs/ndpi/tiff by default')
###*********************************
args = parser.parse_args()

#%%
## Parameters for test
#parser = argparse.ArgumentParser(description='CLAM Attention Map Script')
#args = parser.parse_args()
#
#args.data_root_dir = "/media/visiopharm5/WDRed(backup)/clam_extension/results/patches_mondor_tumor" #'./results/patches_tumor'
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
            ###*********************************
            # 25/05/2021, for fast pipeline       
            patch_level = f['coords'].attrs['patch_level']
            patch_size = f['coords'].attrs['patch_size']
            
            if patch_size == args.target_patch_size:
                target_patch_size = None
            elif args.target_patch_size > 0:
                target_patch_size = (args.target_patch_size, ) * 2
            elif args.custom_downsample > 1:
                target_patch_size = (patch_size // args.custom_downsample, ) * 2
            else:
                target_patch_size = None
            ###*********************************
            
            # Studying the structure of the file by printing what HDF5 groups are present
            for key in f.keys():
                print(key) # Names of the groups in HDF5 file.
            
            #file_feature = "/media/visiopharm5/WDGold/deeplearning/MIL/CLAM/results/colorectal/patches/TCGA-A6-6137-01A-01-TS1.58247feb-5bdc-41e1-9075-908a65c40273.h5"
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
            
            ###*********************************
            # 12/05/2021, for fast pipeline
            #slide_file_path = os.path.join(args.data_slide_dir, bag_name.replace('.h5', args.slide_ext))
            # priority: NOT, AND, OR!!
            slide_file_path = os.path.join(args.data_slide_dir, [sli for sli in os.listdir(args.data_slide_dir) if 
                                                                     sli.endswith(tuple(args.slide_ext)) and 
                                                                     sli.startswith(os.path.splitext(os.path.basename(fpatch))[0])][0])
            # 25/05/2021
            # For normal pipeline: "coords" have no attributes, "imgs" attributes: 'downsample'==np.array([1., 1.], 
            # 'downsampled_level_dim', 'level_dim', 'patch_level'==0, 'wsi_name'
            # For fast pipeline, "coords" have these attribute: 'downsample'==np.array([1., 1.] or np.array([2., 2.], 
            # 'downsampled_level_dim', 'level_dim', 'name', 'patch_level'==0 or 1, 'patch_size', 'save_path'
#            for k in range(len(f["imgs"].attrs.values())): 
#                dset.attrs[list(f["imgs"].attrs.keys())[k]] = list(f["imgs"].attrs.values())[k]
           
            with openslide.open_slide(slide_file_path) as wsi:
                ###*********************************
                ## 03/02/2022, openslide.PROPERTY_NAME_OBJECTIVE_POWER works for both ndpi and svs but not tiff generated by libvips
#                if slide_file_path.endswith("ndpi"):
                if slide_file_path.endswith("ndpi") or slide_file_path.endswith(".svs"):
                    if (wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] == '20'): # 20x(no exception magnification)~= 0.5 or 1.0 (not correctly recognized)
                        downsample = 1.0
                    elif (wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] == '40'): # 40x pixelsize ~= 0.25
                        downsample = 2.0
                    else:
                        raise Exception("The highest magnification should be 20x or 40x.")
                
                ## TOTEST
                elif slide_file_path.endswith(".tiff"):
                    # the properties in tiff slide generated by libvips can not be correctly decoded
                    loc1st = str(wsi.properties).find('openslide.objective-power&lt;/name&gt;\\n      &lt;value type="VipsRefString"&gt;')
                    if (str(wsi.properties)[loc1st+80:loc1st+82] == '20'):
                        downsample = 1.0
                    elif (str(wsi.properties)[loc1st+80:loc1st+82] == '40'):
                        downsample = 2.0
                    else:
                        raise Exception("The highest magnification should be 20x or 40x. Check your slide properties first.")
                        
                else:
                    raise Exception("Please indicate the downsample factor for your slide ext.")
                ###*********************************
    
                dset_attrs_downsampled_level_dim = np.asarray([np.int64(math.floor(wsi.dimensions[0]/downsample)), np.int64(math.floor(wsi.dimensions[1]/downsample))])
                dset_attrs_level_dim = np.asarray([np.int64(wsi.dimensions[0]), np.int64(wsi.dimensions[1])])
               
            dset_attrs_downsample = np.asarray([np.float64(1), np.float(1)])
            dset_attrs_patch_level = np.int64(0)
            dset_attrs_wsi_name = os.path.splitext(patch_bags[i])[0]

            # 27/05/2021, add auto_skip option
            if args.auto_skip and os.path.isfile(os.path.join(save_dir, patch_bags[i].replace(".h5", "_snapshot.png"))):
                print('{} already exist in destination location, skipped'.format(patch_bags[i].replace(".h5", "_snapshot.png")))
                continue
            ###*********************************
            
            if args.snapshot:
                ###*********************************
                # 05/2021, for fast pipeline
#                snapshot = StitchPatches(fpatch, downscale=args.downscale, bg_color=(255,255,255), alpha=-1, draw_grid=False)
                snapshot = StitchCoords(fpatch, WholeSlideImage(slide_file_path), downscale=args.downscale, 
                                        bg_color=(255,255,255), alpha=-1, draw_grid=False, downsampled_level_dim=dset_attrs_downsampled_level_dim)
                ###*********************************
                snapshot.save(os.path.join(save_dir, patch_bags[i].replace(".h5", "_snapshot.png")))
                #snapshot.show()
            
            ###*********************************
            # 27/05/2021, add cpu mode for VM NIUV
#            att = torch.load(os.path.join(args.save_dir, "attention_scores_"+str(fold), "attention_score_"+
#                                          patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage.cuda(0))
            if args.cpu:
                att = torch.load(os.path.join(args.save_dir, "attention_scores_"+str(fold), "attention_score_"+
                                              patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage)
            else:
                att = torch.load(os.path.join(args.save_dir, "attention_scores_"+str(fold), "attention_score_"+
                                              patch_bags[i].replace(".h5", ".pt")), map_location=lambda storage, loc: storage.cuda(0))
            ###*********************************
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
                    ###*********************************
                    # 25/05/2021, for fast pipeline
#                    im = Image.fromarray(f["imgs"][ind])
                    coords = f["coords"][ind]
            
                    with openslide.open_slide(slide_file_path) as wsi:
                        im = wsi.read_region(coords, patch_level, (patch_size, patch_size)).convert('RGB')
            
                    if target_patch_size is not None:
                        im = im.resize(target_patch_size) # (256, 256, 3)
                    ###*********************************

                    im.save(os.path.join(patch_dir, patch_bags[i].replace(".h5", '_'+str(n)+'_'+str(nor[ind])+'_['+
                                         str(f["coords"][ind][0])+','+str(f["coords"][ind][1])+'].tif'))) # coords on level 0
                    
            
            cmap = matplotlib.cm.get_cmap('RdBu_r')
            heatmap = (cmap(nor)*255)[:,:3] # (N, 3)
            # del(nor)
            
            ###*********************************
            # 25/05/2021, for fast pipeline
            #dset = f["imgs"][()] # convert dataset to array, to stop the pointer
            dset = np.zeros((f["coords"].shape[0], 256, 256, 3))
            ###*********************************
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
            
            ###*********************************
            # 25/05/2021, for fast pipeline
            # For normal pipeline: "coords" have no attributes, "imgs" attributes: 'downsample'==np.array([1., 1.], 
            # 'downsampled_level_dim', 'level_dim', 'patch_level'==0, 'wsi_name'
            # For fast pipeline, "coords" have these attribute: 'downsample'==np.array([1., 1.] or np.array([2., 2.], 
            # 'downsampled_level_dim', 'level_dim', 'name', 'patch_level'==0 or 1, 'patch_size', 'save_path'
#            for k in range(len(f["imgs"].attrs.values())): 
#                dset.attrs[list(f["imgs"].attrs.keys())[k]] = list(f["imgs"].attrs.values())[k]
            dset.attrs['downsample'] = dset_attrs_downsample
            dset.attrs['downsampled_level_dim'] = dset_attrs_downsampled_level_dim
            dset.attrs['level_dim'] = dset_attrs_level_dim
            dset.attrs['wsi_name'] = dset_attrs_wsi_name
            ###*********************************
            
            downscale=args.downscale
            draw_grid=False
            bg_color=(255,255,255)
            alpha=-1
    
            ###*********************************
            # 25/05/2021, for fast pipeline
#            if 'downsampled_level_dim' in f["imgs"].attrs.keys():
#                w, h = f["imgs"].attrs['downsampled_level_dim']
#            else:
#                w, h = f["imgs"].attrs['level_dim']
            w, h = dset.attrs['downsampled_level_dim']
            #print('original size: {} x {}'.format(w, h)) # the patching level size
            print('original size of 20x: {} x {}'.format(w, h)) # the actual patching level size (could be not existed level)
            ###*********************************
    
            w = w // downscale
            h = h //downscale # downsample 64 of the patching level
            # f["coords"] = (f["coords"] / downscale).astype(np.int32)
            print('downscaled size for stiching: {} x {}'.format(w, h))
            
            ###*********************************
            # 25/05/2021, for fast pipeline
#            print('number of patches: {}'.format(len(f["imgs"]))) # 22857
            print('number of patches: {}'.format(len(f["coords"])))
            
#            img_shape = f["imgs"][0].shape # (256, 256, 3)
            img_shape = (256, 256, 3)
            ###*********************************
            
            print('patch shape: {}'.format(img_shape))
    
            downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)
            
            if w*h > Image.MAX_IMAGE_PIXELS: 
                raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
            
            if alpha < 0 or alpha == -1:
                heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
            else:
                heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
            
            heatmap = np.array(heatmap)
            
            ###*********************************
            # 25/05/2021, for fast pipeline
#            heatmap = DrawMap(heatmap, dset, (f["coords"][:] / downscale).astype(np.int32), downscaled_shape, indices=None, draw_grid=draw_grid)
            heatmap = DrawMap(heatmap, dset, (f["coords"][:]/(dset.attrs['level_dim'][0]//w)).astype(np.int32), downscaled_shape, indices=None, draw_grid=draw_grid)
            ###*********************************
            
            print(patch_bags[i])
            # heatmap.show()
            if args.colormap:
                heatmap.save(os.path.join(save_dir, patch_bags[i].replace(".h5", "_heatmap.png")))
            
            if args.grayscale:
                grayscale = Image.new('L', (w, h))
                draw = ImageDraw.Draw(grayscale)
                
                for n in range(len(nor)):
                    ###*********************************
                    # 25/05/2021, for fast pipeline
#                    xcoord = (f["coords"][n][0] / downscale).astype(np.int32)
#                    ycoord = (f["coords"][n][1] / downscale).astype(np.int32)
                    xcoord = (f["coords"][n][0] / (dset.attrs['level_dim'][0]//w)).astype(np.int32)
                    ycoord = (f["coords"][n][1] / (dset.attrs['level_dim'][0]//w)).astype(np.int32)
                    ###*********************************
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
