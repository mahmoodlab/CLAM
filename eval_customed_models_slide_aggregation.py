#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 00:23:34 2021
Load the patch evaluation results from eval_custmed_models.py or eval_custmed_models_fp.py
And perform slide-level aggregation
@author: Q Zeng
"""

#%% 
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, precision_score
from sklearn.metrics import auc as calc_auc
import matplotlib.pyplot as plt
import statistics

#%% 
# Generic validation settings
parser = argparse.ArgumentParser(description='Configurations for WSI aggregation')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
                    help='the directory to save eval results relative to project root (default: ./eval_results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--thresholds_dir', type=str, default=None,
                    help='the directory base to load thresholds (default: None means to calculate cutoffs on the current dataset)')
args = parser.parse_args()

#%% 
# Parameters for test
#parser = argparse.ArgumentParser(description='Configurations for WSI aggregation')
#args = parser.parse_args()
#args.eval_dir = "./eval_results_349_custom/fromjeanzay/eval_results_349_custom"
#args.save_exp_code = "EVAL_mondor_hcc_tumor-masked_139_Interferon_Gamma_Biology_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv"
#args.thresholds_dir = "EVAL_mondor_hcc_tumor-masked_139_Interferon_Gamma_Biology_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv"
#args.k = 10

#%% 
def Find_Optimal_Cutoff(target, probs):
    fpr, tpr, thresholds = roc_curve(target, probs)
    
    listy = tpr - fpr
    optimal_idx = np.argwhere(listy == np.amax(listy))
    optimal_idx = optimal_idx.flatten().tolist()

    optimal_thresholds = [thresholds[optimal_id] for optimal_id in optimal_idx]

    if len(optimal_thresholds) > 1: # If multiple optimal cutoffs, the one closer to the median was chosen.
#         raise Exception("Multiple optimal cutoffs!")
        optimal_thresholds = [min(optimal_thresholds, key=lambda x:abs(x - statistics.median(thresholds)))]
        print("Multiple optimal cutoffs! {} is chosen.".format(str(optimal_thresholds)))

    return optimal_thresholds

#%% 
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

#%% slide-level aggregation
if __name__ == "__main__":
    fig = plt.figure(figsize=(15,15))
    
    path = os.path.join(args.eval_dir, args.save_exp_code)
    
    labels_for_roc = []
    probs_for_roc = []
    
    if args.thresholds_dir is None:
        opt_thresholds = []
    else:
        opt_thresholds = pd.read_csv(os.path.join(args.eval_dir, args.thresholds_dir, 'cutoffs.csv'), header=None, index_col=0)

    for fold in range(args.k):
        file = os.path.join(path, "split_{}_results.pkl".format(fold))
        unpickled_dict = pd.read_pickle(file)

        tile_ids = []
        for i in range(len(unpickled_dict.keys())):
            tile_ids.extend(unpickled_dict[i]['tile_ids'])
        print('In total, {} patches evaluated'.format(len(tile_ids)))
            
        probs = []
        labels = unpickled_dict[len(unpickled_dict.keys())-1]['labels']
        # extract probs of class 1
        for j in range(len(unpickled_dict[len(unpickled_dict.keys())-1]['probs'])):
            probs.append(unpickled_dict[len(unpickled_dict.keys())-1]['probs'][j][1])
            
        # plot histogram for patch-level probabilities 
        fig.add_subplot(4, 3, fold+1)
        plt.hist(np.array(probs), density=False, bins=10)  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data');
        plt.xlim((0, 1))
          
        if args.thresholds_dir is None:
            threshold = Find_Optimal_Cutoff(labels, probs)[0]
            print('Patch-level threshold of fold {}: {}'.format(fold, threshold))
            opt_thresholds.append(threshold)
        else:
            threshold = opt_thresholds.iloc[fold].values[0]
        
        slide_id = tile_ids[0].split(":")[0]
    
        n_patches = 0
        n_positives = 0
        slide_probs = {}
        slide_labels = {}
    
        for i in range(len(tile_ids)):
            if slide_id != tile_ids[i].split(":")[0]:
                slide_probs[slide_id] = n_positives / n_patches
                slide_labels[slide_id] = labels[i-1]
                slide_id = tile_ids[i].split(":")[0]
                n_patches = 0
                n_positives = 0
            if probs[i] >= threshold:
                n_positives = n_positives + 1
            n_patches = n_patches + 1
    
        labels_for_roc.append(list(slide_labels.values()))
        probs_for_roc.append(list(slide_probs.values()))
        
        df_slide = pd.DataFrame({'slide_id': list(slide_labels.keys()), 'Y': list(slide_labels.values()), 'p_1': list(slide_probs.values())})
        df_slide.to_csv(os.path.join(path, 'slide_aggregation_fold_{}.csv'.format(fold)))
        
    aucs, mean_auc, std_auc, max_auc, max_idx = draw_mean_roc(k = args.k, labels_for_roc = labels_for_roc, probs_for_roc = probs_for_roc)
    plt.savefig(os.path.join(path, "roc.png"))
#    plt.show()
    print("Mean AUC: {:.3f}".format(mean_auc))
    print("AUC sd: {:.3f}".format(std_auc))
    print("Max AUC: {:.3f}".format(max_auc))
    print("Max index: {}".format(max_idx))
    
    auc_summary = pd.DataFrame([aucs]).T
    if args.thresholds_dir is None:
        auc_summary.loc['Mean AUC'] = mean_auc
        auc_summary.loc['AUC sd'] = std_auc
        auc_summary.loc['Max AUC'] = max_auc
        auc_summary.loc['Max index'] = max_idx
    
    auc_summary.index.names = ['folds']
    auc_summary.rename(columns={0:'test_auc'}, inplace=True)
    auc_summary.to_csv(os.path.join(path, 'slide_aggregation_summary.csv'))
    
    if args.thresholds_dir is None:
        pd.DataFrame(opt_thresholds).to_csv(os.path.join(path, 'cutoffs.csv'), header=False)
