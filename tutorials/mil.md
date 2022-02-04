Tutorial for MIL Approach (without annotations)
===========

***Classic Multiple Instance Learning***

*Max-pooling based, predict the WSI-level label by the highliest predictive patch. If the highliest predictive patch is Cluster High meaning that there are one or more Cluster High patches, this WSI will predicted as Cluster High. If the highliest predictive patch is Cluster Median/Low meaning that there is no Cluster High patch, this WSI will be predicted as Cluster Median/Low.*


#### Reference

*Maron O, Lozano-Pérez T. A Framework for Multiple-Instance Learning. Adv Neural Inf Process Syst 1997;10.*
*Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w*

[Original Github repository](https://github.com/mahmoodlab/CLAM) © [Mahmood Lab](http://www.mahmoodlab.org)


***Installation - tissue segmentation - patch extraction - label preparation - feature extraction - dataset splitting - training - training visulaization - inference***


#### Installation
OS: Linux (Tested on Ubuntu 18.04) 
The installation will be done along with that of CLAM. Please refer to the tutorial [install_clam](./install_clam.md)


***
### WSI preparation
```shell
python create_patches.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --mask_save_dir results/masks --patch_save_dir results/patches --stitch_save_dir results/stitches
```
The segmented mask will be saved in **[save_dir]/masks**, and the folders **[save_dir]/patches** and **[save_dir]/stitches** will be empty. A configuration file **[save_dir]/process_list_autogen.csv** will be generated automatically. There is a possibility to fine-tune all the configuration for segmentation, filtering, visualization and patching for each WSI. But remember to rename the file (e.g. **process_list_edited.csv**) before modify it, otherwise it will be overwritten by a new execution.

Now you can redo the segmentation with modified configuration using the following command
```shell
python create_patches.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --mask_save_dir results/masks --patch_save_dir results/patches --stitch_save_dir results/stitches --process_list process_list_edited.csv
```

A segmentation example presented on the right (green --> tissue, blue --> holes).
<img src="../docs/seg_A9H4.png" width="350px" align="right" />

After having a satifying segmentation, here we seperate the modified configuration file into 2 files (**process_list_edited_20x.csv** and **process_list_edited_40x.csv**), for WSIs whose highest magnification are 20x and 40x, respectively. It would facilitate the next step as the patching commands for two kinds of WSIs are slightly different, otherwise the values of *process* column will have to modified to select WSI(s) to process (1 means to select while 0 means to exlude).

#### Patch extraction
For 20x WSIs
```shell
python create_patches.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_20x.csv --patch --stitch --mask_save_dir results/masks --patch_save_dir results/patches --stitch_save_dir results/stitches
```

For 40x WSIs, we need to downsample to 20x as there is no such a native level
```shell
python create_patches.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_40x.csv --patch --stitch --mask_save_dir results/masks --patch_save_dir results/patches --stitch_save_dir results/stitches
```


***
### Label preparation
1. Prepare a file of slide id. Copy the column *slide_id* from the configuration file **[save_dir]/process_list_autogen.csv** with the extention (e.g. *.svs*) removed, and save as a new csv file (e.g. **tcga_hcc_feature_349.csv**) in the folder **dataset_csv**. 
2. Preprare WSI labels for each gene signature with **gene_clust/codes/tcga_label_csv_for_clam.ipynb**.
3. Dataset splitting: (gene signature *Inflammatory* used as an example for the following steps)
```shell
python create_splits_seq.py --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --seed 1 --label_frac 1 --k 10
```

***
### Deep learning
#### Feature extraction
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir results/patches --csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir results/features --batch_size 256
```


#### Training
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features --results_dir results/training_gene_signatures --exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_MIL_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type mil --log_data --B 8 > log_Inflammatory_mil.txt
```

#### Training visulaization
If tensorboard logging is enabled (with the arugment toggle --log_data), run this command in the results folder for the particular experiment (**results/training_gene_signatures/tcga_hcc_349_Inflammatory_cv_highvsrest_622_MIL_50_s1**):
```shell
tensorboard --logdir=.
```

#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_MIL_50_s1 --eval_dir ./eval_results_349 --save_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_MIL_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type mil
```




