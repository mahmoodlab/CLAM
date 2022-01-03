Tutorial for Patch-Based Approach (fp, without annotations)
===========

#### Reference
*Kather, J. N., Heij, L. R., Grabsch, H. I., Loeffler, C., Echle, A., Muti, H. S., ... & Luedde, T. (2020). Pan-cancer image-based detection of clinically actionable genetic alterations. Nature cancer, 1(8), 789-799.*


***Installation - tissue segmentation - patch extraction - label preparation - dataset splitting - training - inference - WSI-level aggregation***


#### Installation
OS: Linux (Tested on Ubuntu 18.04)
1. Install Anaconda with Python 3 via [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
2. Install *openslide*:
```shell
sudo apt-get install openslide-tools
```
3. Create conda environment with **deepai.yml**:
```shell
conda env create -n deepai -f docs/deepai.yml
```
4. Activate the conda environment:
```shell
conda activate deepai
```
5. Deactivate the conda environment when finishing the experiments:
```shell
conda deactivate
```

***
### WSI preparation

#### Tissue segmentation
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg
```
The segmented mask will be saved in **[save_dir]/masks**, and the folders **[save_dir]/patches** and **[save_dir]/stitches** will be empty. A configuration file **[save_dir]/process_list_autogen.csv** will be generated automatically. There is a possibility to fine-tune all the configuration for segmentation, filtering, visualization and patching for each WSI. But remember to rename the file (e.g. **process_list_edited.csv**) before modify it, otherwise it will be overwritten by a new execution.

Now you can redo the segmentation with modified configuration using the following command
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --process_list process_list_edited.csv
```

A segmentation example presented on the right (green --> tissue, blue --> holes).
<img src="../docs/seg_A9H4.png" width="350px" align="right" />

After having a satifying segmentation, here we seperate the modified configuration file into 2 files (**process_list_edited_20x.csv** and **process_list_edited_40x.csv**), for WSIs whose highest magnification are 20x and 40x, respectively. It would facilitate the next step as the patching commands for two kinds of WSIs are slightly different, otherwise the values of *process* column will have to modified to select WSI(s) to process (1 means to select while 0 means to exlude).

#### Patch extraction
For 20x WSIs
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_20x.csv --patch --stitch --mask_save_dir results/masks --patch_save_dir results/patches-fp --stitch_save_dir results/stitches
```

For 40x WSIs, we need to downsample to 20x as there is no such a native level
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_40x.csv --patch --stitch --mask_save_dir results/masks --patch_save_dir results/patches-fp --stitch_save_dir results/stitches
```

Patches extracted from several example WSIs can be found in [**results/patches-fp**](../results/patches-fp) (to play with the follwoing steps). 

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
#### Training
```shell
CUDA_VISIBLE_DEVICES=0 python train_customed_models_fp.py --early_stopping --patience 2 --min_epochs 5 --lr 5e-5 --reg 1e-5 --opt adam --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --label_frac 1 --data_dir ./results/patches-fp --data_slide_dir PATH_TO_TCGA_WSI --target_patch_size 256 --trnsfrms imagenet --results_dir ./results/training_custom --exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet --train_weighting --bag_loss ce --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --freeze 3 --log_data > log_Inflammatory_shufflenet_frz3_imagenet.txt
```

#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models_fp.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --data_slide_dir PATH_TO_TCGA_WSI --data_dir ./results/patches-fp --trnsfrms imagenet --results_dir ./results/training_custom --eval_dir ./eval_results_349_custom --save_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --split test --target_patch_size 256
```

#### WSI-level Aggregation
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models_slide_aggregation.py --eval_dir ./eval_results_349_custom --save_exp_code EVAL_tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv --k 10
```
Without passing the *--thresholds_dir* explicitly, the optimal threshold will be automatically calculated on the current test data.



