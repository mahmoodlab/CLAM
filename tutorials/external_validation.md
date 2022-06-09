Tutorial for External Validation
===========

The first four steps are shared by all the three deep learning approaches.

***Tissue segmentation - patch extraction - label preparation - dataset splitting***

### WSI preparation

#### Tissue segmentation

##### Without annotations
```shell
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --mask_save_dir results/masks_mondor --patch_save_dir results/patches_mondor --stitch_save_dir results/stitches_mondor
```
##### With annotations
```shell
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --mask_save_dir results/masks_mondor_tumor_masked --patch_save_dir results/patches_mondor_tumor_masked --stitch_save_dir results/stitches_mondor_tumor_masked --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
```
##### fp, without annotations
```shell
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_save_dir results/patches-tp_mondor --stitch_save_dir results/stitches_mondor
```
##### fp, with annotations
```shell
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --mask_save_dir results/masks_mondor_tumor_masked --patch_save_dir results/patches-tp_mondor_tumor_masked --stitch_save_dir results/stitches_mondor_tumor_masked --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
```


#### Patch extraction
In our external validation dataset (Mondor), there are three types of WSIs: 20x as highest magnification, 40x as highest magnification with and without 20x native level. We extracted patches on 20x, which means to extract from the first level, the second level and the first level with x2 custom downsampling, respectively. We thus split the satisfatory configuration into three csv files.

##### Without annotations
```shell
# 20x_level
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_mondor_20x.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches_mondor --stitch_save_dir results/stitches_mondor
# 40x_level1
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 1 --process_list process_list_edited_mondor_40x_level1.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches_mondor --stitch_save_dir results/stitches_mondor
# 40x_custom
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_mondor_40x_custom.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches_mondor --stitch_save_dir results/stitches_mondor
```
##### With annotations
```shell
# 20x_level
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_mondor_20x.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches_mondor --stitch_save_dir results/stitches_mondor --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
# 40x_level1
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 1 --process_list process_list_edited_mondor_40x_level1.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches_mondor --stitch_save_dir results/stitches_mondor --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
# 40x_custom
python create_patches.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_mondor_40x_custom.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches_mondor --stitch_save_dir results/stitches_mondor --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
```

##### fp, without annotations
```shell
# 20x_level
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_mondor_20x.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches-tp_mondor --stitch_save_dir results/stitches_mondor
# 40x_level1
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 1 --process_list process_list_edited_mondor_40x_level1.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches-tp_mondor --stitch_save_dir results/stitches_mondor
# 40x_custom
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_mondor_40x_custom.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches-tp_mondor --stitch_save_dir results/stitches_mondor
```

##### fp, with annotations
```shell
# 20x_level
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_mondor_20x.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches-tp_mondor --stitch_save_dir results/stitches_mondor --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
# 40x_level1
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 1 --process_list process_list_edited_mondor_40x_level1.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches-tp_mondor --stitch_save_dir results/stitches_mondor --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
# 40x_custom
python create_patches_fp.py --source PATH_TO_MONDOR_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_mondor_40x_custom.csv --patch --stitch --mask_save_dir results/masks_mondor --patch_save_dir results/patches-tp_mondor --stitch_save_dir results/stitches_mondor --use_annotations --annotation_type ANNO_FORMAT --annotation_dir PATH_TO_ANNO
```

### Label preparation
1. Prepare a file of slide id. Copy the column *slide_id* from the configuration file **[save_dir]/process_list_autogen.csv** with the extention (e.g. *.ndpi*) removed, and save as a new csv file (e.g. **mondor_hcc_feature_139.csv**) in the folder **dataset_csv**. 
2. Preprare WSI labels for each gene signature.
3. Dataset splitting: (gene signature *Inflammatory* used as an example for the following steps)
```shell
python create_splits_seq.py --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --seed 1 --label_frac 1 --k 10
```

***
### Approach 1: Patch-based strategy

***Inference - WSI-level aggregation***

#### Inference
##### without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python -u eval_customed_models.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --data_dir ./results/patches_mondor --trnsfrms imagenet --results_dir ./results/training_custom --eval_dir ./eval_results_349_custom --save_exp_code mondor_hcc_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type shufflenet --split test --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100
```
##### with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python -u eval_customed_models.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10--data_dir ./results/patches_mondor_tumor_masked --trnsfrms imagenet --results_dir ./results/training_custom_tumor_masked --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type shufflenet --split test --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100
```
##### fp, without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python -u eval_customed_models_fp.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --data_slide_dir PATH_TO_MONDOR_WSI --data_dir ./results/patches-tp_mondor --trnsfrms imagenet --results_dir ./results/training_custom --eval_dir ./eval_results_349_custom --save_exp_code mondor_hcc_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type shufflenet --split test --target_patch_size 256 --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100
```
##### fp, with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python -u eval_customed_models_fp.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --data_slide_dir PATH_TO_MONDOR_WSI --data_dir ./results/patches-tp_mondor_tumor_masked --trnsfrms imagenet --results_dir ./results/training_custom_tumor_masked --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type shufflenet --split test --target_patch_size 256 --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100
```

#### WSI-level Aggregation
The optimal cut-off calculated on TCGA test split is used here.
##### without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models_slide_aggregation.py --eval_dir ./eval_results_349_custom --save_exp_code EVAL_mondor_hcc_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --k 10 --thresholds_dir EVAL_tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv
```
##### with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models_slide_aggregation.py --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code EVAL_mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --k 10 --thresholds_dir EVAL_tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv
```

***
### Approach 2: Classic MIL

***Feature extraction - inference***

#### Feature extraction
##### without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir results/patches --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor --batch_size 256
```
##### with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir results/patches_tumor_masked --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor_tumor_masked --batch_size 256
```
##### fp, without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor --batch_size 256 --target_patch_size 256
```
##### fp, with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor_tumor_masked --batch_size 256 --target_patch_size 256
```

#### Inference
##### without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_MIL_50_s1 --eval_dir ./eval_results_349 --save_exp_code mondor_hcc_139_Inflammatory_cv_highvsrest_00X_MIL_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type mil
```
##### with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_MIL_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_MIL_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type mil
```

***
### Approach 3: CLAM

***Feature extraction - inference - extract attention scores - construct attention maps***

#### Feature extraction
##### without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir results/patches --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor --batch_size 256
```
##### with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir results/patches_tumor_masked --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor_tumor_masked --batch_size 256
```
##### fp, without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor --batch_size 256 --target_patch_size 256
```
##### fp, with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features_mondor_tumor_masked --batch_size 256 --target_patch_size 256
```

#### Inference
##### without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349 --save_exp_code mondor_hcc_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```
##### with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

#### Extract attention scores
##### without annotations
```shell
python attention_score.py --drop_out --k 10 --results_dir ./results/training_gene_signatures --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349 --save_exp_code mondor_hcc_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb  --model_size small --data_dir ./results/features --fold 5
```
##### with annotations
```shell
python attention_score.py --drop_out --k 10 --results_dir ./results/training_gene_signatures_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb  --model_size small --data_dir ./results/features_tumor_masked --fold 5
```

#### Construct attention maps
##### without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python attention_map.py --eval_dir ./eval_results_349 --save_exp_code mondor_hcc_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --k 10 --B 8 --downscale 4 --snapshot --grayscale --colormap --blended --data_root_dir ./results/patches_mondor --fold 5
```
##### with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python attention_map.py --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --k 10 --B 8 --downscale 4 --snapshot --grayscale --colormap --blended --data_root_dir ./results/patches_mondor_tumor_masked --fold 5
```
##### fp, without annotations
```shell
CUDA_VISIBLE_DEVICES=0 python attention_map_fp.py --eval_dir ./eval_results_349 --save_exp_code mondor_hcc_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --k 10 --B 8 --downscale 4 --snapshot --grayscale --colormap --blended --data_root_dir ./results/patches-fp_mondor --data_slide_dir PATH_TO_MONDOR_WSI --target_patch_size 256 --fold 5
```
##### fp, with annotations
```shell
CUDA_VISIBLE_DEVICES=0 python attention_map_fp.py --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --k 10 --B 8 --downscale 4 --snapshot --grayscale --colormap --blended --data_root_dir ./results/patches-fp_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --target_patch_size 256 --fold 5
```

