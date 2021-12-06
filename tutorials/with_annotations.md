### WSI preparation
1. Rough annotations of tumoral regions are provided in **data/annotations**. Tissue segmentation and patch extraction: 

For patch extraction, there are 2 options according to the original CLAM repository. Either to save both patch coordinates and images, or only save only patch coordinates. The second option is named fp workflow, which helps to save storage space for large dataset or multiple modified patch versions.

For 20x WSIs
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_20x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```
For 40x WSIs
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_40x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```
For 20x WSIs with fp workflow
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_20x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```
For 40x WSIs with fp workflow
```shell
python create_patches_fp.py --source PATH_TO_TCGA_WSI --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_40x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```

<img src="docs/TCGA-2Y-A9H4-01Z-00-DX1.897C9E71-7FD7-4229-9A95-F61AE43D0FDA.jpg" width="350px" align="right" />

Segmention and stitched results will be saved in **results**. Please find in [results/patches_tumor_masked](https://github.com/qinghezeng/CLAM/tree/master/results/patches_tumor_masked) some patch examples (to play with the follwoing steps). 

A segmentation example presented on the right (green --> tissue, blue --> holes, red --> tumor).


### Label preparation
1. Preprare WSI labels for each gene signature with **gene_clust/codes/tcga_label_csv_for_clam.ipynb**
2. Dataset splitting: (gene signature *Inflammatory* used as an example for the following steps)
```shell
python create_splits_seq.py --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --seed 1 --label_frac 1 --k 10
```


***
### Approach 1: Patch-based strategy
#### Reference
*Kather, J. N., Heij, L. R., Grabsch, H. I., Loeffler, C., Echle, A., Muti, H. S., ... & Luedde, T. (2020). Pan-cancer image-based detection of clinically actionable genetic alterations. Nature cancer, 1(8), 789-799.*

#### Installation
OS: Linux (Tested on Ubuntu 18.04)
1. Install Anaconda with Python 3 via [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
2. Install *openslide*:
```shell
sudo apt-get install openslide-tools
```
3. Create a conda environment with **deepai.yml**: (to use ShuffleNet, we need a higher pytorch version than that of CLAM environment described below)
```shell
conda env create -n clam -f docs/deepai.yaml
```
4. Activate the conda environment:
```shell
conda activate deepai
```
5. Deactivate the conda environment when finishing the experiments:
```shell
conda deactivate
```

#### Training
```shell
CUDA_VISIBLE_DEVICES=0 python train_customed_models.py --early_stopping --patience 2 --min_epochs 5 --lr 5e-5 --reg 1e-5 --opt adam --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --label_frac 1 --data_dir ./results/patches_tumor_masked --trnsfrms imagenet --results_dir ./results/training_custom_tumor_masked --exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet --train_weighting --bag_loss ce --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --freeze 3 --log_data > log_Inflammatory_shufflenet_frz3_imagenet.txt
```
For fp workflow
```shell
CUDA_VISIBLE_DEVICES=0 python train_customed_models_fp.py --early_stopping --patience 2 --min_epochs 5 --lr 5e-5 --reg 1e-5 --opt adam --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --label_frac 1 --data_dir ./results/patches_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --target_patch_size 256 --trnsfrms imagenet --results_dir ./results/training_custom_tumor_masked --exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet --train_weighting --bag_loss ce --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --freeze 3 --log_data > log_Inflammatory_shufflenet_frz3_imagenet.txt
```
#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --data_dir ./results/patches_tumor_masked --trnsfrms imagenet --results_dir ./results/training_custom_tumor_masked --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --split test
```
For fp workflow
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models_fp.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --data_slide_dir PATH_TO_TCGA_WSI --data_dir ./results/patches_tumor_masked --trnsfrms imagenet --results_dir ./results/training_custom_tumor_masked --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --split test --target_patch_size 256
```

#### External validation
For fp workflow
```shell
CUDA_VISIBLE_DEVICES=0 python -u eval_customed_models_fp.py --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --data_slide_dir PATH_TO_MONDOR_WSI --data_dir ./results/patches_mondor_tumor_masked --trnsfrms imagenet --results_dir ./results/training_custom_tumor_masked --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type shufflenet --split test --target_patch_size 256 --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100
```

#### WSI-level Aggregation
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models_slide_aggregation.py --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code EVAL_tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv --k 10
```

The optimal cut-off will be save in as **PATH_TO_EVAL_RESULTS/cutoffs.csv**, to be used for external validation. For external validation:
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models_slide_aggregation.py --eval_dir ./eval_results_349_custom_tumor_masked --save_exp_code EVAL_mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_shufflenet_frz3_imagenet_s1_cv --k 10 --thresholds_dir EVAL_tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv
```

***
### Multiple instance learning (MIL) strategy

#### Installation

OS: Linux (Tested on Ubuntu 18.04)

Please refer our [installation tutorial] (install_clam.md) for detailed instructions about the conda environment creation.

#### Feature exaction
Encode the patches into 512-dimensional features using the ResNet50 pretrained on ImageNet
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_tumor_masked ---csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir results/features_tumor_masked --batch_size 256
```
For fp workflow
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir results/features_tumor_masked --target_patch_size 256 --batch_size 256
```
### Approach 2: Classic MIL
#### Training
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir ./results/features_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked --exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_MIL_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type mil --log_data --B 8 > log_Inflammatory_mil.txt
```
 
#### Training visulaization
If tensorboard logging is enabled (with the arugment toggle --log_data), run this command in the results folder for the particular experiment:
```shell
tensorboard --logdir=.
```

#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir ./results/features_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_MIL_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_MIL_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type mil
```
##### External validation
We tested the 10 models (trained on TCGA) on the whole Mondor series.
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir ./results/features_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_MIL_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_MIL_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type mil
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

### Approach 3: CLAM

***Clustering-constrained Attention Multiple Instance Learning***

*A deep-learning-based weakly-supervised method that uses attention-based learning to automatically identify sub-regions of high diagnostic value in order to accurately classify the whole slide, while also utilizing instance-level clustering over the representative regions identified to constrain and refine the feature space.*

#### Reference
*Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w*

[Original Github repository](https://github.com/mahmoodlab/CLAM) © [Mahmood Lab](http://www.mahmoodlab.org)

[Interactive Demo](http://clam.mahmoodlab.org) 


#### Training
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features_tumor_masked --results_dir results/training_gene_signatures_tumor_masked --exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8 > log_Inflammatory_clam_tumor_masked.txt
```

#### Training visulaization
If tensorboard logging is enabled (with the arugment toggle --log_data), run this command in the results folder for the particular experiment:
```shell
tensorboard --logdir=.
```

#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small
```
##### External validation
We tested the 10 models (trained on TCGA) on the whole Mondor series.
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked --save_exp_code mondor_hcc_tumor-masked_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```
#### Extract attention scores
Calculate the attention scores and visualize the attention map for interpretability. The user could pass --fold for a specific fold, or --k_start and --k_end to specify a fold range, otherwise the default will process all the k folds.

Here we used TCGA test set for example:
```shell
CUDA_VISIBLE_DEVICES=0 python attention_score.py --drop_out --k 10 --results_dir ./results/training --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --save_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --fold 5 --data_dir ./results/features_tumor
```
#### Construct attention maps
```shell
CUDA_VISIBLE_DEVICES=0 python attention_map.py --save_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --fold 5  --downscale 4  --snapshot --grayscale --colormap --blended --data_root_dir ./results/patches_tumor
```
