# Predict Immune and Inflammatory Gene Signature Expression Directly from Histology Images 


**Predict 6 gene signatures associated with response to nivolumab in  advanced hepatocellular carcinoma (HCC) from [Sangro, Bruno, et al](https://pubmed.ncbi.nlm.nih.gov/32710922/).**
- *6-Gene Interferon Gamma*
- *Gajewski 13-Gene Inflammatory*
- *Inflammatory*
- *Interferon Gamma Biology*
- *Ribas 10-Gene Interferon Gamma*
- *T-cell Exhaustion*

Clustering was performed on the gene expression data to generate slide labels. Tumoral areas were annotated on slides and only patches from tumoral area were used.
The deep learning models were trained and validated (60% training / 20% validation / 20% test) on the [TCGA LIHC dataset](https://portal.gdc.cancer.gov/projects/TCGA-LIHC), with 10-fold Monte-carlo cross validation. Our in-house dataset (from APHP Henri Mondor) was used for external validation.


**3 Deep learning approaches:**
- *Patch-based* ([original repo](https://github.com/jnkather/DeepHistology))
- *2 Multiple Instance Learning (MIL): CLAM and classic MIL* ([original repo](https://github.com/mahmoodlab/CLAM))

Results
===========
**AUROC in the discovery series (TCGA-LIHC):**

<table  align="center">
	<tbody>
		<tr>
			<td align="center" valign="center" rowspan="2"><b><sub>Gene signature<sub></td>
			<td align="center" valign="center" colspan="2"><b><sub>Patch-based<sub></td>
			<td align="center" valign="center" colspan="2"><b><sub>Classic MIL<sub></td>
			<td align="center" valign="center" colspan="2"><b><sub>CLAM<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Best fold<sub></td>
			<td align="center" valign="center"><b><sub>Mean ± sd<sub></td>
			<td align="center" valign="center"><b><sub>Best fold<sub></td>
			<td align="center" valign="center"><b><sub>Mean ± sd<sub></td>
			<td align="center" valign="center"><b><sub>Best fold<sub></td>
			<td align="center" valign="center"><b><sub>Mean ± sd<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>6G Interferon Gamma<sub></td>
			<td align="center" valign="center"><sub>0.661<sub></td>
			<td align="center" valign="center"><sub>0.560 ± 0.067<sub></td>
			<td align="center" valign="center"><sub>0.758<sub></td>
			<td align="center" valign="center"><sub>0.630 ± 0.078<sub></td>
			<td align="center" valign="center"><sub>0.780<sub></td>
			<td align="center" valign="center"><sub>0.635 ± 0.097<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Gajewski 13G Inflammatory<sub></td>
			<td align="center" valign="center"><b><sub>0.809<sub></td>
			<td align="center" valign="center"><b><sub>0.688 ± 0.062<sub></td>
			<td align="center" valign="center"><b><sub>0.893<sub></td>
			<td align="center" valign="center"><b><sub>0.694 ± 0.125<sub></td>
			<td align="center" valign="center"><b><sub>0.914<sub></td>
			<td align="center" valign="center"><b><sub>0.728 ± 0.096<sub><sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Inflammatory<sub></td>
			<td align="center" valign="center"><sub>0.706<sub></td>
			<td align="center" valign="center"><sub>0.580 ± 0.077<sub></td>
			<td align="center" valign="center"><sub>0.806<sub></td>
			<td align="center" valign="center"><sub>0.641 ± 0.123<sub></td>
			<td align="center" valign="center"><sub>0.796<sub></td>
			<td align="center" valign="center"><sub>0.665 ± 0.081<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Interferon Gamma biology<sub></td>
			<td align="center" valign="center"><sub>0.783<sub></td>
			<td align="center" valign="center"><sub>0.561 ± 0.119<sub></td>
			<td align="center" valign="center"><sub>0.677<sub></td>
			<td align="center" valign="center"><sub>0.610 ± 0.051<sub></td>
			<td align="center" valign="center"><sub>0.822<sub></td>
			<td align="center" valign="center"><sub>0.674 ± 0.102<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Ribas 10G Inflammatory<sub></td>
			<td align="center" valign="center"><sub>0.727<sub></td>
			<td align="center" valign="center"><sub>0.640 ± 0.074<sub></td>
			<td align="center" valign="center"><sub>0.726<sub></td>
			<td align="center" valign="center"><sub>0.618 ± 0.065<sub></td>
			<td align="center" valign="center"><sub>0.806<sub></td>
			<td align="center" valign="center"><sub>0.669 ± 0.067<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>T cell exhaustion<sub></td>
			<td align="center" valign="center"><sub>0.661<sub></td>
			<td align="center" valign="center"><sub>0.543 ± 0.073<sub><sub></td>
			<td align="center" valign="center"><sub>0.788<sub></td>
			<td align="center" valign="center"><sub>0.606 ± 0.086<sub></td>
			<td align="center" valign="center"><sub>0.788<sub></td>
			<td align="center" valign="center"><sub>0.577 ± 0.092<sub></td>
		</tr>
	</tbody>
</table>

**AUROC (of best-fold model) in the external validation series (Mondor):**

<table  align="center">
	<tbody>
		<tr>
			<td align="center" valign="center"><b><sub>Gene signature<sub></td>
			<td align="center" valign="center"><b><sub>Patch-based<sub></td>
			<td align="center" valign="center"><b><sub>Classic MIL<sub></td>
			<td align="center" valign="center"><b><sub>CLAM<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>6G Interferon Gamma<sub></td>
			<td align="center" valign="center"><sub>0.694<sub></td>
			<td align="center" valign="center"><sub>0.745<sub></td>
			<td align="center" valign="center"><sub>0.871<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Gajewski 13G Inflammatory<sub></td>
			<td align="center" valign="center"><sub>0.657<sub></td>
			<td align="center" valign="center"><sub>0.782<sub></td>
			<td align="center" valign="center"><sub>0.810<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Inflammatory<sub></td>
			<td align="center" valign="center"><sub>0.657<sub></td>
			<td align="center" valign="center"><sub>0.816<sub></td>
			<td align="center" valign="center"><sub>0.850<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Interferon Gamma biology<sub></td>
			<td align="center" valign="center"><sub>0.755<sub></td>
			<td align="center" valign="center"><sub>0.793<sub></td>
			<td align="center" valign="center"><sub>0.823<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>Ribas 10G Inflammatory<sub></td>
			<td align="center" valign="center"><sub>0.605<sub></td>
			<td align="center" valign="center"><sub>0.779<sub></td>
			<td align="center" valign="center"><sub>0.810<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><b><sub>T cell exhaustion<sub></td>
			<td align="center" valign="center"><b><sub>0.810<sub></td>
			<td align="center" valign="center"><b><sub>0.868<sub></td>
			<td align="center" valign="center"><b><sub>0.921<sub></td>
		</tr>
	</tbody>
</table>

**Visualization / exlainability:**
<img src="docs/vis.png" width="1000px" align="below" />

Workflow
===========
<img src="docs/workflow.png" width="1000px" align="below" />

## Part 1. Gene expression clustering 
***To generate labels for Whole Slide Images (WSIs)***
1. Process TCGA FPKM data with **gene_clust/codes/tcga_fpkm_processing.ipynb**
2. Perform hierarchical clustering with **gene_clust/codes/PlotHeatmapGeneSignature.R** (to reproduce the heatmap). Or using Python with **gene_clust/codes/tcga_fpkm_clustering.ipynb** (to get the same clustering results)

*All TCGA data used and clutering results are provided in **gene_clust/data** and **gene_clust/results**. Due to privacy issues, the data in Mondor series is not provided but commands for external validation are described below.*

## Part 2. Deep learning 
### Label preparation
1. Preprare sample labels for each gene signature with **gene_clust/codes/tcga_label_csv_for_clam.ipynb**
2. Dataset splitting: (gene signature *Inflammatory* used as an example for the following steps)
```shell
python create_splits_seq.py --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --seed 1 --label_frac 1 --k 10
```

### WSI preparation
1. Rough annotations of tumoral regions are provided in **data/annotations**. Tissue segmentation and patch extraction: 
<img src="docs/TCGA-2Y-A9H4-01Z-00-DX1.897C9E71-7FD7-4229-9A95-F61AE43D0FDA.jpg" width="300px" align="left" />
20x with saving both patch coordinates and images
```shell
python create_patches_fp.py --source data/data_tcga_hcc --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_20x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```
40x with saving both patch coordinates and images
```shell
python create_patches_fp.py --source data/data_tcga_hcc --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_40x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```
20x with saving only patch coordinates (named fp workflow, saving storage space for large dataset or multiple modified patch versions)
```shell
python create_patches_fp.py --source data/data_tcga_hcc --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --process_list process_list_edited_20x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```
40x with saving only patch coordinates
```shell
python create_patches_fp.py --source data/data_tcga_hcc --save_dir results --patch_size 256 --step_size 256 --seg --patch_level 0 --custom_downsample 2 --process_list process_list_edited_40x.csv --patch --stitch --mask_save_dir results/masks_tumor --patch_save_dir results/patches_tumor_masked --stitch_save_dir results/stitches_tumor_masked --use_annotations --annotation_type txt --annotation_dir data/annotations
```
Segmention and stitched results will be saved in **results**. A segmentation example presented (green: tissue, blue: holes, red: tumor).


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
CUDA_VISIBLE_DEVICES=1 python train_customed_models_fp.py --early_stopping --patience 2 --min_epochs 5 --lr 5e-5 --reg 1e-5 --opt adam --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 10 --label_frac 1 --data_dir ./results/patches_tumor_masked --data_slide_dir ./data/data_tcga_hcc --target_patch_size 256 --trnsfrms imagenet --results_dir ./results/training_custom --exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet --train_weighting --bag_loss ce --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --freeze 3 --log_data > log_Inflammatory_shufflenet_frz3_imagenet.txt
```
#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval_customed_models.py --batch_size 512 --seed 1 --k 10 --k_start -1 --k_end 10 --data_dir ./results/patches_tumor_masked --trnsfrms imagenet --results_dir ./results/training_custom --eval_dir ./eval_results_349_custom --save_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1_cv --models_exp_code tcga_hcc_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_s1 --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --split test
```
#### Aggregation

#### External validation with 


### Multiple instance learning (MIL) strategy

#### Installation

OS: Linux (Tested on Ubuntu 18.04)

Please refer to Mahmood Lab's [Installation guide](INSTALLATION.md) for detailed instructions. Of note, please use **clam.yml** for the environment creation if you would like to use our modified/added functions.

#### Feature exaction
Encode the patches into 512-dimensional features using the default network ResNet50 pretrained on ImageNet
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir ./results/patches_tumor/ --csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir ./results/features_tumor --batch_size 256  --model resnet50
```
### Approach 2: Classic MIL
#### Training
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir ./results/features_tumor --results_dir ./results/training_gene_signatures --exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8
```
By default results will be saved to results/exp_code corresponding to the exp_code input argument from the user. If tensorboard logging is enabled (with the arugment toggle --log_data), the user can go into the results folder for the particular experiment, run:
```shell
tensorboard --logdir=.
```

#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir ./results/features_tumor --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --save_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small
```
##### External validation
We tested the 10 models (trained on TCGA) on the whole Mondor series.
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir ./results/features_mondor_tumor --splits_dir ./splits/mondor_hcc_258_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --save_exp_code mondor_hcc_tumor_258_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_258_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

### Approach 3: CLAM

***Clustering-constrained Attention Multiple Instance Learning**

A deep-learning-based weakly-supervised method that uses attention-based learning to automatically identify sub-regions of high diagnostic value in order to accurately classify the whole slide, while also utilizing instance-level clustering over the representative regions identified to constrain and refine the feature space.*

#### Reference
*Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w*

#### Original repository
[Github repository](https://github.com/mahmoodlab/CLAM) © [Mahmood Lab](http://www.mahmoodlab.org)
[Interactive Demo](http://clam.mahmoodlab.org) 


#### Training
For evaluating the CLAM's performance, 10-fold of train/val/test splits were used. We used 10-fold 60/20/20 splits for tcga-lihc, with 100% of training data can be found under the splits folder. These splits can be automatically generated using the create_splits_seq.py script with minimal modification just like with main.py. For example, gene signature of Inflammatory:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir ./results/features_tumor --results_dir ./results/training_gene_signatures --exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8
```
By default results will be saved to results/exp_code corresponding to the exp_code input argument from the user. If tensorboard logging is enabled (with the arugment toggle --log_data), the user can go into the results folder for the particular experiment, run:
```shell
tensorboard --logdir=.
```

#### Inference
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir ./results/features_tumor --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --save_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small
```
##### External validation
We tested the 10 models (trained on TCGA) on the whole Mondor series.
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir ./results/features_mondor_tumor --splits_dir ./splits/mondor_hcc_258_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --save_exp_code mondor_hcc_tumor_258_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_258_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```
#### Extract attention scores
Calculate the attention scores and visualize the attention map for interpretability. The user could pass --fold for a specific fold, or --k_start and --k_end to specify a fold range, otherwise the default will process all the k folds.

Here we used TCGA test set for example:
```shell
CUDA_VISIBLE_DEVICES=0 python attention_score.py --drop_out --k 10 --results_dir ./results/training --models_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --save_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --fold 5 --data_dir ./results/features_tumor
```
#### Construct attention maps
```shell
CUDA_VISIBLE_DEVICES=0 python attention_map.py --save_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --fold 5  --downscale 4  --snapshot --grayscale --colormap --blended --data_root_dir ./results/patches_tumor
```
### Other training settings
#### Color unmixing with Macenko PCA method






