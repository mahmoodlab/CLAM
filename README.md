### Predict Immune and Inflammatory Gene Signature Expression Directly from Histology Images 

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

**AUROC (of best-fold model) in the external validation series (private dataset from APHP Henri Mondor):**

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

<img src="docs/workflow.png" width="1000px" align="below" />

Part 1. Gene expression clustering 
===========
**To generate labels for Whole Slide Images (WSIs)**

Part 2. Deep learning 
===========

## Approach 1: CLAM

**Clustering-constrained Attention Multiple Instance Learning**

A deep-learning-based weakly-supervised method that uses attention-based learning to automatically identify sub-regions of high diagnostic value in order to accurately classify the whole slide, while also utilizing instance-level clustering over the representative regions identified to constrain and refine the feature space.

#### Reference
Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images
[ArXiv](https://arxiv.org/abs/2004.09666)
```
@inproceedings{lu2020clam,
  title     = {Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images},
  author    = {Ming Y. Lu, Drew F. K. Williamson, Tiffany Y. Chen, Richard J. Chen, Matteo Barbieri, Faisal Mahmood},
  booktitle = {Nature Biomedical Engineering - In Press},
  year = {2020}
}
```

#### Original repository
[Github repository](https://github.com/mahmoodlab/CLAM) © [Mahmood Lab](http://www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

[Interactive Demo](http://clam.mahmoodlab.org) 
***

***Modified by Qinghe ZENG to apply for the prediction of the gene signatures associated with response to nivolumab from the [sangro paper](https://pubmed.ncbi.nlm.nih.gov/32710922/).***

**Gene signatures:**
- 6-Gene Interferon Gamma
- Gajewski 13-Gene Inflammatory
- Inflammatory
- Interferon Gamma Biology
- Ribas 10-Gene Interferon Gamma
- T-cell Exhaustion

**The models were trained and validated on the TCGA LIHC dataset. Our in-house dataset (Mondor series) was used for external validation.**

**Clustering was performed on the gene expression data to generate slide labels. Tumoral areas were annotated on slides and only patches from tumoral area were used.**
***

## Installation

OS: Linux (Tested on Ubuntu 18.04)

Please refer to Mahmood Lab's [Installation guide](INSTALLATION.md) for detailed instructions. 

## Usage

#### Feature exaction
Encode the patches into 512-dimensional features using the default network ResNet50 pretrained on ImageNet
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir ./results/patches_tumor/ --csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir ./results/features_tumor --batch_size 256  --model resnet50
```

#### Dataset splitting
```shell
python create_splits_seq.py --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --seed 1 --label_frac 1 --k 10
```

#### Training
For evaluating the CLAM's performance, 10-fold of train/val/test splits were used. We used 10-fold 60/20/20 splits for tcga-lihc, with 100% of training data can be found under the splits folder. These splits can be automatically generated using the create_splits_seq.py script with minimal modification just like with main.py. For example, gene signature of Inflammatory:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir ./results/features_tumor --results_dir ./results/training_gene_signatures --exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8
```
By default results will be saved to results/exp_code corresponding to the exp_code input argument from the user. If tensorboard logging is enabled (with the arugment toggle --log_data), the user can go into the results folder for the particular experiment, run:
```shell
tensorboard --logdir=.
```

#### Testing
##### TCGA test set
```shell
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir ./results/features_tumor --results_dir ./results/training_gene_signatures --models_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --save_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small
```
##### Mondor
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
***

## Approach 2: Classic MIL


## Approach 3: Patch-based method
### Workflow proposed in Dr. Kather's [pan-cancer paper](https://www.nature.com/articles/s43018-020-0087-6)
#### Training
For this workflow, we have to create a new conda environment. Newer version packages are needed to use some pytorch pre-trained models.
```shell
CUDA_VISIBLE_DEVICES=0 python train_customed_models.py --early_stopping --patience 5 --max_epochs 30 --min_epochs 12 --lr 5e-5 --reg 1e-5 --opt adam --batch_size 128 --seed 1 --k 10 --k_start -1 --k_end 3 --label_frac 1 --data_dir ./results/patches_tumor --trnsfrms imagenet --results_dir ./results/training_custom --exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_shufflenet_frz3_imagenet_blc3 --train_weighting --bag_loss ce --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet --freeze 3
```
#### Testing
```shell
CUDA_VISIBLE_DEVICES=0 python eval_custom.py --batch_size 128 --k 10 --k_start -1 --k_end 10 --data_dir ./results/patches_tumor --results_dir ./results/training_custom --models_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_shufflenet_s1 --save_exp_code tcga_hcc_tumor_349_Inflammatory_cv_highvsrest_622_shufflenet_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type shufflenet
```





