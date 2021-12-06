# Predict Immune and Inflammatory Gene Signature Expression Directly from Histology Images 


**Predict 6 gene signatures associated with response to nivolumab and survival in advanced hepatocellular carcinoma (HCC) from [Sangro, Bruno, et al](https://pubmed.ncbi.nlm.nih.gov/32710922/).**
- ***6-Gene Interferon Gamma** ([Ayers, Mark, et al.](https://pubmed.ncbi.nlm.nih.gov/28650338/))* 
- ***Gajewski 13-Gene Inflammatory** ([Spranger, Stefani, Riyue Bao, and Thomas F. Gajewski.](https://pubmed.ncbi.nlm.nih.gov/25970248/))*
- ***Inflammatory** ([Sangro, Bruno, et al](https://pubmed.ncbi.nlm.nih.gov/32710922/))* 
- ***Interferon Gamma Biology** ([Ayers, Mark, et al.](https://pubmed.ncbi.nlm.nih.gov/28650338/))* 
- ***Ribas 10-Gene Interferon Gamma** ([Ayers, Mark, et al.](https://pubmed.ncbi.nlm.nih.gov/28650338/))* 
- ***T-cell Exhaustion** ([Ayers, Mark, et al.](https://pubmed.ncbi.nlm.nih.gov/28650338/))* 

Hierarchical clustering was performed on the gene expression data to generate labels for Whole Slide Images (WSIs). The deep learning models were trained (60%) with 10-fold Monte-carlo cross validation (20%) and tested (20%) on the [TCGA LIHC dataset](https://portal.gdc.cancer.gov/projects/TCGA-LIHC). Our in-house dataset (from APHP Henri Mondor) was then used for external validation. Results using tumoral annotations (regions of interest drawn by our expert pathologist)  are superior to those using all the tissue regions.

Of note, the discovery series was stained with hematein-eosin (H&E) while external validation series was stained with hematein-eosin-saffron (HES). Thus we tested stain unmixing (3 methods implemented: Macenko PCA or XU SNMF or a fixed HES vector) and saffron removal for external validation series. Color noralization (2 methods: Reinhard or Macenco PCA) was also tested for both discovery and validation series. Furthermore, on-the-fly basic geometric augmentation were also tested during the training.


**3 Deep learning approaches:**
- *Patch-based* ([original repo](https://github.com/jnkather/DeepHistology))
- *2 Multiple Instance Learning (MIL): CLAM and classic MIL* ([original repo](https://github.com/mahmoodlab/CLAM))

Results
===========
**AUROC in the discovery series (TCGA-LIHC) with/without tumoral annotations:**

<table  align="center">
	<tbody>
		<tr>
			<td align="center" valign="center" rowspan="2"><b><sub>Gene signature<sub></td>
			<td align="center" valign="center" rowspan="2"><b><sub>tumor annot<sub></td>
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
			<td align="center" valign="center" rowspan="2"><b><sub>6G Interferon Gamma<sub></td>
			<td align="center" valign="center"><sub>:x:<sub></td>
			<td align="center" valign="center"><sub>0.578<sub></td>
			<td align="center" valign="center"><sub>0.492 ± 0.065<sub></td>
			<td align="center" valign="center"><sub>0.690<sub></td>
			<td align="center" valign="center"><sub>0.576 ± 0.102<sub></td>
			<td align="center" valign="center"><sub>0.734<sub></td>
			<td align="center" valign="center"><sub>0.600 ± 0.080<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><sub>:heavy_check_mark:<sub></td>	
			<td align="center" valign="center"><sub>0.661<sub></td>
			<td align="center" valign="center"><sub>0.560 ± 0.067<sub></td>
			<td align="center" valign="center"><sub>0.758<sub></td>
			<td align="center" valign="center"><sub>0.630 ± 0.078<sub></td>
			<td align="center" valign="center"><sub>0.780<sub></td>
			<td align="center" valign="center"><sub>0.635 ± 0.097<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center" rowspan="2"><b><sub>Gajewski 13G Inflammatory<sub></td>
			<td align="center" valign="center"><sub>:x:<sub></td>
			<td align="center" valign="center"><b><sub>0.780<sub></td>
			<td align="center" valign="center"><b><sub>0.666 ± 0.072<sub></td>
			<td align="center" valign="center"><b><sub>0.851<sub></td>
			<td align="center" valign="center"><b><sub>0.577 ± 0.179<sub></td>
			<td align="center" valign="center"><b><sub>0.824<sub></td>
			<td align="center" valign="center"><b><sub>0.632 ± 0.107<sub><sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><sub>:heavy_check_mark:<sub></td>	
			<td align="center" valign="center"><b><sub>0.809<sub></td>
			<td align="center" valign="center"><b><sub>0.688 ± 0.062<sub></td>
			<td align="center" valign="center"><b><sub>0.893<sub></td>
			<td align="center" valign="center"><b><sub>0.694 ± 0.125<sub></td>
			<td align="center" valign="center"><b><sub>0.914<sub></td>
			<td align="center" valign="center"><b><sub>0.728 ± 0.096<sub><sub></td>
		</tr>
		<tr>
			<td align="center" valign="center" rowspan="2"><b><sub>Inflammatory<sub></td>
			<td align="center" valign="center"><sub>:x:<sub></td>
			<td align="center" valign="center"><sub>0.673<sub></td>
			<td align="center" valign="center"><sub>0.523 ± 0.079<sub></td>
			<td align="center" valign="center"><sub>0.717<sub></td>
			<td align="center" valign="center"><sub>0.539 ± 0.139<sub></td>
			<td align="center" valign="center"><sub>0.738<sub></td>
			<td align="center" valign="center"><sub>0.607 ± 0.090<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><sub>:heavy_check_mark:<sub></td>	
			<td align="center" valign="center"><sub>0.706<sub></td>
			<td align="center" valign="center"><sub>0.580 ± 0.077<sub></td>
			<td align="center" valign="center"><sub>0.806<sub></td>
			<td align="center" valign="center"><sub>0.641 ± 0.123<sub></td>
			<td align="center" valign="center"><sub>0.796<sub></td>
			<td align="center" valign="center"><sub>0.665 ± 0.081<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center" rowspan="2"><b><sub>Interferon Gamma biology<sub></td>
			<td align="center" valign="center"><sub>:x:<sub></td>
			<td align="center" valign="center"><sub>0.700<sub></td>
			<td align="center" valign="center"><sub>0.541 ± 0.088<sub></td>
			<td align="center" valign="center"><sub>0.672<sub></td>
			<td align="center" valign="center"><sub>0.562 ± 0.117<sub></td>
			<td align="center" valign="center"><sub>0.759<sub></td>
			<td align="center" valign="center"><sub>0.622 ± 0.088<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><sub>:heavy_check_mark:<sub></td>	
			<td align="center" valign="center"><sub>0.783<sub></td>
			<td align="center" valign="center"><sub>0.561 ± 0.119<sub></td>
			<td align="center" valign="center"><sub>0.677<sub></td>
			<td align="center" valign="center"><sub>0.610 ± 0.051<sub></td>
			<td align="center" valign="center"><sub>0.822<sub></td>
			<td align="center" valign="center"><sub>0.674 ± 0.102<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center" rowspan="2"><b><sub>Ribas 10G Inflammatory<sub></td>
			<td align="center" valign="center"><sub>:x:<sub></td>
			<td align="center" valign="center"><sub>0.672<sub></td>
			<td align="center" valign="center"><sub>0.583 ± 0.081<sub></td>
			<td align="center" valign="center"><sub>0.652<sub></td>
			<td align="center" valign="center"><sub>0.552 ± 0.083<sub></td>
			<td align="center" valign="center"><sub>0.758<sub></td>
			<td align="center" valign="center"><sub>0.627 ± 0.082<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><sub>:heavy_check_mark:<sub></td>	
			<td align="center" valign="center"><sub>0.727<sub></td>
			<td align="center" valign="center"><sub>0.640 ± 0.074<sub></td>
			<td align="center" valign="center"><sub>0.726<sub></td>
			<td align="center" valign="center"><sub>0.618 ± 0.065<sub></td>
			<td align="center" valign="center"><sub>0.806<sub></td>
			<td align="center" valign="center"><sub>0.669 ± 0.067<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center" rowspan="2"><b><sub>T cell exhaustion<sub></td>
			<td align="center" valign="center"><sub>:x:<sub></td>
			<td align="center" valign="center"><sub>0.661<sub></td>
			<td align="center" valign="center"><sub>0.490 ± 0.108<sub><sub></td>
			<td align="center" valign="center"><sub>0.744<sub></td>
			<td align="center" valign="center"><sub>0.516 ± 0.123<sub></td>
			<td align="center" valign="center"><sub>0.627<sub></td>
			<td align="center" valign="center"><sub>0.555 ± 0.063<sub></td>
		</tr>
		<tr>
			<td align="center" valign="center"><sub>:heavy_check_mark:<sub></td>	
			<td align="center" valign="center"><sub>0.661<sub></td>
			<td align="center" valign="center"><sub>0.543 ± 0.073<sub><sub></td>
			<td align="center" valign="center"><sub>0.788<sub></td>
			<td align="center" valign="center"><sub>0.606 ± 0.086<sub></td>
			<td align="center" valign="center"><sub>0.788<sub></td>
			<td align="center" valign="center"><sub>0.577 ± 0.092<sub></td>
		</tr>
	</tbody>
</table>

**AUROC (of best-fold model) in the external validation series (Mondor) with tumoral anotations:**

<table  align="center">
	<tbody>
		<tr>
			<td align="center" valign="center"><b><sub>Gene signature (with tumor annot :heavy_check_mark:)<sub></td>
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
***To generate labels for WSIs***
1. Process TCGA FPKM data with **gene_clust/codes/tcga_fpkm_processing.ipynb**
2. Perform hierarchical clustering with **gene_clust/codes/PlotHeatmapGeneSignature.R** (to reproduce the heatmap). Or using Python with **gene_clust/codes/tcga_fpkm_clustering.ipynb** (to get the same clustering results)

*All TCGA data used and clutering results are provided in **gene_clust/data** and **gene_clust/results**. Due to privacy issues, the data in Mondor series is not provided but commands for external validation are described in this [tutorial](https://github.com/qinghezeng/CLAM/blob/master/external_validation.md).*

## Part 2. Deep learning
***To classify WSIs***

Without annotations: [tutorial](https://github.com/qinghezeng/CLAM/blob/master/with_annotations.md)

With annotations: [tutorial](https://github.com/qinghezeng/CLAM/blob/master/without_annotations.md)

Other settings: [tutorial](https://github.com/qinghezeng/CLAM/blob/master/other_settings.md) (color unmixing and saffron removal, color normalization, data augmentation)


***
### Other training settings

#### Color unmixing (3 methods implemented) and remove saffron
##### Macenko PCA method
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139_all.csv --feat_dir results/features-unmix-macenko-pca_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --unmixing --separate_stains_method macenko_pca --file_bgr data/bgr_intensity_mondor.csv --delete_third_stain --convert_to_rgb

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-unmix-macenko-pca_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_unmix --save_exp_code mondor_hcc_tumor-masked_unmix-macenko-pca_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

##### XU SNMF
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139_all.csv --feat_dir results/features-unmix-xu-snmf_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --unmixing --separate_stains_method xu_snmf --file_bgr data/bgr_intensity_mondor.csv --delete_third_stain --convert_to_rgb

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-unmix-xu-snmf_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_unmix --save_exp_code mondor_hcc_tumor-masked_unmix-xu-snmf_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

##### Fixed_hes_vector
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139_all.csv --feat_dir results/features-unmix-fixed-hes-vector_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --unmixing --separate_stains_method fixed_hes_vector --file_bgr data/bgr_intensity_mondor.csv --delete_third_stain --convert_to_rgb

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-unmix-fixed-hes-vector_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_unmix --save_exp_code mondor_hcc_tumor-masked_unmix-fixed-hes-vector_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

#### Color normalization (2 methods implemented)
##### Reinhard
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --csv_path ./dataset_csv/tcga_hcc_feature_354.csv --feat_dir results/features-norm-reinhard_tumor_masked --batch_size 256 --target_patch_size 256 --color_norm --color_norm_method reinhard

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139_all.csv --feat_dir results/features-norm-reinhard_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --color_norm --color_norm_method reinhard

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features-norm-reinhard_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8 > log_Inflammatory_tumor_masked_norm.txt

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features-norm-reinhard_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-norm-reinhard_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code mondor_hcc_tumor-masked_norm-reinhard_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

##### Macenko PCA
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --csv_path ./dataset_csv/tcga_hcc_feature_354.csv --feat_dir results/features-norm-macenko-pca_tumor_masked --batch_size 256 --target_patch_size 256 --color_norm --color_norm_method macenko_pca --file_bgr data/bgr_intensity_tcga.csv --save_images_to_h5 --image_h5_dir results/patches-norm-macenko-pca_tumor_masked

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139_all.csv --feat_dir results/features-norm-macenko-pca_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --color_norm --color_norm_method macenko_pca --file_bgr data/bgr_intensity_mondor.csv --save_images_to_h5 --image_h5_dir results/patches-norm-macenko-pca_mondor_tumor_masked

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features-norm-macenko-pca_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8 > log_Inflammatory_tumor_masked_norm.txt

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features-norm-macenko-pca_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-norm-macenko-pca_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code mondor_hcc_tumor-masked_norm-macenko-pca_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

#### Data augmentation
```shell
CUDA_VISIBLE_DEVICES=0 python data_augm.py --data_dir results/patches_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --result_dir results/patches-augm-8-flips-rots_tumor_masked --csv_path dataset_csv/tcga_hcc_feature_354.csv --target_patch_size 256

CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir results/patches-augm-8-flips-rots_tumor_masked --csv_path ./dataset_csv/tcga_hcc_feature_354.csv --feat_dir results/features-augm-8-flips-rots_tumor_masked --batch_size 256 --model resnet50 --trnsfrms imagenet --train_augm

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features-augm-8-flips-rots_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_augm --exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8 --train_augm > log_Inflammatory_tumor_masked_augm.txt

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_augm --models_exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_augm --save_exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked_augm --models_exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_augm --save_exp_code mondor_hcc_tumor-masked_augm-8-flips-rots_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```







