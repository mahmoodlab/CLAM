Tutorial for Saffron Removing / Color Normalization / Data Augmentation
===========

*Only commands for fp and with annotations are tested and presented as following.*

### Color unmixing/deconvolution (3 methods implemented) and saffron removing during external validation
#### Macenko PCA method
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features-unmix-macenko-pca_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --unmixing --separate_stains_method macenko_pca --file_bgr data/bgr_intensity_mondor.csv --delete_third_stain --convert_to_rgb

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-unmix-macenko-pca_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_unmix --save_exp_code mondor_hcc_tumor-masked_unmix-macenko-pca_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

#### XU SNMF
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features-unmix-xu-snmf_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --unmixing --separate_stains_method xu_snmf --file_bgr data/bgr_intensity_mondor.csv --delete_third_stain --convert_to_rgb

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-unmix-xu-snmf_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_unmix --save_exp_code mondor_hcc_tumor-masked_unmix-xu-snmf_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

#### Fixed_hes_vector
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features-unmix-fixed-hes-vector_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --unmixing --separate_stains_method fixed_hes_vector --file_bgr data/bgr_intensity_mondor.csv --delete_third_stain --convert_to_rgb

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-unmix-fixed-hes-vector_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked --models_exp_code tcga_hcc_tumor-masked_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_unmix --save_exp_code mondor_hcc_tumor-masked_unmix-fixed-hes-vector_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

### Color normalization (2 methods implemented)
#### Reinhard
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir results/features-norm-reinhard_tumor_masked --batch_size 256 --target_patch_size 256 --color_norm --color_norm_method reinhard

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features-norm-reinhard_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --color_norm --color_norm_method reinhard

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features-norm-reinhard_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8 > log_Inflammatory_tumor_masked_norm.txt

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features-norm-reinhard_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-norm-reinhard_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-reinhard_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code mondor_hcc_tumor-masked_norm-reinhard_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

#### Macenko PCA
```shell
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir results/features-norm-macenko-pca_tumor_masked --batch_size 256 --target_patch_size 256 --color_norm --color_norm_method macenko_pca --file_bgr data/bgr_intensity_tcga.csv --save_images_to_h5 --image_h5_dir results/patches-norm-macenko-pca_tumor_masked

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_dir results/patches-fp_mondor_tumor_masked --data_slide_dir PATH_TO_MONDOR_WSI --csv_path ./dataset_csv/mondor_hcc_feature_139.csv --feat_dir results/features-norm-macenko-pca_mondor_tumor_masked --target_patch_size 256 --batch_size 128 --color_norm --color_norm_method macenko_pca --file_bgr data/bgr_intensity_mondor.csv --save_images_to_h5 --image_h5_dir results/patches-norm-macenko-pca_mondor_tumor_masked

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features-norm-macenko-pca_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8 > log_Inflammatory_tumor_masked_norm.txt

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features-norm-macenko-pca_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features-norm-macenko-pca_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked_norm --models_exp_code tcga_hcc_tumor-masked_norm-macenko-pca_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_norm --save_exp_code mondor_hcc_tumor-masked_norm-macenko-pca_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```

### Data augmentation
```shell
CUDA_VISIBLE_DEVICES=0 python data_augm.py --data_dir results/patches-fp_tumor_masked --data_slide_dir PATH_TO_TCGA_WSI --result_dir results/patches-augm-8-flips-rots_tumor_masked --csv_path dataset_csv/tcga_hcc_feature_349.csv --target_patch_size 256

CUDA_VISIBLE_DEVICES=0 python extract_features.py --data_dir results/patches-augm-8-flips-rots_tumor_masked --csv_path ./dataset_csv/tcga_hcc_feature_349.csv --feat_dir results/features-augm-8-flips-rots_tumor_masked --batch_size 256 --model resnet50 --trnsfrms imagenet --train_augm

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out --early_stopping --lr 2e-4 --k 10 --label_frac 1 --data_dir results/features-augm-8-flips-rots_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_augm --exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small --log_data --B 8 --train_augm > log_Inflammatory_tumor_masked_augm.txt

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --data_dir results/features_tumor_masked --results_dir ./results/training_gene_signatures_tumor_masked_augm --models_exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_augm --save_exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1_cv --task tcga_hcc_349_Inflammatory_cv_highvsrest_622 --model_type clam_sb --model_size small

CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out --k 10 --k_start 0 --k_end 10 --data_dir results/features_mondor_tumor_masked --splits_dir ./splits/mondor_hcc_139_Inflammatory_cv_highvsrest_00X_100 --results_dir ./results/training_gene_signatures_tumor_masked_augm --models_exp_code tcga_hcc_tumor-masked_augm-8-flips-rots_349_Inflammatory_cv_highvsrest_622_CLAM_50_s1 --eval_dir ./eval_results_349_tumor_masked_augm --save_exp_code mondor_hcc_tumor-masked_augm-8-flips-rots_139_Inflammatory_cv_highvsrest_00X_CLAM_50_s1_cv --task mondor_hcc_139_Inflammatory_cv_highvsrest_00X --model_type clam_sb --model_size small
```
