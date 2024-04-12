CLAM <img src="clam-logo.png" width="280px" align="right" />
===========
Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images.
*Nature Biomedical Engineering*

[ArXiv](https://arxiv.org/abs/2004.09666) | [Journal Link](https://www.nature.com/articles/s41551-020-00682-w) | [Interactive Demo](http://clam.mahmoodlab.org) | [Cite](#reference) 

***TL;DR:** CLAM is a high-throughput and interpretable method for data efficient whole slide image (WSI) classification using slide-level labels without any ROI extraction or patch-level annotations, and is capable of handling multi-class subtyping problems. Tested on three different WSI datasets, trained models adapt to independent test cohorts of WSI resections and biopsies as well as smartphone microscopy images (photomicrographs).*

[<img src="ani.gif" width="470px" align="left" />](http://clam.mahmoodlab.org)
## CLAM: A Deep-Learning-based Pipeline for Data Efficient and Weakly Supervised Whole-Slide-level Analysis 
[Pre-requisites](#pre-requisites) • [Installation](INSTALLATION.md) • [Segmentation and Patching](#wsi-segmentation-and-patching) • [Feature Extraction](#weakly-supervised-learning-using-slide-level-labels-with-clam) • [Weakly Supervised Training](#Training-Splits) • [Testing](#Testing-and-Evaluation-Script) • [Trained Models](#Trained-Model-Checkpoints) • [Heatmap Visualization](#Heatmap-Visualization) • [Examples](#examples) • [Pre-print](https://arxiv.org/abs/2004.09666) • [Demo](http://clam.mahmoodlab.org) • [Cite](#reference)

***How does CLAM work?** Clustering-constrained Attention Multiple Instance Learning (CLAM) is a deep-learning-based weakly-supervised method that uses attention-based learning to automatically identify sub-regions of high diagnostic value in order to accurately classify the whole slide, while also utilizing instance-level clustering over the representative regions identified to constrain and refine the feature space.*

© [Mahmood Lab](http://www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes. 

## Updates:
* **04/06/2024**: [UNI](https://github.com/mahmoodlab/UNI) and [CONCH](https://github.com/mahmoodlab/CONCH) are now available to select as pretrained encoders. See [Using CONCH / UNI as Pretrained Encoders](#using-conch--uni-as-pretrained-encoders) for more details. Please make sure all dependencies are installed correctly by installing the latest **env.yml** file (see [Installation guide](INSTALLATION.md) for details), and using the corresponding **clam_latest** conda environment.
* 03/19/2024: We are releasing [UNI](https://github.com/mahmoodlab/UNI) and [CONCH](https://github.com/mahmoodlab/CONCH), a pair of SOTA pretrained encoders that produce strong representations for histopathology images and enhance performance on various computational pathology workflows, including the MIL-based CLAM workflow. 
* 05/24/2021: Script for heatmap visualization now available via **create_heatmaps.py**, with the configuration template located in **heatmaps/configs**. See [Heatmap visualization for demo and instructions.](#Heatmap-Visualization)
* 03/01/2021: New, fast patching/feature extraction pipeline is now available. **TL;DR:** since CLAM only requires image features for training, it is not necessary to save the actual image patches, the new pipeline rids of this overhead and instead only saves the coordinates of image patches during "patching" and loads these regions on the fly from WSIs during feature extraction. This is significantly faster than the old pipeline and usually only takes 1-2s for "patching" and a couple minutes to featurize a WSI. To use the new pipeline, make sure you are calling **create_patches_fp.py** and **extract_features_fp.py** instead of the old **create_patches.py** and **extract_features.py** scripts.

**Note**: while we hope that the newest update will require minimal changes to the user's workflow, if needed, you may reference the old version of the code base [here](https://github.com/mahmoodlab/CLAM/tree/deprecated). Please report any issues in the public forum. 

**Warning**: the latest update will by default resize image patches to 224 x 224 before extracting features using the pretrained encoder. This change serves to make it more consistent with the evaluation protocol used in UNI, CONCH and other studies. If you wish to preserve the original size of the image patches generated during patching or use a different image size for feature extraction, you can do so by specifying `--target_patch_size` in **extract_features_fp.py**.

**RE update 03/01/21**: note that the README has been updated to use the new, faster pipeline by default. If you still wish to use the old pipeline, refer to: [Guide for Old Pipeline](README_old.md). It saves tissue patches, which is signficantly slower and takes up a lot of storage space but can still be useful if you need to work with original image patches instead of feature embeddings.

## Installation:
Please refer to our [Installation guide](INSTALLATION.md) for detailed instructions on how to get started.

## WSI Segmentation and Patching 

<img src="CLAM1.jpg" width="1000px" align="center" />
The first step focuses on segmenting the tissue and excluding any holes. The segmentation of specific slides can be adjusted by tuning the individual parameters (e.g. dilated vessels appearing as holes may be important for certain sarcomas.) 
The following example assumes that digitized whole slide image data in well known standard formats (.svs, .ndpi, .tiff etc.) are stored under a folder named DATA_DIRECTORY

```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

### Basic, Fully Automated Run
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 
```

The above command will segment every slide in DATA_DIRECTORY using default parameters, extract all patches within the segemnted tissue regions, create a stitched reconstruction for each slide using its extracted patches (optional) and generate the following folder structure at the specified RESULTS_DIRECTORY:

```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```

The **masks** folder contains the segmentation results (one image per slide).
The **patches** folder contains arrays of extracted tissue patches from each slide (one .h5 file per slide, where each entry corresponds to the coordinates of the top-left corner of a patch)
The **stitches** folder contains downsampled visualizations of stitched tissue patches (one image per slide) (Optional, not used for downstream tasks)
The auto-generated csv file **process_list_autogen.csv** contains a list of all slides processed, along with their segmentation/patching parameters used.

Additional flags that can be passed include:
* `--custom_downsample`: factor for custom downscale (not recommended, ideally should first check if native downsamples exist)
* `--patch_level`: which downsample pyramid level to extract patches from (default is 0, the highest available resolution)
* `--no_auto_skip`: by default, the script will skip over files for which patched .h5 files already exist in the desination folder, this toggle can be used to override this behavior

Some parameter templates are also availble and can be readily deployed as good choices for default parameters:
* `bwh_biopsy.csv`: used for segmenting biopsy slides scanned at BWH (Scanned using Hamamatsu S210 and Aperio GT450) 
* `bwh_resection.csv`: used for segmenting resection slides scanned at BWH
* `tcga.csv`: used for segmenting TCGA slides

Simply pass the name of the template file to the --preset argument, for example, to use the biopsy template:
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch
```
### Custom Default Segmentation Parameters
For advanced usage, in addition to using the default, single set of parameters defined in the script **create_patches_fp.py**, the user can define custom templates of parameters depending on the dataset. These templates are expected to be stored under **presets**, and contain values for each of the parameters used during segmentation and patching. 

The list of segmentation parameters is as follows:
* `seg_level`: downsample level on which to segment the WSI (default: -1, which uses the downsample in the WSI closest to 64x downsample)
* `sthresh`: segmentation threshold (positive integer, default: 8, using a higher threshold leads to less foreground and more background detection)
* `mthresh`: median filter size (positive, odd integer, default: 7)
* `use_otsu`: use otsu's method instead of simple binary thresholding (default: False) 
* `close`: additional morphological closing to apply following initial thresholding (positive integer or -1, default: 4)

The list of contour filtering parameters is as follows:
* `a_t`: area filter threshold for tissue (positive integer, the minimum size of detected foreground contours to consider, relative to a reference patch size of 512 x 512 at level 0, e.g. a value 10 means only detected foreground contours of size greater than 10 512 x 512 sized patches at level 0 will be processed, default: 100)
* `a_h`: area filter threshold for holes (positive integer, the minimum size of detected holes/cavities in foreground contours to avoid, once again relative to 512 x 512 sized patches at level 0, default: 16)
* `max_n_holes`: maximum of holes to consider per detected foreground contours (positive integer, default: 10, higher maximum leads to more accurate patching but increases computational cost)

The list of segmentation visualization parameters is as follows:
* `vis_level`: downsample level to visualize the segmentation results (default: -1, which uses the downsample in the WSI closest to 64x downsample)
* `line_thickness`: line thickness to draw visualize the segmentation results (positive integer, in terms of number of pixels occupied by drawn line at level 0, default: 250)

The list of patching parameters is as follows:
* `use_padding`: whether to pad the border of the slide (default: True)
* `contour_fn`: contour checking function to decide whether a patch should be considered foreground or background (choices between 'four_pt' - checks if all four points in a small, grid around the center of the patch are inside the contour, 'center' - checks if the center of the patch is inside the contour, 'basic' - checks if the top-left corner of the patch is inside the contour, default: 'four_pt')


### Two-Step Run (Mannually Adjust Parameters For Specific Slides)
To ensure that high quality segmentation and extraction of relevant tissue patches, user has the option of first performing segmentation (typically around 1s per slide), inspecting the segmentation results and tweaking the parameters for select slides if necessary and then extracting patches using the tweaked parameters. i.e., first run:

``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg  
```
The above command will segment every slide in DATA_DIRECTORY using default parameters and generate the csv file, but will NOT patch just yet (**patches** and **stitches** folders will be empty)

The csv file can be tweaked for specific slides, and be passed to the script via the --process_list CSV_FILE_NAME such that the script will use the user-updated specifications. Before tweaking the segmentation parameters, the user should make a copy of the csv file and give it a new name (e.g. process_list_edited.csv) because otherwise this file with the default name is overwritten the next time the command is run. Then the user has the option to tweak the parameters for specific slides by changing their corresponding fields in the csv file. The **process** column stores a binary variable (0 or 1) for whether the script should process a specific slide. This allows the user to toggle on just the select few slides to quickly confirm whether the tweaked parameters produce satisfactory results. For example, to re-segment just slide_1.svs again using user-updated parameters, make the appropriate changes to its fields, update its **process** cell to 1, save the csv file, and pass its name to the same command as above:

``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --process_list process_list_edited.csv
```

When satisfied with the segmentation results, the user should make the **process** cell for all slides that need to be processed to 1, save the csv file, and run patching with the saved csv file (just like in the fully-automated run use case, with the additional csv file argument):

``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --process_list CSV_FILE_NAME --patch --stitch
```
## Weakly-Supervised Learning using Slide-Level Labels with CLAM

<img src="CLAM2.jpg" width="1000px" align="center" />

### Feature Extraction (GPU Example)
```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```
The above command expects the coordinates .h5 files to be stored under DIR_TO_COORDS and a batch size of 512 to extract 1024-dim features from each tissue patch for each slide and produce the following folder structure:
```bash
FEATURES_DIRECTORY/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
```
where each .h5 file contains an array of extracted features along with their patch coordinates (note for faster training, a .pt file for each slide is also created for each slide, containing just the patch features). The csv file is expected to contain a list of slide filenames (without the filename extensions) to process (the easiest option is to take the csv file auto generated by the previous segmentation/patching step, and delete the filename extensions)

### Using CONCH / UNI as Pretrained Encoders
If using UNI or CONCH, first refer to their respective HF page below to request and download the model weights (pytorch_model.bin). 

UNI: https://huggingface.co/MahmoodLab/UNI

CONCH: https://huggingface.co/MahmoodLab/CONCH

After successfully downloading the model checkpoints, you need to set the `CONCH_CKPT_PATH` and `UNI_CKPT_PATH` environment variable to the path of the pretrained encoder checkpoints, before running the feature extraction script. For example, if you have downloaded the pretrained UNI and CONCH checkpoints and placed them in the **checkpoints/conch** and **checkpoints/uni** folders respectively, you can set the environment variables as follows:
```bash
export CONCH_CKPT_PATH=checkpoints/conch/pytorch_model.bin
export UNI_CKPT_PATH=checkpoints/uni/pytorch_model.bin
```
When running the **extract_features_fp.py** also set `--model_name` to either 'uni_v1' or 'conch_v1' to use the respective encoder.

Note that these encoder models (especially UNI, which uses ViT-L) are more computationally expensive and require more GPU memory than the default ResNet50 encoder, so expect longer runtimes and reduced batch sizes accordingly if you run out of GPU memory. UNI will produce 1024-dim features, while CONCH will produce 512-dim features.

### Datasets
The data used for training and testing are expected to be organized as follows:
```bash
DATA_ROOT_DIR/
    ├──DATASET_1_DATA_DIR/
        ├── h5_files
                ├── slide_1.h5
                ├── slide_2.h5
                └── ...
        └── pt_files
                ├── slide_1.pt
                ├── slide_2.pt
                └── ...
    ├──DATASET_2_DATA_DIR/
        ├── h5_files
                ├── slide_a.h5
                ├── slide_b.h5
                └── ...
        └── pt_files
                ├── slide_a.pt
                ├── slide_b.pt
                └── ...
    └──DATASET_3_DATA_DIR/
        ├── h5_files
                ├── slide_i.h5
                ├── slide_ii.h5
                └── ...
        └── pt_files
                ├── slide_i.pt
                ├── slide_ii.pt
                └── ...
    └── ...
```
Namely, each dataset is expected to be a subfolder (e.g. DATASET_1_DATA_DIR) under DATA_ROOT_DIR, and the features extracted for each slide in the dataset is stored as a .pt file sitting under the **pt_files** folder of this subfolder.
Datasets are also expected to be prepared in a csv format containing at least 3 columns: **case_id**, **slide_id**, and 1 or more labels columns for the slide-level labels. Each **case_id** is a unique identifier for a patient, while the **slide_id** is a unique identifier for a slide that correspond to the name of an extracted feature .pt file. This is necessary because often one patient has multiple slides, which might also have different labels. When train/val/test splits are created, we also make sure that slides from the same patient do not go to different splits. The slide ids should be consistent with what was used during the feature extraction step. We provide 2 dummy examples of such dataset csv files in the **dataset_csv** folder: one for binary tumor vs. normal classification (task 1) and one for multi-class tumor_subtyping (task 2). 

Dataset objects used for actual training/validation/testing can be constructed using the **Generic_MIL_Dataset** Class (defined in **datasets/dataset_generic.py**). Examples of such dataset objects passed to the models can be found in both **main.py** and **eval.py**. 

For training, look under main.py:
```python 
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_feat_resnet'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            label_col = 'label',
                            ignore=[])
```
The user would need to pass:
* csv_path: the path to the dataset csv file
* data_dir: the path to saved .pt features
* label_dict: a dictionary that maps labels in the label column to numerical values
* label_col: name of the label column (optional, by default it's 'label')
* ignore: labels to ignore (optional, by default it's an empty list)

Finally, the user should add this specific 'task' specified by this dataset object in the --task arguments as shown below:

```python
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
```

### Training Splits
For evaluating the algorithm's performance, multiple folds (e.g. 10-fold) of train/val/test splits can be used. Example 10-fold 80/10/10 splits for the two dummy datasets can be found under the **splits** folder. These splits can be automatically generated using the create_splits_seq.py script with minimal modification just like with **main.py**. For example, tumor_vs_normal splits can be created by calling:
 
``` shell
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k 10
```
The script uses the **Generic_WSI_Classification_Dataset** Class for which the constructor expects the same arguments as 
**Generic_MIL_Dataset** (without the data_dir argument). For details, please refer to the dataset definition in **datasets/dataset_generic.py**

### GPU Training Example for Binary Positive vs. Negative Classification (e.g. Lymph Node Status)
Note: --embed_dim should be set to 512 for CONCH, and 1024 for UNI and resnet50_trunc.

``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir DATA_ROOT_DIR --embed_dim 1024
```

### GPU Training Example for Subtyping Problems (e.g. 3-class RCC Subtyping)
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_2_tumor_subtyping_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping --model_type clam_sb --log_data --subtyping --data_root_dir DATA_ROOT_DIR --embed_dim 1024
``` 
Note: We have included the option to use a single-attention-branch CLAM model, which performs favoribly in most experiments and can be set via --model_type clam_sb (single branch) or clam_mb (multi branch). clam_sb is the default choice. Additionally, the user can adjust the number of patches used for clustering via --B.

By default results will be saved to **results/exp_code** corresponding to the exp_code input argument from the user. If tensorboard logging is enabled (with the arugment toggle --log_data), the user can go into the results folder for the particular experiment, run:
``` shell
tensorboard --logdir=.
```
This should open a browser window and show the logged training/validation statistics in real time. 
For information on each argument, see:
``` shell
python main.py -h
```

### Testing and Evaluation Script
User also has the option of using the evluation script to test the performances of trained models. Examples corresponding to the models trained above are provided below:
``` shell
CUDA_VISIBLE_DEVICES=0 python eval.py --k 10 --models_exp_code task_1_tumor_vs_normal_CLAM_50_s1 --save_exp_code task_1_tumor_vs_normal_CLAM_50_s1_cv --task task_1_tumor_vs_normal --model_type clam_sb --results_dir results --data_root_dir DATA_ROOT_DIR --embed_dim 1024
```

``` shell
CUDA_VISIBLE_DEVICES=0 python eval.py --k 10 --models_exp_code task_2_tumor_subtyping_CLAM_50_s1 --save_exp_code task_2_tumor_subtyping_CLAM_50_s1_cv --task task_2_tumor_subtyping --model_type clam_sb --results_dir results --data_root_dir DATA_ROOT_DIR --embed_dim 1024
```


Once again, for information on each commandline argument, see:
``` shell
python eval.py -h
```

By adding your own custom datasets into **eval.py** the same way as you do for **main.py**, you can also easily test trained models on independent test sets. 

### Heatmap Visualization
Heatmap visualization can be computed in bulk via **create_heatmaps.py** by filling out the config file and storing it in **/heatmaps/configs** and then running **create_heatmaps.py** with the --config NAME_OF_CONFIG_FILE flag. A demo template is included (**config_template.yaml**) for lung subtyping on two WSIs from the CPTAC. 
To run the demo (raw results are saved in **heatmaps/heatmap_raw_results** and final results are saved in **heatmaps/heatmap_production_results**):
``` shell
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config config_template.yaml
```
See **/heatmaps/configs/config_template.yaml** for explanations for each configurable option.

Similar to feature extraction, if using UNI / CONCH, set the environment variables before running the script. See [Using CONCH / UNI as Pretrained Encoders](#using-conch--uni-as-pretrained-encoders) for more details.


### Trained Model Checkpoints
For reproducability, all trained models used can be accessed [here](https://drive.google.com/drive/folders/1NZ82z0U_cexP6zkx1mRk-QeJyKWk4Q7z?usp=sharing).
The 3 main folders (**tcga_kidney_cv**, **tcga_cptac_lung_cv** and **camelyon_40x_cv**) correspond to models for RCC subtyping trained on the TCGA, for NSCLC subtyping trained on TCGA and CPTAC and for Lymph Node Metastasis (Breast) detection trained on Camelyon16+17 respectively. In each main folder, each subfolder corresponds to one set of 10-fold cross-validation experiments. For example, the subfolder tcga_kidney_cv_CLAM_50_s1 contains the 10 checkpoints corresponding to the 10 cross-validation folds for TCGA RCC subtyping, trained using CLAM with multi-attention branches using 50% of cases in the full training set. 

For reproducability, these models can be evaluated on data prepared by following the same pipeline described in the sections above by calling **eval.py** with the appropriate arguments that specify the model options (either --model_type clam_mb or --model_type mil should be set, for evaluation only, --subtyping flag does not make a difference) as well as where the model checkpoints (--results_dir and --models_exp_code) and data (--data_root_dir and --task) are stored.

### Examples

Please refer to our pre-print and [interactive demo](http://clam.mahmoodlab.org) for detailed results on three different problems and adaptability across data sources, imaging devices and tissue content. 

<img src="fig-git-hm.jpg" width="1000px" align="center" />  

Visulize additional examples here: http://clam.mahmoodlab.org

## Issues
- Please report all issues on the public forum.

## License
© [Mahmood Lab](http://www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Funding
This work was funded by NIH NIGMS [R35GM138216](https://reporter.nih.gov/search/sWDcU5IfAUCabqoThQ26GQ/project-details/10029418).

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://www.nature.com/articles/s41551-020-00682-w):

Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

```
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```
