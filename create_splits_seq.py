import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= -1,
                    help='fraction of labels (default: [0.25, 0.5, 0.75, 1.0])')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['camelyon_40x_cv', 'tcga_kidney_cv', 'tcga_hcc_test_cv_c2_622',
                                                 'tcga_hcc_177_cv_c2_811', 'tcga_hcc_177_cv_c3_811', 
                                                 'tcga_hcc_177_cv_c2_622', 'tcga_hcc_177_cv_c3_622', 
                                                 'tcga_hcc_354_v1_cv_c2_811','tcga_hcc_354_v1_cv_c2_622', 
                                                 'tcga_hcc_349_v1_cv_c2_811','tcga_hcc_349_v1_cv_c2_622',
                                                 'tcga_hcc_354_v2_cv_c2_811','tcga_hcc_349_v2_cv_c2_622', 
                                                 'tcga_hcc_349_v2_cv_c3_622', 'tcga_hcc_349_v3_cv_c2_622',
                                                 'tcga_hcc_349_v4_cv_c3_core_811', 'tcga_hcc_349_v4_cv_c2_core_811',
                                                 'tcga_hcc_349_v1_cv_highvsrest_622', 'tcga_hcc_349_v1_cv_highvsrest_core_622',
                                                 'tcga_hcc_349_Inflammatory_cv_lowvsrest_core_622', 
                                                 'tcga_hcc_349_Inflammatory_cv_lowvsrest_622',
                                                 'tcga_hcc_349_Inflammatory_cv_highvsrest_core_622', 
                                                 'tcga_hcc_349_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_core_622',
                                                 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_core_622',
                                                 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_core_622',
                                                 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622',
                                                 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_core_622',
                                                 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_core_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_lowvsrest_core_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_lowvsrest_622',
                                                 
                                                 'tcga_hcc_354_Inflammatory_cv_lowvsrest_622',
                                                 'tcga_hcc_354_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_354_Gajewski_13G_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_354_6G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_354_Interferon_Gamma_Biology_cv_highvsrest_622',
                                                 'tcga_hcc_354_T-cell_Exhaustion_cv_highvsrest_622',
                                                 'tcga_hcc_354_Ribas_10G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_354_Ribas_10G_Interferon_Gamma_cv_lowvsrest_622',
                                                 
                                                 'mondor_hcc_258_v1_cv_highvsrest_00X', 'mondor_hcc_258_v1_cv_highvsrest_core_00X',
                                                 'mondor_hcc_258_Inflammatory_cv_lowvsrest_core_00X', 
                                                 'mondor_hcc_258_Inflammatory_cv_lowvsrest_00X',
                                                 'mondor_hcc_258_Inflammatory_cv_highvsrest_core_00X', 
                                                 'mondor_hcc_258_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_139-asc-first_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_258_Gajewski_13G_Inflammatory_cv_highvsrest_core_00X',
                                                 'mondor_hcc_258_Gajewski_13G_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Gajewski_13G_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_139-asc-first_Gajewski_13G_Inflammatory_cv_highvsrest_00X',
                                                 'mondor_hcc_258_6G_Interferon_Gamma_cv_highvsrest_core_00X',
                                                 'mondor_hcc_258_6G_Interferon_Gamma_cv_highvsrest_00X',
                                                 'mondor_hcc_139_6G_Interferon_Gamma_cv_highvsrest_00X',
                                                 'mondor_hcc_139-asc-first_6G_Interferon_Gamma_cv_highvsrest_00X',
                                                 'mondor_hcc_258_Interferon_Gamma_Biology_cv_highvsrest_core_00X',
                                                 'mondor_hcc_258_Interferon_Gamma_Biology_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Interferon_Gamma_Biology_cv_highvsrest_00X',
                                                 'mondor_hcc_139-asc-first_Interferon_Gamma_Biology_cv_highvsrest_00X',
                                                 'mondor_hcc_258_T-cell_Exhaustion_cv_highvsrest_core_00X',
                                                 'mondor_hcc_258_T-cell_Exhaustion_cv_highvsrest_00X',
                                                 'mondor_hcc_139_T-cell_Exhaustion_cv_highvsrest_00X',
                                                 'mondor_hcc_139-asc-first_T-cell_Exhaustion_cv_highvsrest_00X',
                                                 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_highvsrest_core_00X',
                                                 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X',
                                                 'mondor_hcc_139-asc-first_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X',
                                                 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_lowvsrest_core_00X',
                                                 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_lowvsrest_00X',
                                                 
                                                 'mondor-biopsy_hcc_cv_random_00X',
                                                 
                                                 'tcga_hcc_349_10G_preliminary_IFN-γ_cv_highvsrest_622',
                                                 'tcga_hcc_349_Expanded_immune_gene_cv_highvsrest_622',
                                                 'mondor_hcc_258_10G_preliminary_IFN-γ_cv_highvsrest_00X',
                                                 'mondor_hcc_139_10G_preliminary_IFN-γ_cv_highvsrest_00X',
                                                 'mondor_hcc_258_Expanded_immune_gene_cv_highvsrest_00X',
                                                 'mondor_hcc_139_Expanded_immune_gene_cv_highvsrest_00X',
                                                 
                                                 'tcga_colorectal_1_cv_00X'])

args = parser.parse_args()

if args.task == 'tcga_kidney':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
                            label_col = 'label',
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=['TCGA-SARC'])

    val_num = (10, 48, 26)
    test_num = (10, 48, 26)


elif args.task == 'camelyon_40x_cv':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/camelyon_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat= True,
                            ignore=[])

    val_num = (31, 19)
    test_num = (31, 19)


elif args.task == 'tcga_hcc_test_cv_c2_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/test_dataset_2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)    
    
elif args.task == 'tcga_hcc_177_cv_c2_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_177_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_177_cv_c3_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_177_c3.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster Median':1, 'Cluster High':2},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_177_cv_c2_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_177_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_177_cv_c3_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_177_c3.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster Median':1, 'Cluster High':2},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_v1_cv_c2_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_v1_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_354_v1_cv_c2_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_v1_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_v1_cv_c2_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v1_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_349_v1_cv_c2_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v1_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_v2_cv_c2_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_v2_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_349_v2_cv_c2_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v2_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_v2_cv_c3_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v2_c3.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster Median':1, 'Cluster High':2},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_v3_cv_c2_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v3_c2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_v1_cv_c2_core_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v1_c2_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_349_v3_cv_c2_core_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v3_c2_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_349_v4_cv_c3_core_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v4_c3_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster Median':1, 'Cluster High':2},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_349_v4_cv_c2_core_811':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v4_c2_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.1).astype(int)
    test_num = np.floor(num_slides_cls * 0.1).astype(int)
    
elif args.task == 'tcga_hcc_349_v1_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v1_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_v1_cv_highvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_v1_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Inflammatory_cv_lowvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Inflammatory_lowvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Inflammatory_cv_lowvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Inflammatory_lowvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_Inflammatory_cv_lowvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_Inflammatory_lowvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Inflammatory_cv_highvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Inflammatory_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_Inflammatory_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Inflammatory_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Gajewski_13G_Inflammatory_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Gajewski_13G_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_Gajewski_13G_Inflammatory_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_Gajewski_13G_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_6G_Interferon_Gamma_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_6G_Interferon_Gamma_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_6G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_6G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)

elif args.task == 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Interferon_Gamma_Biology_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Interferon_Gamma_Biology_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_Interferon_Gamma_Biology_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_Interferon_Gamma_Biology_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_T-cell_Exhaustion_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_T-cell_Exhaustion_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_T-cell_Exhaustion_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_T-cell_Exhaustion_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)

    
elif args.task == 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Ribas_10G_Interferon_Gamma_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_Ribas_10G_Interferon_Gamma_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_lowvsrest_core_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Ribas_10G_Interferon_Gamma_lowvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_lowvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Ribas_10G_Interferon_Gamma_lowvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_354_Ribas_10G_Interferon_Gamma_cv_lowvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_354_Ribas_10G_Interferon_Gamma_lowvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
### Mondor ##########################################################################################################
    
elif args.task == 'mondor_hcc_258_v1_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_v1_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_v1_cv_highvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_v1_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Inflammatory_cv_lowvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Inflammatory_lowvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Inflammatory_cv_lowvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Inflammatory_lowvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Inflammatory_cv_highvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Inflammatory_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139_Inflammatory_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139-asc-first_Inflammatory_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139-asc-first_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Inflammatory_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Gajewski_13G_Inflammatory_cv_highvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Gajewski_13G_Inflammatory_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Gajewski_13G_Inflammatory_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Gajewski_13G_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139_Gajewski_13G_Inflammatory_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Gajewski_13G_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139-asc-first_Gajewski_13G_Inflammatory_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139-asc-first_Gajewski_13G_Inflammatory_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_6G_Interferon_Gamma_cv_highvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_6G_Interferon_Gamma_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_6G_Interferon_Gamma_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_6G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139_6G_Interferon_Gamma_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_6G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139-asc-first_6G_Interferon_Gamma_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139-asc-first_6G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls

elif args.task == 'mondor_hcc_258_Interferon_Gamma_Biology_cv_highvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Interferon_Gamma_Biology_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Interferon_Gamma_Biology_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Interferon_Gamma_Biology_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139_Interferon_Gamma_Biology_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Interferon_Gamma_Biology_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139-asc-first_Interferon_Gamma_Biology_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139-asc-first_Interferon_Gamma_Biology_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_T-cell_Exhaustion_cv_highvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_T-cell_Exhaustion_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_T-cell_Exhaustion_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_T-cell_Exhaustion_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
    
elif args.task == 'mondor_hcc_139_T-cell_Exhaustion_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_T-cell_Exhaustion_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139-asc-first_T-cell_Exhaustion_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139-asc-first_T-cell_Exhaustion_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls

    
elif args.task == 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_highvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Ribas_10G_Interferon_Gamma_highvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139-asc-first_Ribas_10G_Interferon_Gamma_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139-asc-first_Ribas_10G_Interferon_Gamma_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_lowvsrest_core_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Ribas_10G_Interferon_Gamma_lowvsrest_core.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Ribas_10G_Interferon_Gamma_cv_lowvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Ribas_10G_Interferon_Gamma_lowvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Low':0, 'Cluster High + Median':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
 
### Test biopsy ############################################################################################
elif args.task == 'mondor-biopsy_hcc_cv_random_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor-biopsy_hcc.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls    
    
### Melanoma gene signatures ############################################################################################
    
elif args.task == 'tcga_hcc_349_10G_preliminary_IFN-γ_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_10G_preliminary_IFN-γ_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'tcga_hcc_349_Expanded_immune_gene_cv_highvsrest_622':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_Expanded_immune_gene_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.floor(num_slides_cls * 0.2).astype(int)
    test_num = np.floor(num_slides_cls * 0.2).astype(int)
    
elif args.task == 'mondor_hcc_258_10G_preliminary_IFN-γ_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_10G_preliminary_IFN-γ_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139_10G_preliminary_IFN-γ_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_10G_preliminary_IFN-γ_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_258_Expanded_immune_gene_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_258_Expanded_immune_gene_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
elif args.task == 'mondor_hcc_139_Expanded_immune_gene_cv_highvsrest_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/mondor_hcc_139_Expanded_immune_gene_highvsrest.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
    
### Colorectal ##########################################################################################################
    
elif args.task == 'tcga_colorectal_1_cv_00X':
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tcga_colorectal.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Cluster Median + Low':0, 'Cluster High':1},
                            label_col = 'cluster',
                            patient_strat= True,
                            ignore=[])
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = [0, 0]
    test_num = num_slides_cls
	
    
    
else:
    raise NotImplementedError

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



