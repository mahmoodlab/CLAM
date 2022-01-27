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
parser.add_argument('--task', type=str, choices=['tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622',
                                                 'tcga_hcc_349_Gajewski_13G_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Inflammatory_cv_highvsrest_622',
                                                 'tcga_hcc_349_Interferon_Gamma_Biology_cv_highvsrest_622',
                                                 'tcga_hcc_349_Ribas_10G_Interferon_Gamma_cv_highvsrest_622'
                                                 'tcga_hcc_349_T-cell_Exhaustion_cv_highvsrest_622'])

args = parser.parse_args()

if args.task == 'tcga_hcc_349_6G_Interferon_Gamma_cv_highvsrest_622':
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



