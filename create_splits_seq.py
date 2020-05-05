import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= -1,
                    help='fraction of labels (default: [0.25, 0.5, 0.75, 1.0])')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['camelyon_40x_cv', 'tcga_kidney_cv'])

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



