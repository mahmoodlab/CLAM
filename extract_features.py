import time
import os
import argparse
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide

from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag, get_eval_transforms
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--csv_path', type=str)
parser.add_argument('--feat_dir', type=str)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224,
					help='the desired size of patches for scaling before feature embedding')
args = parser.parse_args()

if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	dest_files = os.listdir(args.feat_dir)

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)		
	model = model.to(device)
	_ = model.eval()

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
	
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id + '.h5'
		bag_candidate = os.path.join(args.data_dir, 'patches', bag_name)

		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(bag_name)
		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		file_path = bag_candidate
		time_start = time.time()

		dataset = Whole_Slide_Bag(file_path=file_path, img_transforms=img_transforms)
		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
