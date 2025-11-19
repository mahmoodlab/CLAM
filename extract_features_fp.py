# ========== 1. 基础库与配置 ==========
import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5 # 自定义工具：把 dict 写入 h5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP # 自定义 Dataset：读取 h5 中的 patch 坐标与图像
from models import get_encoder # 自定义模型：根据名字拿到预训练编码器

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ========== 2. 核心函数：用 DataLoader 批量推理并保存特征 ==========
def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
    逐 batch 推理，把 {features, coords} 追加写入同一个 h5。
    第一次写用 mode='w'，后续全部 mode='a'（追加）。
    """
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

# ========== 3. 命令行参数 ==========
parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()

# ========== 4. 主流程 ==========
if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError
	# 4.1 读取 CSV → 得到所有待处理 WSI 的 slide_id 列表
	bags_dataset = Dataset_All_Bags(csv_path)
	
	# 4.2 创建三级输出目录
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True) # 最终特征
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True) # 中间临时
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files')) # 已完成的 *.pt

	# 4.3 初始化 CNN 模型 + 预处理流水线
	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval() # 切换到推理模式
	model = model.to(device) # 搬到 GPU
	total = len(bags_dataset) # 总共要跑多少张 WSI

	# 4.4 设置 DataLoader 的加速参数
	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	# 4.5 大循环：逐张 WSI 提取特征
	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		 # 4.6 自动跳过：如果 *.pt 已存在且用户没强制重跑
		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 
		# 4.7 打开 WSI → 构造 Dataset → 构造 DataLoader
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		# 4.8 真正推理：把整张片子的 patch 特征写进 h5
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		# 4.9 简单日志：耗时、shape 检查
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
		# 4.10 h5 → Tensor → .pt，训练阶段直接 torch.load 即可
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



