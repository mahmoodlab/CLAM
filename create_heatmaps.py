from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models import get_encoder
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from tqdm import tqdm

"""
热图推理脚本
--------------------------------
1) 读取一份 heatmap 配置（YAML），解析出数据、模型、patching 以及可视化等参数；
2) 逐张处理指定的 WSI：分割/过滤 -> 建立可视化 mask -> 以给定步长提取 block/patch；
3) 用给定的 encoder 提取 patch 特征（写入 .h5），随后转存为 .pt；
4) 调用训练好的 CLAM 模型对单张 WSI 的 bag 级特征进行推断，得到类别概率与注意力 A；
5) 将注意力 A 与 patch 坐标写入 *_blockmap.h5；
6) 可选：根据注意力分数采样 ROI patch 图；
7) 生成并保存注意力热图，以及原始缩略图（可选）；
8) 持续把中间统计结果写入 CSV；
"""
# -----------------------------
# 命令行参数：只接受少量覆盖
# -----------------------------
parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code') #覆盖 YAML 中 exp_arguments.save_exp_code
parser.add_argument('--overlap', type=float, default=None) #覆盖 YAML 中 patching_arguments.overlap
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml") # YAML 配置文件路径
args = parser.parse_args()
# 设备：优先用 GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 针对单张 WSI（已提取 bag 特征）的推断函数
# -----------------------------
def infer_single_slide(model, features, label, reverse_label_dict, k=1):
    """
    输入：
    - model: 训练好的 CLAM 模型
    - features: [N, D] 的 bag 级特征张量（来自 h5/pt）
    - label: 该 slide 的标注（可能是数值编码或字符串）
    - reverse_label_dict: {class_id -> class_name}
    - k: 取 Top-K 预测（通常等于类别数）
    输出：
    - ids: Top-K 的类别 id（numpy 数组）
    - preds_str: Top-K 的类别名（numpy 数组）
    - probs: Top-K 的概率（对应 ids 顺序）
    - A: 注意力分数（[N, 1] numpy）
    """    
    features = features.to(device)
    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            # 对 CLAM：前向会返回 logits, Y_prob, Y_hat, A, ...
            model_results_dict = model(features)
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item() # 预测类别 id

            # 对多分支 CLAM_MB，注意力 A 的第 0 维是类别分支，需要取当前预测类别对应的注意力图
            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
        
        # 取 Top-K 标签与概率
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

# -----------------------------
# 将 DataFrame 行中覆盖参数写回到 params 字典（带类型转换）
# -----------------------------
def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params

# -----------------------------
# 根据命令行覆盖 YAML 配置
# -----------------------------
def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

# -----------------------------
# 主流程
# -----------------------------
#1. 读取 YAML 配置 → 2. 展开 argparse.Namespace → 3. 枚举切片 → 4. 特征提取 →
#5. CLAM 推断 → 6. 注意力落盘 → 7. 采样 patch → 8. 多版本热图 → 9. 结果 CSV
if __name__ == '__main__':
    # 1) 读取并打印配置
    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n'+key)
            for value_key, value_value in value.items():
                print (value_key + " : " + str(value_value))
        else:
            print ('\n'+key + " : " + str(value))
    
    # 简单的人机确认，避免误操作
    decision = input('Continue? Y/N ')
    if decision in ['Y', 'y', 'Yes', 'yes']:
        pass
    elif decision in ['N', 'n', 'No', 'NO']:
        exit()
    else:
        raise NotImplementedError

    # 2) 将 YAML 的分组参数展开为 argparse.Namespace，便于以点号访问
    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    encoder_args = args['encoder_arguments']
    encoder_args = argparse.Namespace(**encoder_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])
    
    # 3) 计算 patch 尺寸与步长（step = patch_size * (1 - overlap)）
    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    # 一组默认的分割/过滤/可视化/patch 参数，后续会被 preset/CSV 覆盖
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    # 如果提供了 preset CSV，则从其中的第 1 行覆盖默认参数
    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    # 4) 构建待处理列表 process_stack（DataFrame）
    if data_args.process_list is None:
        # 从 data_dir 目录枚举 slide 文件
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            # 从 CSV 指定的 process_list 读入
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
        
    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))
    
    # 5) 初始化模型与 encoder
    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))
    
    if model_args.initiate_fn == 'initiate_model':
        model =  initiate_model(model_args, ckpt_path) # 返回已加载权重的 CLAM 模型（eval 模式通常在内部设定）
    else:
        raise NotImplementedError

    feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    _ = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    print('Done!')

    # label 编码映射（YAML 中 data_arguments.label_dict）
    label_dict =  data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
    
    # 准备输出目录
    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    # block 模式提特征时用到的关键参数（注意这里 step_size=patch_size，意味着无 overlap 的 block map）
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
    'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

    # 6) 主循环：逐张 slide 处理
    for i in tqdm(range(len(process_stack))):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name+=data_args.slide_ext
        print('\nprocessing: ', slide_name)	

        # slide 的标签（可能不存在）
        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')

        # grouping 用于生成保存路径的上级目录（如果 label 是数值，则转回类名）
        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        # 生产级与原始级保存目录
        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)
        # ROI 框（可选）
        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)
        # 构建 slide 路径：既支持 data_dir 是字符串，也支持是一个 dict（多来源目录）
        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters# 从默认参数拷贝一份，随后用 CSV 行覆盖
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        # keep_ids / exclude_ids 从字符串解析为 int 数组
        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        # 打印参数以便复现
        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        # 初始化 WSI 对象：内部会做分割与过滤
        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        # 针对可视化层级计算真正的像素级 patch 尺寸（考虑 level_downsample 与 custom_downsample）
        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        # 选择可视化层级：若 <0 则自动找一个合适的 level
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
    

        ##### check if h5_features_file exists ####### 7) 若未提取过特征，则计算并写入 h5
        if not os.path.isfile(h5_path) :
            _, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
                                            model=model, 
                                            feature_extractor=feature_extractor, 
                                            img_transforms=img_transforms,
                                            batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
                                            attn_save_path=None, feat_save_path=h5_path, 
                                            ref_scores=None)				
        
        ##### check if pt_features_file exists ####### 8) 将 h5 的 features 转存为 .pt，便于后续 torch.load
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            file.close()

        # load features # 载入特征，记录 bag 大小
        features = torch.load(features_path)
        process_stack.loc[i, 'bag_size'] = len(features)
        
        # 保存分割 mask（pkl）
        wsi_object.saveSegmentation(mask_file)
        # 9) 调用 CLAM 推断，得到 Top-K、概率与注意力 A
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
        del features
        
        # 10) 写入 blockmap（注意力 + 坐标），便于可视化
        if not os.path.isfile(block_map_save_path): 
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        # save top 3 predictions # 将 Top-K 结果写回 process_stack（Pred_c, p_c）
        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        # 持续写结果 CSV：若使用了 process_list，就复用其名字；否则用 save_exp_code
        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
        else:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
        # 读回 blockmap 以便后续采样/绘图
        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        # 11) ROI 采样（可选）：根据不同分数段、模式采样若干 patch，并保存 PNG
        samples = sample_args.samples
        for sample in samples:
            if sample['sample']:
                tag = "label_{}_pred_{}".format(label, Y_hats[0])
                sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

        # 12) 绘制热图# 12) 生成整体热图（block 层面，无 ROI 限制）
        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
        'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            pass
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
        
            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        # 13) 针对 ROI/blur/percentile 等设定，计算并保存细粒度热图（可多版本）
        save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None
        
        # 14) 可选：保存原图缩略图（按 vis_level、custom_downsample）
        if heatmap_args.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, 
                                img_transforms=img_transforms,
                                clam_pred=Y_hats[0], model=model, 
                                feature_extractor=feature_extractor, 
                                batch_size=exp_args.batch_size, **wsi_kwargs, 
                                attn_save_path=save_path,  ref_scores=ref_scores)

        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue
        
        with h5py.File(save_path, 'r') as file:
            file = h5py.File(save_path, 'r')
            dset = file['attention_scores']
            coord_dset = file['coords']
            scores = dset[:]
            coords = coord_dset[:]

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
                                                                                        int(heatmap_args.blur), 
                                                                                        int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
                                                                                        float(heatmap_args.alpha), int(heatmap_args.vis_level), 
                                                                                        int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass
        
        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
                                  cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
                                  binarize=heatmap_args.binarize, 
                                    blank_canvas=heatmap_args.blank_canvas,
                                    thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
                                    overlap=patch_args.overlap, 
                                    top_left=top_left, bot_right = bot_right)
            if heatmap_args.save_ext == 'jpg':
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
        
        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                if heatmap_args.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
    # 15) 将最终使用过的配置完整落盘（便于复现）
    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)


