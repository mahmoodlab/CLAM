# internal imports
import cv2

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
import tifffile
import h5py
from openslide import OpenSlide
import slideio


def stitching(file_path, wsi_object, downscale = 64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)


    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, tissue_contours_save_dir,
                  patch_size = 256, step_size = 256,
                  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
                  vis_params = {'vis_level': -1, 'line_thickness': 500},
                  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level = 0,
                  use_default_params = False,
                  seg = False, save_mask = True,
                  stitch= False,
                  patch = False, auto_skip=True, process_list = None):



    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
        'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
        'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
        'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
        'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        # we load the wsi here and pass it, this is so we can test for failure cases
        if '.scn' in slide:
            print('Leica scan found, using SlideIO to load {}'.format(slide))
            slideio_wsi = slideio.open_slide(full_path, driver="SCN")
            scene = slideio_wsi.get_scene(0)
            wsi_img = scene.read_block()
        else:
            print('Using tifffile to load {}'.format(slide))
            wsi_img = tifffile.imread(full_path)

        if wsi_img.shape[0] == 0:
            print('{} was not properly loaded by tifffile. The tiff may be corrupt! Skipping..')
            df.loc[idx, 'status'] = 'failed_to_load'
            continue

        WSI_object = WholeSlideImage(full_path, wsi_img)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}


            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = - 1
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = - 1
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        h, w = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e14:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


        seg_time_elapsed = -1
        if seg:
            # we are going to seg using all of the provided presets, then selecting the segmentation that results in
            # the maximum number of contours with the smallest total area. If we have ties, we pick the first.
            # We also set a hard cutoff for maximum number of contours.

            print('Generating scores for {}, scores close to 0 indicate a strong match, 1 indicates failure.'.format(WSI_object.name))

            contour_cutoff = 10
            optim_list = []
            contours_list = []
            contour_counter = []
            contour_pixel_counter = []
            tumor_contours_list = []
            holes_list = []
            for preset in os.listdir('presets'):
                preset_df = pd.read_csv(os.path.join('presets', preset))

                for key in current_seg_params.keys():
                    current_seg_params[key] = preset_df.loc[0, key]
                    if (key == 'keep_ids') or (key == 'exclude_ids'):
                        if preset_df.loc[0, key] == 'none':
                            current_seg_params[key] = []

                for key in current_filter_params.keys():
                    current_filter_params[key] = preset_df.loc[0, key]

                WSI_object_pass, seg_time_elapsed_pass = segment(WSI_object, current_seg_params, current_filter_params)

                if seg_time_elapsed_pass != -1:
                    if seg_time_elapsed == -1:
                        seg_time_elapsed += seg_time_elapsed_pass + 1
                    else:
                        seg_time_elapsed += seg_time_elapsed_pass

                contours_list.append(WSI_object_pass.contours_tissue.copy())
                holes_list.append(WSI_object_pass.holes_tissue.copy())
                tumor_contours_list.append([])

                if (len(WSI_object_pass.contours_tissue) == 0) or (len(WSI_object_pass.contours_tissue) > contour_cutoff):
                    optim_list.append(1)
                    optim_print = 1
                    contour_counter.append(len(WSI_object_pass.contours_tissue))
                    if (len(WSI_object_pass.contours_tissue) == 0):
                        contour_pixel_counter.append(0)
                    else:
                        contour_pixels = 0
                        for contours in WSI_object_pass.contours_tissue:
                            contour_pixels += cv2.contourArea(contours)
                        contour_pixel_counter.append(contour_pixels)

                else:
                    contour_pixels = 0
                    optim_mult = 1/len(WSI_object_pass.contours_tissue)
                    contour_counter.append(len(WSI_object_pass.contours_tissue))
                    for contours in WSI_object_pass.contours_tissue:
                        contour_pixels += cv2.contourArea(contours)
                    contour_pixel_counter.append(contour_pixels)
                    optim_score = contour_pixels / (WSI_object_pass.level_dim[0][0] * WSI_object_pass.level_dim[0][1])
                    optim_list.append((optim_mult + optim_score) / 2)
                    optim_print = (optim_mult + optim_score) / 2

                print('{} pass completed, optim score: {}'.format(preset, optim_print))

            if np.all(np.array(optim_list) > 0.9):
                print('WARNING: {} appears to have no ideal configuration for tissue seg. '
                      'Check slide quality. Taking seg with most contours...'.format(WSI_object_pass.name))
                # pass the run with the most contours
                winner = np.argmax(np.array(contour_counter))
                df.loc[idx, 'status'] = 'failed_seg'

            else:
                # get the ids of the two smallest scored masks
                idx_small = np.argpartition(np.array(optim_list), 2)
                if (optim_list[idx_small[1]] - optim_list[idx_small[0]]) < 0.05:
                    # the scores are close, we pick the one with the largest total area
                    print('{} and {} are close, we pick the one with the largest total area.'.format(
                    os.listdir('presets')[idx_small[0]],
                    os.listdir('presets')[idx_small[1]]))
                    id0_pixels_score = contour_pixel_counter[idx_small[0]] / (WSI_object_pass.level_dim[0][0] * WSI_object_pass.level_dim[0][1])
                    id1_pixels_score = contour_pixel_counter[idx_small[1]] / (WSI_object_pass.level_dim[0][0] * WSI_object_pass.level_dim[0][1])
                    id0_score = (1 - id0_pixels_score)
                    id1_score = (1 - id1_pixels_score)
                    if id0_score < id1_score:
                        winner = idx_small[0]
                    else:
                        winner = idx_small[1]
                else:
                    winner = np.argmin(np.array(optim_list))
                print('Passes complete, {} provided the best results'.format(
                    os.listdir('presets')[winner]))
                df.loc[idx, 'status'] = 'processed'

            # update the contours
            WSI_object.contours_tissue = contours_list[winner]
            WSI_object.holes_tissue = holes_list[winner]
            WSI_object.contours_tumor = tumor_contours_list[winner]

            # save the contours
            tissue_contours_path = os.path.join(tissue_contours_save_dir, slide_id + '.h5')
            hf = h5py.File(tissue_contours_path, 'w')
            if len(contours_list[winner]) == 0:
                hf.create_dataset('tissue_contours_null', data=np.array([]))
            else:
                for c_id, contour_sample in enumerate(contours_list[winner]):
                    hf.create_dataset('tissue_contours_{}'.format(c_id), data=contour_sample)
            if len(holes_list[winner]) == 0:
                hf.create_dataset('holes_contours_null', data=np.array([]))
            else:
                for h_id, holes_sample in enumerate(holes_list[winner]):
                    # weird things going on with holes, inconsistent data structure
                    if holes_sample == []:
                        hf.create_dataset('holes_contours_null_{}'.format(h_id), data=np.array(holes_sample))
                    elif len(holes_sample) == 1:
                        hf.create_dataset('holes_contours_{}'.format(h_id), data=holes_sample[0])
                    else:
                        for hsub_id, holes_sub_sample in enumerate(holes_sample):
                            hf.create_dataset('holes_contours_{}_{}'.format(h_id, hsub_id), data=holes_sub_sample)
            hf.close()

            # update params
            preset_df = pd.read_csv(os.path.join('presets', os.listdir('presets')[winner]))

            for key in current_seg_params.keys():
                df.loc[idx, key] = preset_df.loc[0, key]

            for key in current_filter_params.keys():
                df.loc[idx, key] = preset_df.loc[0, key]

            for key in current_vis_params.keys():
                current_vis_params[key] = preset_df.loc[0, key]
                df.loc[idx, key] = preset_df.loc[0, key]

            for key in current_patch_params.keys():
                current_patch_params[key] = preset_df.loc[0, key]
                df.loc[idx, key] = preset_df.loc[0, key]

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1 # Default time
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                         'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type=str, default='/data/public/HULA/WSIs',
                    help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type=int, default=1024,
                    help='step_size')
parser.add_argument('--patch_size', type=int, default=4096,
                    help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type=str, default='/data/public/HULA/WSIs_tissue_masks_CLAM',
                    help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
                    help='downsample level at which to patch')
parser.add_argument('--process_list',  type=str, default=None,
                    help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')
    tissue_contours_save_dir = os.path.join(args.save_dir, 'tissue_contours')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)
    print('tissue_contours_save_dir: ', tissue_contours_save_dir)

    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir' : mask_save_dir,
                   'stitch_save_dir': stitch_save_dir,
                   'tissue_contours_save_dir': tissue_contours_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 11, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                            patch_size = args.patch_size, step_size=args.step_size,
                                            seg = args.seg,  use_default_params=False, save_mask = True,
                                            stitch= args.stitch,
                                            patch_level=args.patch_level, patch = args.patch,
                                            process_list = process_list, auto_skip=args.no_auto_skip)
