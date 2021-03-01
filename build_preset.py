import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='preset_builder')
parser.add_argument('--preset_name', type=str,
					help='name of preset')
parser.add_argument('--seg_level', type=int, default=-1, 
					help='downsample level at which to segment')
parser.add_argument('--sthresh', type=int, default=8, 
					help='segmentation threshold')
parser.add_argument('--mthresh', type=int, default=7, 
					help='median filter threshold')
parser.add_argument('--use_otsu', action='store_true', default=False)
parser.add_argument('--close', type=int, default=4, 
					help='additional morphological closing')
parser.add_argument('--a_t', type=int, default=100, 
					help='area filter for tissue')
parser.add_argument('--a_h', type=int, default=16, 
					help='area filter for holes')
parser.add_argument('--max_n_holes', type=int, default=8, 
					help='maximum number of holes to consider for each tissue contour')
parser.add_argument('--vis_level', type=int, default=-1, 
					help='downsample level at which to visualize')
parser.add_argument('--line_thickness', type=int, default=250, 
					help='line_thickness to visualize segmentation')
parser.add_argument('--white_thresh', type=int, default=5, 
					help='saturation threshold for whether to consider a patch as blank for exclusion')
parser.add_argument('--black_thresh', type=int, default=50, 
					help='mean rgb threshold for whether to consider a patch as black for exclusion')
parser.add_argument('--no_padding', action='store_false', default=True)
parser.add_argument('--contour_fn', type=str, choices=['four_pt', 'center', 'basic', 'four_pt_hard'], default='four_pt',
					help='contour checking function')


if __name__ == '__main__':
	args = parser.parse_args()
	seg_params = {'seg_level': args.seg_level, 'sthresh': args.sthresh, 'mthresh': args.mthresh, 
				  'close': args.close, 'use_otsu': args.use_otsu, 'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':args.a_t, 'a_h': args.a_h, 'max_n_holes': args.max_n_holes}
	vis_params = {'vis_level': args.vis_level, 'line_thickness': args.line_thickness}
	patch_params = {'white_thresh': args.white_thresh, 'black_thresh': args.black_thresh, 
					'use_padding': args.no_padding, 'contour_fn': args.contour_fn}

	all_params = {}
	all_params.update(seg_params)
	all_params.update(filter_params)
	all_params.update(vis_params)
	all_params.update(patch_params)
	params_df = pd.DataFrame(all_params, index=[0])
	params_df.to_csv('presets/{}'.format(args.preset_name), index=False)
	