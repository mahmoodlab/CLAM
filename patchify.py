import os
from PIL import Image
from itertools import product


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

def main():
	patch_size = 256
	input_dir = '/home/ashrith/Documents/DL-Fungal/Repos/CLAM_model/CLAM_1/image_sets/original'
	output_dir = '/home/ashrith/Documents/DL-Fungal/Repos/CLAM_model/CLAM_1/image_sets/patches'

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)
	
	for filename in os.listdir(input_dir):
		name, ext = os.path.splitext(filename)
		output_patches_dir = os.path.join(output_dir, name)
		
		if not os.path.isdir(output_patches_dir):
			os.mkdir(output_patches_dir)

		tile(filename, input_dir, output_patches_dir, patch_size)


main()
