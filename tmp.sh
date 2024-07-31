python3 create_patches_fp_mag.py \
    --source "/home/heiheiyu127/Documents/PMCC/slides/PKG - CPTAC-LUAD_v12/LUAD/" \
    --save_dir "/home/heiheiyu127/Documents/PMCC/slides/processed/LUAD_mag/" \
    --seg --patch --stitch \
    --expected_mag_size 20 \
    --expected_patch_size 224 \
    --step_size_overlap 0 \
    --process_list "/home/heiheiyu127/Documents/PMCC/slides/processed/LUAD_mag/process_list_mag.csv"