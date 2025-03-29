# -*- coding: utf-8 -*-
# @Time    : 2025/1/12 14:50
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: main.py
# @Project : Radiological_image_processing
import numpy as np
import nibabel as nib
from dcm2nii import *
from resampling import *


# idx = 1
# image_path = f'../sample_data/images/ct_{idx:06d}.nii.gz'
# label_path = f'../sample_data/labels/ct_{idx:06d}.nii.gz'
# resample_z_direction(nii_path=image_path, output_file='./image.nii.gz', is_print=True, spacing=5)
# resample_z_direction(nii_path=label_path, is_mask=True, output_file='./label.nii.gz', is_print=True, spacing=5)


# gz_nii(nii_path='../sample_data/images/ct_000006.nii',
#        output_file='../sample_data/images/ct_00006.nii.gz',
#        is_print=True)
# gz_nii(nii_path='../sample_data/labels/ct_000006.nii',
#        output_file='../sample_data/labels/ct_00006.nii.gz',
#        is_print=True)


