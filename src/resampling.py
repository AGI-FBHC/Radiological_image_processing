# -*- coding: utf-8 -*-
# @Time    : 2025/1/12 14:32
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: resampling.py
# @Project : Radiological_image_processing
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def resample_z_direction(
        nii_path: str,
        is_mask: bool = False,
        output_file: str = 'output.nii.gz',
        spacing: float = 1.0,
        is_print: bool = False
        ):
    """Resample the z direction of a nifti file.
    :param nii_path: nii.gz file path.
    :param is_mask: Is file mask or not?
    :param output_file: resample result file path.
    :param spacing: z-spacing (in mm).
    :param is_print: Is print input path?
    :return: None.
    """
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    affine = nii.affine
    z_spacing = np.abs(affine[2, 2])
    zoom_factor = z_spacing / spacing
    scale_factors = [1, 1, zoom_factor]
    new_data = zoom(data, scale_factors, order=0)
    new_data = np.rint(new_data).astype(np.uint8) if is_mask else new_data  # mask 可能存在差值后的精度问题，需要舍入
    new_affine = affine.copy()
    new_affine[2, 2] = np.sign(affine[2, 2]) * spacing
    new_img = nib.Nifti1Image(new_data, affine=new_affine, header=nii.header)
    nib.save(new_img, output_file)
    print(f'resampling {nii_path} completed.') if is_print else None
    pass





