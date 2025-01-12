# -*- coding: utf-8 -*-
# @Time    : 2024/12/13 15:54
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: dcm2nii.py
# @Project : Radiological_image_processing
import nibabel as nib
import SimpleITK as sitk


def dicom_to_nifti(dicom_dir, output_file='output.nii.gz'):
    """
    将一个目录下的 DICOM 文件合成为一个 NIfTI 文件。
    :param dicom_dir: DICOM 文件所在目录
    :param output_file: 输出的 NIfTI 文件路径，例如 "output.nii.gz"
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    flipped_image = sitk.Flip(image, [False, False, True])  # 翻转z轴 (根据实际情况考虑，是否需要翻转z轴。)
    sitk.WriteImage(flipped_image, output_file)


def gz_nii(nii_path, output_file='output.nii.gz', is_print=False):
    """将 .nii 为后缀的文件压缩为 .nii.gz 为后缀的文件."""
    nii = nib.load(nii_path)
    nib.save(nii, output_file)
    print(f'compression of {nii_path} is complete.') if is_print else None


def z_flip_nifti(nii_path, output_file='output.nii.gz', is_print=False):
    """翻转 nii.gz 影像的z轴"""
    nii = sitk.ReadImage(nii_path)
    flipped_nii = sitk.Flip(nii, [False, False, True])
    sitk.WriteImage(flipped_nii, output_file)
    print(f'flip of {nii_path} is complete.') if is_print else None
