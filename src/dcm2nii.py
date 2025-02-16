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

# ********************************DEMO:数据整理********************************

"""
遍历指定的根目录，寻找包含.dcm文件的文件夹，将dcm文件转换为nii.gz文件。
将输出文件命名为 患者名称-子文件夹名称.nii.gz。(患者名称默认取子文件夹的上一级名称，具体可根据实际情况修改相关代码)
将输出文件保存到根目录下的 NIfTI_Output 文件夹中。
e.g.
    /home/***/***/code-project/A-DataSet/Colorectal_Medical_Imaging_bak/
    ├── Patient1
    │   ├── Subfolder1
    │   │   ├── image1.dcm
    │   │   ├── image2.dcm
    │   │   └── ...
    │   └── Subfolder2
    │       ├── image1.dcm
    │       └── ...
    ├── Patient2
    │   ├── Subfolder1
    │   │   ├── image1.dcm
    │   │   └── ...
    │   └── ...
    └── NIfTI_Output
        ├── Patient1-Subfolder1.nii.gz
        ├── Patient1-Subfolder2.nii.gz
        ├── Patient2-Subfolder1.nii.gz
        └── ...
"""
import os
import SimpleITK as sitk

OUTPUT_NIFTI_SUBFOLDER = "NIfTI_Output"  # 定义 NIfTI 输出子文件夹名称

def convert_dicom_to_nifti(dicom_folder, output_path):
    """
    将 DICOM 文件夹中的 DICOM 文件合并为一个 NIfTI 文件并保存到指定路径。

    Args:
        dicom_folder (str): 包含 DICOM 文件的文件夹路径。
        output_path (str): NIfTI 文件的输出路径。
    """
    try:
        # 使用 SimpleITK 读取 DICOM 文件序列
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        if not dicom_names:
            print(f"  警告: 文件夹 '{dicom_folder}' 中没有 DICOM 文件序列。")
            return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # 保存为 NIfTI 文件到指定路径
        sitk.WriteImage(image, output_path)
        print(f"  已保存 NIfTI 文件: {output_path}")
        return output_path

    except Exception as e:
        print(f"  错误: 处理文件夹 '{dicom_folder}' 失败，原因: {e}")
        return None

def get_names_from_path(dicom_folder_path, root_dir):
    """
    从 DICOM 文件夹路径中提取患者名称和子文件夹名称，用于 NIfTI 文件命名。
    按照 "向上两级" 的规则，如果层级不足则取最近的父文件夹名称。

    Args:
        dicom_folder_path (str): DICOM 文件夹的完整路径。
        root_dir (str): 根目录路径。

    Returns:
        tuple: (患者名称, 子文件夹名称)
    """
    relative_path = os.path.relpath(dicom_folder_path, root_dir)
    parts = relative_path.split(os.sep)

    subfolder_name = parts[-1] if parts else "UnknownSubfolder"
    patient_name = "UnknownPatient"

    if len(parts) >= 2:
        patient_name = parts[-2]

    return patient_name, subfolder_name


def process_directory_recursive(root_dir):
    """
    递归遍历根目录下的所有文件夹，查找包含 DICOM 文件的子文件夹并进行处理。
    NIfTI 文件将保存到根目录下的 OUTPUT_NIFTI_SUBFOLDER 中。

    Args:
        root_dir (str): 根目录路径，例如 'Colorectal_Medical_Imaging_bak'。
    """
    output_root_folder = os.path.join(root_dir, OUTPUT_NIFTI_SUBFOLDER) # NIfTI 输出文件夹路径

    # 创建 NIfTI 输出文件夹，如果不存在
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)
        print(f"创建 NIfTI 输出文件夹: {output_root_folder}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        has_dicom = False
        for filename in filenames:
            if filename.lower().endswith(".dcm"):
                has_dicom = True
                break

        if has_dicom:
            print(f"处理 DICOM 文件夹: {dirpath}")
            patient_name, subfolder_name = get_names_from_path(dirpath, root_dir)
            output_filename = f"{patient_name}-{subfolder_name}.nii.gz"
            output_path = os.path.join(output_root_folder, output_filename) # NIfTI 文件输出路径改为输出文件夹

            convert_dicom_to_nifti(dirpath, output_path)
        else:
            pass


if __name__ == "__main__":
    root_directory = "/home/***/***/demo/code-project/A-DataSet/Colorectal_Medical_Imaging_bak" # 替换为你的根目录
    if not os.path.exists(root_directory):
        print(f"错误: 根目录 '{root_directory}' 不存在，请检查路径。")
    else:
        print(f"开始递归处理目录: {root_directory}")
        process_directory_recursive(root_directory)
        print("DICOM to NIfTI 转换完成。NIfTI 文件保存在:", os.path.join(root_directory, OUTPUT_NIFTI_SUBFOLDER))
