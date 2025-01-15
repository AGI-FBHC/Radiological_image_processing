"""
    建议在 Jupyter Notebook 中按需分别执行以下checker方法。
"""

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk


# ============================================================
def find_problematic_images(dataset_dir):
    """
    遍历数据集文件夹，找出无法被 SimpleITK 读取的图像文件。

    Args:
        dataset_dir: 数据集文件夹路径 (e.g., /path/to/nnUNet_raw/Dataset024_Colorectal)

    Returns:
        None. 打印出问题文件的路径和错误信息。
    """
    images_tr_dir = dataset_dir

    for filename in os.listdir(images_tr_dir):
        if filename.endswith(
            (".nii.gz", ".mha", ".mhd")
        ):  # 根据你的数据集文件扩展名修改
            filepath = os.path.join(images_tr_dir, filename)
            try:
                img = sitk.ReadImage(filepath)
                print(
                    f"Successfully read: {filename}, size: {img.GetSize()}, spacing: {img.GetSpacing()}"
                )
            except Exception as e:
                print(f"Error reading: {filename}")
                print(f"Exception: {e}")


"""
    # 使用示例
    dataset_dir = "Dataset024_Colorectal"  # 替换为你的数据集文件夹路径
    find_problematic_images(dataset_dir)
"""
# ============================================================


def check_direction_and_spacing(image_path, seg_path):
    """
    检查原图像和分割图像的方向和间距是否匹配。

    Args:
        image_path: 图像文件路径。
        seg_path: 分割文件路径。

    Returns:
        一个字典，包含错误信息。
    """
    problems = {}
    try:
        img = nib.load(image_path)
        seg = nib.load(seg_path)

        # 检查方向
        if not np.allclose(img.affine, seg.affine):
            direction_diff = np.abs(img.affine - seg.affine)
            if not np.allclose(
                direction_diff, np.zeros((4, 4)), atol=1e-5
            ):  # 允许微小差异
                problems["direction_mismatch"] = {
                    "image_direction": img.affine,
                    "seg_direction": seg.affine,
                }

        # 检查间距
        if not np.allclose(img.header["pixdim"][1:4], seg.header["pixdim"][1:4]):
            problems["spacing_mismatch"] = {
                "image_spacing": img.header["pixdim"][1:4],
                "seg_spacing": seg.header["pixdim"][1:4],
            }

    except Exception as e:
        problems["load_error"] = str(e)

    return problems


def find_problematic_files(images_dir, labels_dir):
    """
    文件夹批处理：
    找出源文件夹和标签文件夹中存在方向或间距不匹配问题的文件组合。

    Args:
        images_dir: 图像文件夹路径。
        labels_dir: 标签文件夹路径。

    Returns:
        一个字典，键值对为 (图像文件名, 分割文件名): 错误信息字典。
    """
    problematic_files = {}
    image_files = {
        f.replace("_0000.nii.gz", ".nii.gz"): f
        for f in os.listdir(images_dir)
        if f.endswith("_0000.nii.gz")
    }
    label_files = {f: f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")}

    for label_file_key, label_file in label_files.items():
        if label_file_key in image_files:
            image_file = image_files[label_file_key]
            image_path = os.path.join(images_dir, image_file)
            seg_path = os.path.join(labels_dir, label_file)

            problems = check_direction_and_spacing(image_path, seg_path)
            if problems:
                problematic_files[(image_file, label_file)] = problems
        else:
            print(f"Warning: No corresponding image found for label file: {label_file}")

    return problematic_files


"""
# 使用示例
images_dir = "imagesTr"  # 替换为你的图像文件夹路径
labels_dir = "labelsTr"  # 替换为你的标签文件夹路径

problematic_files = find_problematic_files(images_dir, labels_dir)

problematic_file_count = 0
problematic_file_names = []

if problematic_files:
    print("Found problematic files:")
    with open("problematic_files.txt", "w") as file:
        file.write("Found problematic files:\n")
        for (image_file, seg_file), problems in problematic_files.items():
            problematic_file_count += 1
            problematic_file_names.append(image_file)
            file.write(f"  Image: {image_file}\n")
            file.write(f"  Segmentation: {seg_file}\n")
            if "direction_mismatch" in problems:
                file.write(f"    Warning: Direction mismatch\n")
                file.write(f"      Image Direction: {problems['direction_mismatch']['image_direction']}\n")
                file.write(f"      Seg Direction: {problems['direction_mismatch']['seg_direction']}\n")
            if "spacing_mismatch" in problems:
                file.write(f"    Error: Spacing mismatch\n")
                file.write(f"      Image Spacing: {problems['spacing_mismatch']['image_spacing']}\n")
                file.write(f"      Seg Spacing: {problems['spacing_mismatch']['seg_spacing']}\n")
            if "load_error" in problems:
                file.write(f"    Error: {problems['load_error']}\n")
            file.write("-" * 20 + "\n")
    print(f"Total problematic files: {problematic_file_count}")
    print(f"Problematic file names: {problematic_file_names}")
else:
    print("No problematic files found.")
"""
# ============================================================


def fix_direction_and_spacing(image_path, seg_path, output_path):
    """
    修正分割文件的方向和间距，使其与图像文件一致，并绕 Z 轴旋转 180 度。

    Args:
        image_path: 图像文件路径。
        seg_path: 分割文件路径。
        output_path: 修正后的分割文件保存路径。
    """
    try:
        # 1. 加载图像和分割
        img = nib.load(image_path)
        seg = nib.load(seg_path)

        # 2. 获取图像的仿射矩阵和头部信息
        img_affine = img.affine
        img_header = img.header

        # 3. 修正分割的仿射矩阵（方向）
        seg_affine = img_affine.copy()  # 复制图像的仿射矩阵作为基础
        seg_affine[:3, 0] = -img_affine[:3, 0]  # 反转 X 轴方向
        seg_affine[:3, 1] = -img_affine[:3, 1]  # 反转 Y 轴方向

        # 添加绕 Z 轴旋转 180 度
        rotation_matrix = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        seg_affine = seg_affine @ rotation_matrix

        # 4. 修正分割的头部信息（间距）
        seg_header = seg.header.copy()
        seg_header["pixdim"][1:4] = img_header["pixdim"][1:4]  # 复制图像的间距信息

        # 5. 创建新的分割对象并保存
        new_seg = nib.Nifti1Image(seg.get_fdata(), seg_affine, seg_header)
        nib.save(new_seg, output_path)
        print(f"Fixed segmentation file saved to: {output_path}")

    except Exception as e:
        print(f"Error processing files: {e}")


def fix_all_labels(images_dir, labels_dir, output_dir):
    """
    修正标签文件夹中所有文件的方向和间距。

    Args:
        images_dir: 图像文件夹路径。
        labels_dir: 标签文件夹路径。
        output_dir: 修正后的标签文件保存路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    for seg_file in os.listdir(labels_dir):
        if seg_file.endswith(".nii.gz"):
            seg_path = os.path.join(labels_dir, seg_file)

            # 找到对应的图像文件
            # image_file = seg_file.replace(".nii.gz", "_0000.nii.gz")
            image_file = seg_file
            image_path = os.path.join(images_dir, image_file)

            # 如果找不到对应的图像文件，打印警告并跳过
            if not os.path.exists(image_path):
                print(
                    f"Warning: Could not find corresponding image file for {seg_file}. Skipping."
                )
                continue

            # 构建输出文件路径
            output_path = os.path.join(output_dir, seg_file)

            # 修正文件
            fix_direction_and_spacing(image_path, seg_path, output_path)


"""
    # 使用示例
    images_dir = "imagesTr"  # 替换为你的图像文件夹路径
    labels_dir = "labelsTr"  # 替换为你的标签文件夹路径
    output_dir = "fixed_labels"  # 替换为你想要保存修正后标签的文件夹路径
    fix_all_labels(images_dir, labels_dir, output_dir)
"""

# ============================================================
