# -*- coding: utf-8 -*-
# @Time    : 2025/1/18 21:25
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: transform_for_3D.py
# @Project : Radiological_image_processing
# !!! 因 pytorch 框架下缺少对 3D 图像的数据增强操作。故次文件多用于 3D 数据增广。
# 使用方式为直接添加在 transforms.Compose([f1, ..., fn]) 当中，其中 f1, ..., fn 可以直接使用下面的类。
# 需要注意 3D 影像的 shape 有4维：(C, D, H, W)。其中 C 常为 1 。
# 最后，默认处理 numpy.array 和 torch.tensor 类型。
import torch
import numpy as np
from typing import Tuple, Union
import torch.nn.functional as F


class RandomCrop3D:
    def __init__(self, crop_size: Tuple[int, int, int], seed: int = None):
        """初始化 RandomCrop3D。
        Args:
            crop_size (tuple): 裁剪目标大小 (depth, height, width)。
            seed (int, optional): 随机数种子，用于结果复现。默认为 None。
        """
        self.crop_size = crop_size
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """对输入 3D 图像进行随机裁剪。
        Args:
            image (torch.Tensor): 输入 3D 图像，形状为 (C, D, H, W)。
        Returns:
            torch.Tensor: 裁剪后的 3D 图像，形状为 (C, crop_depth, crop_height, crop_width)。
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(image)}.")

        if image.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (C, D, H, W), but got {image.shape}.")

        _, depth, height, width = image.shape
        crop_depth, crop_height, crop_width = self.crop_size

        # 自动适应裁剪大小
        crop_depth = min(crop_depth, depth)
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)

        # 随机生成裁剪起始位置
        start_d = np.random.randint(0, depth - crop_depth + 1)
        start_h = np.random.randint(0, height - crop_height + 1)
        start_w = np.random.randint(0, width - crop_width + 1)

        # 裁剪图像
        cropped_image = image[
            :,
            start_d:start_d + crop_depth,
            start_h:start_h + crop_height,
            start_w:start_w + crop_width
        ]

        return cropped_image


class Resize3D:
    def __init__(self, size: Union[Tuple[int, int, int], Tuple[float, float, float]]):
        """初始化 Resize3D。
        Args:
            size (tuple): 可以是绝对大小 (new_depth, new_height, new_width)，
                          或按比例调整大小 (scale_depth, scale_height, scale_width)。
        """
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """对输入 3D 图像进行大小调整。
        Args:
            image (torch.Tensor): 输入 3D 图像，形状为 (C, D, H, W)。
        Returns:
            torch.Tensor: 调整大小后的 3D 图像。
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, but got {type(image)}.")

        if image.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (C, D, H, W), but got {image.shape}.")

        _, depth, height, width = image.shape

        if all(isinstance(x, float) for x in self.size):  # 按比例调整
            scale_depth, scale_height, scale_width = self.size
            new_depth = int(depth * scale_depth)
            new_height = int(height * scale_height)
            new_width = int(width * scale_width)
        elif all(isinstance(x, int) for x in self.size):  # 按绝对大小调整
            new_depth, new_height, new_width = self.size
        else:
            raise ValueError("Size must be a tuple of all floats (scales) or all ints (absolute dimensions).")

        # 使用 PyTorch 的插值方法进行大小调整
        resized_image = F.interpolate(
            image.unsqueeze(0),  # 插值需要输入形状为 (N, C, D, H, W)，所以添加批量维度
            size=(new_depth, new_height, new_width),
            mode="trilinear",  # 使用三线性插值
            align_corners=False
        ).squeeze(0)  # 移除批量维度

        return resized_image
