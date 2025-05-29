# -*- coding: utf-8 -*-
# @Time    : 2025/5/29 08:57
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: gaussian_blur.py
# @Project : Radiological_image_processing
import torch
import numpy as np
import torchio as tio
from torch.backends.cudnn import benchmark
from torchio import ScalarImage, Subject
from typing import Union, Tuple
import torch.nn.functional as F


class GaussianBlurTransform(tio.Transform):
    def __init__(self,
                 blur_sigma: Union[float, Tuple[float, float]] = (0.5, 1.5),
                 p_per_channel: float = 1.0,
                 synchronize_channels: bool = False,
                 synchronize_axes: bool = False,
                 p: float = 1.0):
        """
        高斯模糊变换 (支持2D/3D图像)

        参数说明：
        -----------
        blur_sigma : float 或 Tuple[float, float], 默认值=(0.5, 1.5)
            高斯核标准差范围。若为元组 (a, b)，表示从均匀分布 U(a, b) 中采样σ值

        p_per_channel : float ∈ [0, 1], 默认值=1.0
            每个通道被应用模糊的概率。例如 0.6 表示每个通道有60%概率被处理

        synchronize_channels : bool, 默认值=False
            是否对所有选中通道使用相同的σ值：
            - True: 所有通道使用相同的σ (保持跨通道结构一致性)
            - False: 每个通道独立采样σ

        synchronize_axes : bool, 默认值=False
            是否对所有空间维度使用相同的σ值 (仅对3D有效)：
            - True: x/y/z轴使用相同σ
            - False: 每个维度独立采样σ

        p : float ∈ [0, 1], 默认值=1.0
            应用该变换的全局概率
        """
        super().__init__(p=p)
        self.blur_sigma = blur_sigma
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes

    @staticmethod
    def build_gaussian_kernel(sigma: float,
                              dim: int) -> torch.Tensor:  # 移除 self 参数
        """构建高斯卷积核"""
        kernel_size = int(2 * torch.ceil(torch.tensor(3 * sigma)) + 1)
        kernel_size = max(3, kernel_size)

        # 创建1D高斯核
        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()

        # 扩展维度
        for _ in range(dim - 1):
            kernel = kernel.unsqueeze(-1)
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, *kernel_size)

    def apply_axis_aligned_blur(self,
                                image: torch.Tensor,
                                sigma: list,
                                dim: int) -> torch.Tensor:
        """应用各向异性模糊（沿每个轴做 1D 高斯模糊）"""
        blurred = image.clone()
        for axis in range(1, dim + 1):  # 因为 image 是 [1, D, H, W] 这样的，跳过通道维
            kernel = self.build_gaussian_kernel(sigma[axis - 1], dim=1)  # 获取1D kernel
            padding = kernel.shape[-1] // 2

            # 将目标轴放到最后一维（便于做 conv1d）
            permute_dims = list(range(image.ndim))
            permute_dims.append(permute_dims.pop(axis))  # 将 axis 移到最后
            permuted = image.permute(permute_dims)  # [1, ..., L]

            # 将最后一维 reshape 成 [N, C, L] 的形式
            shape = permuted.shape
            reshaped = permuted.reshape(-1, 1, shape[-1])  # 合并前面的维度成 batch

            # 应用 conv1d
            blurred_1d = F.conv1d(reshaped, kernel, padding=padding, groups=1)

            # 恢复形状
            blurred_1d = blurred_1d.reshape(*shape)
            inv_permute_dims = [permute_dims.index(i) for i in range(len(permute_dims))]
            blurred_1d = blurred_1d.permute(inv_permute_dims)

            # 累加每轴模糊
            blurred += blurred_1d
        return blurred / dim  # 平均模糊

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # 获取图像数据 (形状: [C, (D), H, W])
        image = subject['image'].data
        spatial_dims = image.ndim - 1  # 空间维度数 (2或3)
        C = image.shape[0]  # 通道数

        # 1. 确定需要处理的通道
        apply_to_channel = torch.rand(C) < self.p_per_channel
        if not apply_to_channel.any():
            return subject  # 无通道被选中

        # 2. 采样σ值
        if self.synchronize_channels:
            if self.synchronize_axes and spatial_dims == 3:
                sigma = self._sample_sigma()  # 所有维度和通道共享
                sigma = [sigma] * spatial_dims
            else:
                sigma = [self._sample_sigma() for _ in range(spatial_dims)]
            # 扩展至选中通道
            sigmas = [sigma] * apply_to_channel.sum().item()
        else:
            sigmas = []
            for c in np.where(apply_to_channel)[0]:
                if self.synchronize_axes and spatial_dims == 3:
                    s = self._sample_sigma()
                    sigma_c = [s] * spatial_dims
                else:
                    sigma_c = [self._sample_sigma() for _ in range(spatial_dims)]
                sigmas.append(sigma_c)

        # 3. 应用模糊
        blurred = image.clone()
        for i, c in enumerate(np.where(apply_to_channel)[0]):
            # 提取当前通道图像 [1, (D), H, W]
            channel_img = image[c].unsqueeze(0)

            # 执行各向异性模糊
            blurred_channel = self.apply_axis_aligned_blur(
                channel_img,
                sigma=sigmas[i],
                dim=spatial_dims
            )

            blurred[c] = blurred_channel.squeeze(0)

        subject['image'].set_data(blurred)
        return subject

    def _sample_sigma(self) -> float:
        """从blur_sigma范围采样σ值"""
        if isinstance(self.blur_sigma, (tuple, list)):
            return torch.empty(1).uniform_(*self.blur_sigma).item()
        return self.blur_sigma


if __name__ == '__main__':
    # 创建模拟图像
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.rand(1, 40, 160, 256))  # (C, D, H, W)
    )

    blur_transform = GaussianBlurTransform(
        blur_sigma=(0.5, 1.),
        synchronize_channels=False,
        synchronize_axes=False,
        p_per_channel=0.5,
        p=1
    )

    transformed = blur_transform(subject)
    print("Input shape:", subject['image'].data.shape)
    print("Output shape:", transformed['image'].data.shape)
    pass

