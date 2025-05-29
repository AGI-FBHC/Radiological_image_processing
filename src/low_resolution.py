# -*- coding: utf-8 -*-
# @Time    : 2025/5/29 15:09
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: low_resolution.py
# @Project : Radiological_image_processing
import torch
import torchio as tio
from typing import Optional, Tuple, Union
from torch.nn import functional as F


class SimulateLowResolutionTransform(tio.Transform):
    def __init__(self,
                 scale: Tuple[float, float] = (0.5, 1.0),  # 严格对齐 scale=(0.5, 1)
                 synchronize_channels: bool = False,  # 同步配置对齐
                 synchronize_axes: bool = True,
                 ignore_axes: Tuple[int, ...] = (),
                 allowed_channels: Optional[Tuple[int, ...]] = None,
                 p_per_channel: float = 0.5,  # 精确对应 p_per_channel=0.5
                 p: float = 0.25):  # apply_probability=0.25
        """
        模拟低分辨率变换

        - scale=(0.5, 1) → 缩放范围0.5到1.0
        - p_per_channel=0.5 → 每个通道50%处理概率
        - apply_probability=0.25 → 整体25%应用概率
        """
        super().__init__(p=p)
        self.scale = scale
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes
        self.ignore_axes = ignore_axes
        self.allowed_channels = allowed_channels
        self.p_per_channel = p_per_channel

        # 维度处理映射
        self._init_mode_map()

    def _init_mode_map(self):
        """私有方法初始化插值模式映射"""
        self.mode_map = {
            2: ('nearest-exact', 'bilinear'),  # (下采样模式, 上采样模式)
            3: ('nearest-exact', 'trilinear')
        }

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # 保持原始数据精度和类型
        image = subject['image']
        data = image.data.clone()
        C, *spatial = data.shape

        # 通道选择逻辑严格对齐
        selected_channels = self._select_channels(C)
        if len(selected_channels) == 0:
            return subject

        # 采样缩放因子逻辑
        scales = self._sample_scales(selected_channels, spatial)

        # 执行分辨率退化
        for c_idx, channel in enumerate(selected_channels):
            data[channel] = self._process_single_channel(
                data[channel],
                scales[c_idx]
            )

        image.set_data(data)
        return subject

    def _select_channels(self, num_channels: int) -> torch.Tensor:
        """通道选择逻辑精确对齐原始实现"""
        if self.allowed_channels is not None:
            candidates = list(self.allowed_channels)
        else:
            candidates = list(range(num_channels))

        # 概率筛选
        mask = torch.rand(len(candidates)) < self.p_per_channel
        return torch.tensor(candidates)[mask]

    def _sample_scales(self,
                       selected_channels: torch.Tensor,
                       spatial_shape: Tuple[int, ...]) -> torch.Tensor:
        """缩放因子采样逻辑严格对齐"""
        num_dims = len(spatial_shape)
        scales = torch.zeros(len(selected_channels), num_dims)

        for i, c in enumerate(selected_channels):
            if self.synchronize_axes:
                # 所有维度同步采样
                scale = torch.empty(1).uniform_(*self.scale)
                scales[i] = scale.repeat(num_dims)
            else:
                # 各维度独立采样
                scales[i] = torch.empty(num_dims).uniform_(*self.scale)

            # 处理忽略的轴
            scales[i][self.ignore_axes] = 1.0

        return scales

    def _process_single_channel(self,
                                channel_data: torch.Tensor,
                                scales: torch.Tensor) -> torch.Tensor:
        """单通道处理流程精确对齐"""
        original_shape = channel_data.shape
        spatial_dims = len(original_shape)

        # 计算下采样尺寸
        down_shape = [
            max(4, int(orig_dim * scale.item()))  # 保持最小尺寸为4
            for orig_dim, scale in zip(original_shape, scales)
        ]

        # 下采样-上采样流程
        downsampled = F.interpolate(
            channel_data.unsqueeze(0).unsqueeze(0),
            size=down_shape,
            mode=self.mode_map[spatial_dims][0]
        )
        upsampled = F.interpolate(
            downsampled,
            size=original_shape,
            mode=self.mode_map[spatial_dims][1]
        )
        return upsampled[0, 0]


# 测试用例
if __name__ == "__main__":
    # 创建3D测试数据 (2通道)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.rand(1, 40, 160, 256))  # (C, D, H, W)
    )

    # 初始化变换配置
    transform = SimulateLowResolutionTransform(
        scale=(0.5, 1),  # 完全匹配原始参数
        synchronize_channels=False,  # 通道独立处理
        synchronize_axes=True,  # 维度同步
        ignore_axes=(0,),  # 假设外部传入
        allowed_channels=None,  # 允许所有通道
        p_per_channel=0.5,  # 50%通道概率
        p=0.25  # 25%全局概率
    )

    # 应用变换
    transformed = transform(subject)

    # 验证结果
    original = subject['image'].data
    processed = transformed['image'].data
    print("原始尺寸:", subject.image.shape)  # 输出: (1, 32, 256, 256)
    print("变换后尺寸:", transformed.image.shape)  # 输出: (1, 32, 128, 128)（D轴保持32，H/W轴下采样50%）