# -*- coding: utf-8 -*-
# @Time    : 2025/5/29 10:53
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: brightness.py
# @Project : Radiological_image_processing
import torch
import torchio as tio
from typing import Union, Tuple


class MultiplicativeBrightnessTransform(tio.Transform):
    def __init__(self,
                 multiplier_range: Union[float, Tuple[float, float]] = (0.7, 1.3),
                 p_per_channel: float = 1.0,
                 synchronize_channels: bool = False,
                 p: float = 1.0):
        """
        乘性亮度变换 (支持多模态/多通道图像)

        参数说明：
        -----------
        multiplier_range : float 或 Tuple[float, float], 默认值=(0.7, 1.3)
            亮度乘数采样范围。若为元组 (a, b)，表示从均匀分布 U(a, b) 中采样乘数

        p_per_channel : float ∈ [0, 1], 默认值=1.0
            每个通道被应用变换的概率。例如 0.6 表示每个通道有60%概率被处理

        synchronize_channels : bool, 默认值=False
            是否对所有选中通道使用相同的乘数：
            - True: 所有选中通道使用相同乘数
            - False: 每个通道独立采样乘数

        p : float ∈ [0, 1], 默认值=1.0
            应用该变换的全局概率
        """
        super().__init__(p=p)
        self.multiplier_range = multiplier_range
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels

    def _sample_multiplier(self) -> float:
        """从multiplier_range采样乘数值"""
        if isinstance(self.multiplier_range, (tuple, list)):
            return torch.empty(1).uniform_(*self.multiplier_range).item()
        return self.multiplier_range

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # 获取图像数据 (形状: [C, (D), H, W])
        image_data = subject['image'].data
        C = image_data.shape[0]  # 通道数

        # 1. 确定需要处理的通道
        apply_to_channel = torch.rand(C) < self.p_per_channel
        selected_channels = torch.where(apply_to_channel)[0]
        if len(selected_channels) == 0:
            return subject  # 无通道被选中

        # 2. 采样乘数
        if self.synchronize_channels:
            # 所有选中通道共享同一乘数
            multiplier = self._sample_multiplier()
            multipliers = [multiplier] * len(selected_channels)
        else:
            # 每个通道独立采样乘数
            multipliers = [self._sample_multiplier() for _ in selected_channels]

        # 3. 应用亮度变换
        transformed = image_data.clone()
        for c, m in zip(selected_channels, multipliers):
            transformed[c] *= m

        # 更新subject数据
        subject['image'].set_data(transformed)
        return subject


# 测试用例
if __name__ == "__main__":
    # 创建测试数据 (3通道3D图像)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.rand(1, 40, 160, 256))  # (C, D, H, W)
    )

    # 初始化变换 (50%通道概率，同步乘数)
    brightness_transform = MultiplicativeBrightnessTransform(
        multiplier_range=(0.75, 1.25),
        synchronize_channels=False,  # 通道异步调整
        p_per_channel=1,  # 100%通道应用
        p=0.15,
    )

    # 应用变换
    transformed = brightness_transform(subject)

    # 验证结果
    print("原始数据范围:", subject['image'].data.min(), subject['image'].data.max())
    print("变换后范围:", transformed['image'].data.min(), transformed['image'].data.max())
    print("应用通道:", torch.where(
        brightness_transform.apply_transform(tio.Subject(image=subject.image))['image'].data != subject.image.data)[0])
    pass
