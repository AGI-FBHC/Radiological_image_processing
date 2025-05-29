# -*- coding: utf-8 -*-
# @Time    : 2025/5/29 14:51
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: contrast.py
# @Project : Radiological_image_processing
import torch
import torchio as tio
from typing import Union, Tuple


class ContrastTransform(tio.Transform):
    def __init__(self,
                 contrast_range: Union[float, Tuple[float, float]] = (0.8, 1.2),
                 preserve_range: bool = True,
                 synchronize_channels: bool = False,
                 p_per_channel: float = 1.0,
                 p: float = 1.0):
        """
        对比度变换 (支持2D/3D医学影像)

        参数说明：
        -----------
        contrast_range : float 或 Tuple[float, float], 默认值=(0.8, 1.2)
            对比度调整乘数范围。例如(0.7, 1.3)表示随机缩放70%-130%

        preserve_range : bool, 默认值=True
            是否保持原始数值范围 (医学影像重要！如CT的Hounsfield单位)

        synchronize_channels : bool, 默认值=False
            是否跨通道同步对比度参数 (适用于多模态影像对齐)

        p_per_channel : float ∈ [0, 1], 默认值=1.0
            每个通道被处理的概率

        p : float ∈ [0, 1], 默认值=1.0
            整体应用概率
        """
        super().__init__(p=p)
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def _sample_contrast_factor(self) -> float:
        """从contrast_range采样对比度系数"""
        if isinstance(self.contrast_range, (tuple, list)):
            return torch.empty(1).uniform_(*self.contrast_range).item()
        return self.contrast_range

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # 获取图像数据 (形状: [C, (D), H, W])
        image_data = subject['image'].data
        C = image_data.shape[0]  # 通道数

        # 1. 确定需要处理的通道
        apply_to_channel = torch.rand(C) < self.p_per_channel
        selected_channels = torch.where(apply_to_channel)[0]
        if len(selected_channels) == 0:
            return subject

        # 2. 采样对比度系数
        if self.synchronize_channels:
            # 所有选中通道使用相同系数
            factor = self._sample_contrast_factor()
            factors = [factor] * len(selected_channels)
        else:
            # 每个通道独立采样
            factors = [self._sample_contrast_factor() for _ in selected_channels]

        # 3. 应用对比度变换
        transformed = image_data.clone().float()  # 转换为浮点以保持精度
        for c_idx, channel in enumerate(selected_channels):
            channel_data = transformed[channel]
            original_dtype = channel_data.dtype

            # 计算统计量
            mean = channel_data.mean()
            if self.preserve_range:
                v_min, v_max = channel_data.min(), channel_data.max()

            # 执行对比度调整 (公式: (x - μ)*α + μ)
            channel_data -= mean
            channel_data *= factors[c_idx]
            channel_data += mean

            # 保持数值范围
            if self.preserve_range:
                channel_data.clamp_(v_min, v_max)

            # 恢复原始数据类型
            transformed[channel] = channel_data.to(original_dtype)

        # 更新Subject数据
        subject['image'].set_data(transformed)
        return subject


# 测试用例
if __name__ == "__main__":
    # 创建3D测试数据 (CT模拟数据)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.rand(1, 40, 160, 256))
    )

    # 初始化变换 (同步通道对比度调整)
    contrast_transform = ContrastTransform(
        contrast_range=(0.75, 1.25),
        preserve_range=True,
        synchronize_channels=True,
        p_per_channel=1,
        p=0.15
    )

    # 应用变换
    transformed = contrast_transform(subject)

    # 验证结果
    print("原始数据范围:", subject['image'].data.min(), subject['image'].data.max())
    print("变换后范围:", transformed['image'].data.min(), transformed['image'].data.max())
    print("应用通道:", torch.where(transformed['image'].data != subject.image.data)[0])

