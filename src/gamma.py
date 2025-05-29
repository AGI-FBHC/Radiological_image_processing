# -*- coding: utf-8 -*-
# @Time    : 2025/5/29 15:22
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: gamma.py
# @Project : Radiological_image_processing
import torch
import torchio as tio
from typing import Union, Tuple, Optional

import torch
import torchio as tio
from typing import Tuple


class GammaTransform(tio.Transform):
    def __init__(self,
                 gamma: Tuple[float, float] = (0.7, 1.5),  # 对齐原始 gamma 参数名
                 p_invert_image: float = 1.0,  # 保持原始参数名
                 synchronize_channels: bool = False,  # 与原始参数名一致
                 p_per_channel: float = 1.0,  # 保持原始参数名
                 p_retain_stats: float = 1.0,  # 保持原始参数名
                 p: float = 0.1):  # 对应 Compose 的 apply_probability
        """
        严格参数命名对齐实现

        参数映射说明：
        - gamma: 对应原始 gamma 参数，但改为范围元组
        - apply_probability: 通过父类的 p 参数传递
        """
        super().__init__(p=p)
        self.gamma = gamma
        self.p_invert_image = p_invert_image
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
        self.p_retain_stats = p_retain_stats

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image = subject['image']
        data = image.data.clone().float()
        C = data.shape[0]

        # 强制处理所有通道 (p_per_channel=1)
        selected_channels = torch.arange(C)
        params = self._generate_params(selected_channels)

        # 应用变换 (强制保留统计量)
        for c_idx, channel in enumerate(selected_channels):
            self._process_channel(data[channel], params, c_idx)

        subject['image'].set_data(data.to(image.data.dtype))
        return subject

    def _generate_params(self, channels: torch.Tensor) -> dict:
        """参数生成逻辑"""
        num_channels = len(channels)

        # Gamma采样逻辑
        if self.synchronize_channels:
            gamma_val = self._sample_gamma()
            gamma = torch.full((num_channels,), gamma_val)
        else:
            gamma = torch.tensor([self._sample_gamma() for _ in range(num_channels)])

        return {
            'gamma': gamma,
            'retain_stats': torch.ones(num_channels, dtype=torch.bool),  # 强制保留统计量
            'invert': torch.ones(num_channels, dtype=torch.bool)  # 强制反转
        }

    def _process_channel(self,
                         channel_data: torch.Tensor,
                         params: dict,
                         c_idx: int):
        """单通道处理流程"""
        # 强制双反转 (等效原始操作)
        channel_data *= -1

        # 记录原始统计量
        orig_mean = channel_data.mean()
        orig_std = channel_data.std()

        # 执行伽马校正
        v_min = channel_data.min()
        v_range = channel_data.max() - v_min + 1e-7
        gamma = params['gamma'][c_idx].item()

        channel_data = ((channel_data - v_min) / v_range).pow(gamma) * v_range + v_min

        # 强制恢复统计量
        new_mean = channel_data.mean()
        new_std = channel_data.std()
        channel_data = (channel_data - new_mean) * (orig_std / (new_std + 1e-7)) + orig_mean

        # 恢复反转
        channel_data *= -1

    def _sample_gamma(self) -> float:
        """从 gamma 范围采样"""
        if isinstance(self.gamma, tuple):
            return torch.empty(1).uniform_(*self.gamma).item()
        return self.gamma  # 处理标量情况

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gamma={self.gamma}, "
            f"p_invert_image={self.p_invert_image}, "
            f"synchronize_channels={self.synchronize_channels}, "
            f"p_per_channel={self.p_per_channel}, "
            f"p_retain_stats={self.p_retain_stats}, "
            f"apply_probability={self.p})"  # 通过 self.p 访问父类参数
        )


if __name__ == "__main__":
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.rand(1, 40, 160, 256))
    )
    original = subject['image'].data.clone()

    transform = GammaTransform(
        gamma=(0.7, 1.5),
        p_invert_image=1.0,
        synchronize_channels=False,
        p_per_channel=1.0,
        p_retain_stats=1.0,
        p=1.0  # 强制应用，方便实验
    )

    transformed = transform(subject)
    transformed_data = transformed['image'].data

    print("是否发生变化：", not torch.allclose(original, transformed_data))
    print("原始 mean/std:", original.mean().item(), original.std().item())
    print("变换后 mean/std:", transformed_data.mean().item(), transformed_data.std().item())
