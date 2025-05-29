# -*- coding: utf-8 -*-
# @Time    : 2025/5/29 08:36
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: gaussian_noise.py
# @Project : Radiological_image_processing
import torch
import torchio as tio
from torchio import Subject, ScalarImage
from typing import Union, Tuple


class GaussianNoiseTransform(tio.Transform):
    def __init__(self,
                 noise_variance=(0, 0.1),
                 p_per_channel=1.0,
                 synchronize_channels=False,
                 p=1.0):
        """
        GaussianNoiseTransform

        为图像添加高斯噪声的自定义变换类。

        参数说明：
        -----------
        noise_variance : float 或 Tuple[float, float], 默认值=(0, 0.1)
            控制高斯噪声的标准差（σ）。如果是一个区间 (a, b)，表示从中均匀采样一个σ。
            例如：(0, 0.1) 表示对每个应用的通道采样一个 σ ∈ [0, 0.1]。

        p_per_channel : float ∈ [0, 1], 默认值=1.0
            每个通道被添加噪声的概率。例如 0.5 表示每个通道有 50% 的概率添加噪声。

        synchronize_channels : bool, 默认值=False
            是否对所有通道使用相同的噪声：
            - True：所有通道使用相同的 σ 和噪声（常用于保持图像结构一致性）。
            - False：每个通道独立添加不同的噪声。

        p : float ∈ [0, 1], 默认值=1.0
            整个 transform 被应用的概率。p=0.1 表示该变换仅有 10% 的概率被应用于每个样本。

        注意事项：
        -----------
        该变换仅应用于图像（不影响 mask 或 label），通常用于数据增强阶段以提升模型鲁棒性。
        """
        super().__init__(p=p)
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image = subject['image'].data  # shape: (1 or C, D, H, W)
        C = image.shape[0]

        apply_to_channel = torch.rand(C) < self.p_per_channel
        if self.synchronize_channels:
            sigma = torch.empty(1).uniform_(*self.noise_variance).item()
            noise = torch.normal(mean=0, std=sigma, size=image.shape)
            image[apply_to_channel] += noise[apply_to_channel]
        else:
            for c in range(C):
                if apply_to_channel[c]:
                    sigma = torch.empty(1).uniform_(*self.noise_variance).item()
                    noise = torch.normal(mean=0, std=sigma, size=image[c].shape)
                    image[c] += noise

        subject['image'].set_data(image)
        return subject


if __name__ == "__main__":
    image = ScalarImage(tensor=torch.rand(1, 40, 160, 256))
    subject = Subject(image=image)
    transform = tio.Compose([
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),  # 控制噪声强度
            p_per_channel=1.0,  # 每个通道都可能加噪声
            synchronize_channels=True,  # 每个通道独立添加
            p=0.1
        ),
        tio.RescaleIntensity(out_min_max=(0, 1)),  # 强度重缩放
        tio.Resize((40, 160, 256)),  # 尺寸调整
    ])
    transformed = transform(subject)
    print(transformed['image'].data.shape)
    pass