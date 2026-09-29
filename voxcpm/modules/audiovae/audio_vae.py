import math
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from pydantic import BaseModel


def WNConv1d(*args, **kwargs):
    """带权重归一化（weight norm）的一维卷积。"""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """带权重归一化的一维转置卷积（用于上采样）。"""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Conv1d):
    """因果一维卷积：只在序列左侧（过去方向）补零，保证输出不依赖未来帧。"""

    def __init__(self, *args, padding: int = 0, **kwargs):
        """初始化因果卷积。

        参数:
            padding: 因果填充长度（forward 时在序列左侧补零）
        """
        super().__init__(*args, **kwargs)
        self.__padding = padding  # 保存填充长度，forward 时手动左侧补零

    def forward(self, x):
        """前向：左侧补 2*padding 个零后卷积，保持输出长度与输入一致且严格因果。"""
        x_pad = F.pad(x, (self.__padding * 2, 0))  # 仅在左侧填充 2*padding（配合卷积本身无 padding），保证因果性
        return super().forward(x_pad)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    """因果一维转置卷积：上采样后在右侧（未来方向）裁剪，保证输出与因果流对齐。"""

    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        """初始化因果转置卷积。

        参数:
            padding: 卷积核两侧的裁剪量
            output_padding: 奇数步长时为对齐长度附加的输出补零量
        """
        super().__init__(*args, **kwargs)
        self.__padding = padding  # 保存裁剪量
        self.__output_padding = output_padding  # 保存输出补零量

    def forward(self, x):
        """前向：转置卷积后裁剪右侧多余长度，实现因果上采样。"""
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]  # 裁掉右侧（未来方向）多余输出


def WNCausalConv1d(*args, **kwargs):
    """权重归一化的因果一维卷积。"""
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    """权重归一化的因果一维转置卷积。"""
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    """Snake 激活函数：x + sin^2(alpha*x)/alpha，常用于音频波形生成网络以保留周期性。"""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)  # 展平后续维度，兼容任意形状输入
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)  # snake 公式，alpha+1e-9 防止除零
    x = x.reshape(shape)  # 恢复原始形状
    return x


class Snake1d(nn.Module):
    """逐通道 Snake 激活层（alpha 为逐通道可学习参数，形状 (1, channels, 1)）。"""

    def __init__(self, channels):
        """初始化逐通道 Snake 激活。

        参数:
            channels: 通道数
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))  # 可学习频率参数 alpha，逐通道独立

    def forward(self, x):
        """前向：应用 snake 激活。"""
        return snake(x, self.alpha)


def init_weights(m):
    """卷积层权重初始化：截断正态分布（std=0.02），偏置置零。"""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)  # 截断正态初始化，小方差利于训练稳定
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 偏置初始化为 0


class CausalResidualUnit(nn.Module):
    """因果残差单元：Snake -> 膨胀因果卷积 -> Snake -> 1x1 因果卷积，输出与输入残差相加。"""

    def __init__(self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1):
        """初始化因果残差单元。

        参数:
            dim: 通道数
            dilation: 膨胀率（扩大感受野）
            kernel: 卷积核大小，默认 7
            groups: 分组卷积的组数（1 为普通卷积，等于通道数时为深度卷积）
        """
        super().__init__()
        pad = ((7 - 1) * dilation) // 2  # 按核大小 7 与膨胀率计算两侧填充（此处的 7 为设计核长）
        self.block = nn.Sequential(
            Snake1d(dim),  # Snake 激活
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),  # 膨胀因果卷积，扩大时间感受野
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),  # 1x1 卷积做通道混合
        )

    def forward(self, x):
        """残差前向：block(x) 与输入相加（填充已对齐，理论上长度差为 0）。"""
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2  # 输入输出长度差的一半（对齐时应为 0）
        assert pad == 0
        if pad > 0:
            x = x[..., pad:-pad]  # 若长度不一致则居中裁剪输入对齐
        return x + y  # 残差连接


class CausalEncoderBlock(nn.Module):
    """因果编码器块：三级膨胀残差单元 + Snake + 步长卷积下采样（通道翻倍）。"""

    def __init__(self, output_dim: int = 16, input_dim=None, stride: int = 1, groups=1):
        """初始化编码器块。

        参数:
            output_dim: 输出通道数
            input_dim: 输入通道数，未指定时取 output_dim 的一半
            stride: 下采样步长（时间维压缩倍数）
            groups: 分组卷积组数
        """
        super().__init__()
        input_dim = input_dim or output_dim // 2  # 未指定输入通道时默认减半（与逐级翻倍的通道设计配套）
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),  # 膨胀率 1/3/9 逐级扩大感受野
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),  # kernel=2*stride 的卷积实现 stride 倍下采样
        )

    def forward(self, x):
        """编码器块前向：残差堆叠后下采样，输出长度约为输入的 1/stride。"""
        return self.block(x)


class CausalEncoder(nn.Module):
    """因果 VAE 编码器：逐级下采样并翻倍通道，最后分别输出高斯后验的均值 mu 与对数方差 logvar。"""

    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list = [2, 4, 8, 8],
        depthwise: bool = False,
    ):
        """初始化因果编码器。

        参数:
            d_model: 基础通道数（每级下采样后翻倍）
            latent_dim: 隐变量通道数
            strides: 各级下采样步长列表（总压缩倍数为各步长之积）
            depthwise: 是否使用深度可分离卷积
        """
        super().__init__()
        # Create first convolution
        self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]  # 首层卷积：单声道波形 -> d_model 通道

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2  # 每级下采样通道翻倍
            groups = d_model // 2 if depthwise else 1  # 深度可分离时按半通道分组
            self.block += [CausalEncoderBlock(output_dim=d_model, stride=stride, groups=groups)]

        groups = d_model if depthwise else 1

        # Create two convolution, for mu and logvar
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)  # 输出高斯后验均值 mu
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)  # 输出高斯后验对数方差 logvar

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model  # 记录末级通道数

    def forward(self, x):
        """编码前向。

        参数:
            x: 输入波形，形状 (B, 1, T)

        返回:
            字典，包含 hidden_state（主干特征）、mu、logvar，时间维被压缩为 T/prod(strides)
        """
        hidden_state = self.block(x)  # 逐级下采样后的特征 (B, enc_dim, T')
        return {
            "hidden_state": hidden_state,
            "mu": self.fc_mu(hidden_state),  # 后验均值 mu，(B, latent_dim, T')
            "logvar": self.fc_logvar(hidden_state),  # 后验对数方差 logvar，(B, latent_dim, T')
        }


class NoiseBlock(nn.Module):
    """噪声注入块：把经逐通道调制的随机噪声加到特征上（提高生成多样性/正则化）。"""

    def __init__(self, dim):
        """初始化噪声注入块。

        参数:
            dim: 通道数
        """
        super().__init__()
        self.linear = WNCausalConv1d(dim, dim, kernel_size=1, bias=False)  # 1x1 卷积学习逐通道噪声幅度

    def forward(self, x):
        """前向：x + noise * linear(x)。

        参数:
            x: 输入特征，形状 (B, C, T)
        """
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)  # 每个时间步独立的标准高斯噪声，广播到所有通道
        h = self.linear(x)
        n = noise * h  # 用逐通道缩放调制噪声幅度
        x = x + n  # 注入噪声
        return x


class CausalDecoderBlock(nn.Module):
    """因果解码器块：Snake -> 因果转置卷积上采样 ->（可选噪声块）-> 三级膨胀残差单元。"""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups=1,
        use_noise_block: bool = False,
    ):
        """初始化解码器块。

        参数:
            input_dim: 输入通道数
            output_dim: 输出通道数
            stride: 上采样倍率（时间维放大倍数）
            groups: 分组卷积组数
            use_noise_block: 是否插入噪声注入块
        """
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),  # kernel=2*stride 的转置卷积实现 stride 倍上采样；output_padding=stride%2 处理奇数步长的长度对齐
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))  # 可选：上采样后注入噪声
        layers.extend(
            [
                CausalResidualUnit(output_dim, dilation=1, groups=groups),
                CausalResidualUnit(output_dim, dilation=3, groups=groups),
                CausalResidualUnit(output_dim, dilation=9, groups=groups),  # 膨胀率 1/3/9 的残差单元细化特征
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """解码器块前向：上采样后残差细化，输出长度为输入的 stride 倍。"""
        return self.block(x)


class TransposeLastTwoDim(torch.nn.Module):
    """交换张量最后两个维度（(B, C, T) 与 (B, T, C) 互转）。"""

    def forward(self, x):
        """前向：转置最后两维。"""
        return torch.transpose(x, -1, -2)


class CausalDecoder(nn.Module):
    """因果 VAE 解码器：逐级转置卷积上采样并减半通道，最后经 Tanh 输出波形。"""

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
    ):
        """初始化因果解码器。

        参数:
            input_channel: 输入（隐变量）通道数
            channels: 首级基础通道数（逐级减半）
            rates: 各级上采样倍率列表（与编码器下采样步长互为逆序）
            depthwise: 是否使用深度可分离卷积
            d_out: 输出通道数（波形通常为 1）
            use_noise_block: 是否使用噪声注入块
        """
        super().__init__()

        # Add first conv layer
        if depthwise:
            layers = [
                WNCausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),  # 深度可分离：先逐通道卷积
                WNCausalConv1d(input_channel, channels, kernel_size=1),  # 再 1x1 卷积跨通道混合
            ]
        else:
            layers = [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]  # 普通首层卷积

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i  # 第 i 级输入通道（逐级减半）
            output_dim = channels // 2 ** (i + 1)  # 第 i 级输出通道
            groups = output_dim if depthwise else 1  # 深度可分离时组数等于输出通道数
            layers += [
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=groups,
                    use_noise_block=use_noise_block,
                )
            ]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),  # 最终激活
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),  # 输出卷积：恢复到 d_out 通道（单声道）
            nn.Tanh(),  # Tanh 限幅到 (-1, 1)，对应波形幅度范围
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """解码器前向：隐变量逐级上采样重建波形。"""
        return self.model(x)


class AudioVAEConfig(BaseModel):
    """AudioVAE 配置项。

    字段说明:
        encoder_dim: 编码器基础通道数
        encoder_rates: 编码器各级下采样步长（总压缩率为各元素之积）
        latent_dim: 隐变量通道数
        decoder_dim: 解码器基础通道数
        decoder_rates: 解码器各级上采样倍率（与 encoder_rates 互为逆序）
        depthwise: 是否使用深度可分离卷积
        sample_rate: 目标音频采样率（默认 16000 Hz）
        use_noise_block: 解码器是否使用噪声注入块
    """

    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_dim: int = 1536
    decoder_rates: List[int] = [8, 8, 5, 2]
    depthwise: bool = True
    sample_rate: int = 16000
    use_noise_block: bool = False


class AudioVAE(nn.Module):
    """音频变分自编码器（AudioVAE）：全因果卷积编解码结构，把波形压缩为连续隐变量并可还原。

    编码器输出高斯后验的 mu/logvar（推理时取 mu 作为隐变量），解码器从隐变量重建波形；
    全流程因果设计，支持流式编解码。
    """

    def __init__(
        self,
        config: AudioVAEConfig = None,
    ):
        """初始化 AudioVAE。

        参数:
            config: AudioVAEConfig 配置对象，为 None 时使用默认配置
        """
        # 如果没有传入config，使用默认配置
        if config is None:
            config = AudioVAEConfig()

        super().__init__()

        encoder_dim = config.encoder_dim
        encoder_rates = config.encoder_rates
        latent_dim = config.latent_dim
        decoder_dim = config.decoder_dim
        decoder_rates = config.decoder_rates
        depthwise = config.depthwise
        sample_rate = config.sample_rate
        use_noise_block = config.use_noise_block  # 展开各项配置

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.depthwise = depthwise

        self.use_noise_block = use_noise_block

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))  # 未指定隐变量维度时取编码器末级通道数

        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)  # 总压缩率：每个隐变量帧对应的波形采样点数
        self.encoder = CausalEncoder(
            encoder_dim,
            latent_dim,
            encoder_rates,
            depthwise=depthwise,
        )

        self.decoder = CausalDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=depthwise,
            use_noise_block=use_noise_block,
        )
        self.sample_rate = sample_rate
        self.chunk_size = math.prod(encoder_rates)  # 分块大小 = 总压缩率，即一个隐变量帧对应的采样点数

    def preprocess(self, audio_data, sample_rate):
        """预处理：将音频长度在右侧补零对齐到 hop_length 的整数倍。

        参数:
            audio_data: 输入音频张量，形状 (B, 1, T)
            sample_rate: 音频采样率（None 时使用模型自身采样率）

        返回:
            补零对齐后的音频张量，长度为 hop_length 的整数倍
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate  # 采样率必须与模型一致
        pad_to = self.hop_length  # 对齐目标：总压缩率
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length  # 计算需补的采样点数（向上取整到帧的整数倍）
        audio_data = nn.functional.pad(audio_data, (0, right_pad))  # 仅在右侧（尾部）补零，保持因果性

        return audio_data

    def decode(self, z: torch.Tensor):
        """将给定隐变量解码为音频数据。

        参数:
            z: 连续/量化隐变量，形状 (B, D, T)

        返回:
            解码出的音频张量，形状 (B, 1, length)，length = T * 总上采样倍率
        """
        return self.decoder(z)

    def encode(self, audio_data: torch.Tensor, sample_rate: int):
        """将音频编码为隐变量（取后验均值 mu）。

        参数:
            audio_data: 输入音频，形状 (B, T) 或 (B, 1, T)
            sample_rate: 音频采样率

        返回:
            隐变量 z，形状 (B, D, T')，其中 T' = 对齐后长度 / hop_length
        """
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)  # (B, T) -> (B, 1, T) 补上通道维

        audio_data = self.preprocess(audio_data, sample_rate)  # 长度对齐到 hop_length 整数倍
        return self.encoder(audio_data)["mu"]  # 推理时取后验均值 mu 作为隐变量
