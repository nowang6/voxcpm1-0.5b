"""
VoxCPM1 (0.5B) 模型配置定义。

从 OpenBMB/VoxCPM 抽取而来，仅保留单一架构（voxcpm v1 / VoxCPM-0.5B）
且不含 LoRA：只支持全参数微调。
"""

from typing import Optional

from pydantic import BaseModel

from ..modules.audiovae import AudioVAEConfig
from ..modules.locdit import CfmConfig
from ..modules.minicpm4 import MiniCPM4Config


class VoxCPMEncoderConfig(BaseModel):
    """局部编码器（Local Encoder，VoxCPMLocEnc）的结构配置。"""

    hidden_dim: int = 1024  # 编码器隐藏层维度
    ffn_dim: int = 4096  # 前馈网络（FFN）中间层维度
    num_heads: int = 16  # 注意力头数
    num_layers: int = 4  # 层数
    kv_channels: int = None  # 每个注意力头/QK 向量的通道数（None 表示使用默认值）


class VoxCPMDitConfig(BaseModel):
    """局部 DiT 解码器（Local DiT，VoxCPMLocDiT）的结构配置。"""

    hidden_dim: int = 1024  # DiT 隐藏层维度
    ffn_dim: int = 4096  # 前馈网络（FFN）中间层维度
    num_heads: int = 16  # 注意力头数
    num_layers: int = 4  # 层数
    kv_channels: int = None  # 每个注意力头的通道数（None 表示使用默认值）

    cfm_config: CfmConfig  # 流匹配（CFM）相关超参配置


class PrefixTuneConfig(BaseModel):
    """P-Tuning 说话人前缀（soft prompt）配置。

    模型参数全部冻结，仅学习一组可学习的前缀向量来编码固定说话人，
    注入到 base_lm 与 residual_lm 的输入序列最前端（P-Tuning 风格）。
    """

    num_virtual_tokens: int = 32  # 前缀虚拟 token 数（容量约等效 2.5 秒参考音频）
    enabled: bool = False  # 是否注入前缀（推理/训练开关）


class VoxCPMConfig(BaseModel):
    """VoxCPM 模型整体配置：包含文本-语义 LM、残差声学 LM、局部编码器/解码器等子模块配置。"""

    lm_config: MiniCPM4Config  # 基础语言模型（MiniCPM4）配置
    patch_size: int = 2  # 每个 token 对应的音频 patch 大小（以 AudioVAE 隐变量帧为单位）
    feat_dim: int = 64  # 音频特征维度（patch 内每帧的隐变量维度）
    residual_lm_num_layers: int = 6  # 残差声学 LM 的层数（其余结构复用 lm_config）
    scalar_quantization_latent_dim: int = 256  # 标量量化层（FSQ）的隐变量维度
    scalar_quantization_scale: int = 9  # 标量量化层（FSQ）的量化级数

    encoder_config: VoxCPMEncoderConfig  # 局部编码器配置
    dit_config: VoxCPMDitConfig  # 局部 DiT 解码器配置
    audio_vae_config: Optional[AudioVAEConfig] = None  # AudioVAE 配置（None 时使用默认）

    max_length: int = 4096  # KV 缓存支持的最大序列长度
    device: str = "cuda"  # 默认运行设备
    dtype: str = "bfloat16"  # 默认计算精度
    dit_mean_mode: bool = False  # 流匹配解码器均值的计算模式
    prefix_tune: PrefixTuneConfig = PrefixTuneConfig()  # P-Tuning 前缀配置（默认关闭，保持与旧行为一致）


VoxCPMConfig.model_rebuild()
