from pydantic import BaseModel
from typing import List


class RopeScalingConfig(BaseModel):
    """旋转位置编码（RoPE）缩放配置（LongRoPE 式长度外推参数）。"""

    type: str  # 缩放类型标识
    long_factor: List[float]  # 超过原始长度时各频率维的外推系数
    short_factor: List[float]  # 原始长度内各频率维的缩放系数
    original_max_position_embeddings: int  # 训练时的原始最大位置长度


class MiniCPM4Config(BaseModel):
    """MiniCPM4 Transformer 模型配置。"""

    bos_token_id: int  # 起始 token id
    eos_token_id: int  # 结束 token id
    hidden_size: int  # 隐层维度
    intermediate_size: int  # 前馈网络 FFN 的中间层维度
    max_position_embeddings: int  # 支持的最大位置长度（外推后）
    num_attention_heads: int  # 注意力头数
    num_hidden_layers: int  # Transformer 层数
    num_key_value_heads: int  # KV 头数（分组查询注意力 GQA，可小于注意力头数）
    rms_norm_eps: float  # RMSNorm 的数值稳定小量
    rope_scaling: RopeScalingConfig  # RoPE 缩放配置
    vocab_size: int  # 词表大小（0 表示不使用词嵌入，直接输入连续向量）
    use_mup: bool = True  # 是否使用 muP 风格的残差分支缩放
    scale_emb: float  # 嵌入缩放系数
    dim_model_base: int  # muP 基准模型维度
    scale_depth: float  # muP 深度缩放系数
    rope_theta: float  # RoPE 基频 theta
    kv_channels: int = None  # 每个 KV 头的维度（None 时取 hidden_size / num_attention_heads）
    no_rope: bool = False  # 是否完全禁用 RoPE
