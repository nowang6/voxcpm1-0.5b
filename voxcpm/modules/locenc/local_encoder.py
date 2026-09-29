import torch
import torch.nn as nn
from ..minicpm4 import MiniCPMModel, MiniCPM4Config
from einops import rearrange


class VoxCPMLocEnc(nn.Module):
    """VoxCPM 局部编码器：对每个分段（patch）内的音频隐变量做 Transformer 编码。

    在每个分段序列前拼接一个可学习的特殊 token（类似 CLS token），使用非因果（双向）
    Transformer 提取分段内表示，并取特殊 token 位置的输出作为该分段的摘要表示。
    """

    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        """初始化局部编码器。

        参数:
            config: MiniCPM4 配置对象（要求 vocab_size 为 0，即不使用词嵌入、直接输入连续向量）
            input_dim: 输入音频隐变量维度（对应 AudioVAE 的 latent_dim）
        """
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))  # 可学习的特殊 token（CLS），形状 (1, 1, 1, hidden_size)
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)  # 输入维度投影到模型隐层维度

        assert config.vocab_size == 0, "vocab_size must be 0 for local encoder"
        self.encoder = MiniCPMModel(config)  # 复用 MiniCPM4 Transformer 作为编码器主干

    def forward(self, x):
        """对分段音频隐变量编码，返回每个分段一个 token 的摘要表示。

        参数:
            x: 输入张量，形状 [B, T, P, D]（批大小、分段数、每段帧数、隐变量维度）

        返回:
            每个分段对应的摘要表示，形状 [B, T, hidden_size]
        """
        B, T, P, D = x.shape

        x = self.in_proj(x)  # (B, T, P, hidden_size) 维度投影
        special_tokens = self.special_token.expand(B, T, 1, -1)  # (B, T, 1, hidden_size) 将特殊 token 广播到每个分段
        x = torch.cat([special_tokens, x], dim=2)  # 在分段内序列维（P）前拼接特殊 token -> (B, T, P+1, hidden_size)
        x = rearrange(x, "b t p c -> (b t) p c")  # 合并批与分段维度：(B*T, P+1, hidden_size)，各分段独立编码
        outputs, _ = self.encoder(x, is_causal=False)  # 非因果（双向）注意力编码每个分段
        cls_output = outputs[:, 0, :]  # (B*T, hidden_size) 取特殊 token 位置的输出作为分段摘要

        return rearrange(cls_output, "(b t) c -> b t c", b=B)  # 恢复形状：(B, T, hidden_size)
