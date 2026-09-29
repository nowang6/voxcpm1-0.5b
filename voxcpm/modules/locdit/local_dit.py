import torch
from ..minicpm4 import MiniCPMModel, MiniCPM4Config
import torch.nn as nn
import math


class SinusoidalPosEmb(torch.nn.Module):
    """正弦数值嵌入：将标量（如扩散时间步）映射为正弦-余弦向量表示。"""

    def __init__(self, dim):
        """初始化正弦嵌入。

        参数:
            dim: 输出嵌入维度（必须为偶数，一半用 sin、一半用 cos）
        """
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, time, scale=1000):
        """将标量数值编码为正弦-余弦嵌入。

        参数:
            time: 标量或形状 (N,) 的时间/数值张量
            scale: 数值放大系数（magic number，1000 与常见扩散模型的时间步量级一致）

        返回:
            形状 (N, dim) 的嵌入张量，前半为 sin、后半为 cos
        """
        if time.ndim < 1:
            time = time.unsqueeze(0)  # 标量 -> 一维张量，保证后续形状一致
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  # 频率间隔（对数尺度，base=10000 与标准正弦位置编码一致）
        emb = torch.exp(torch.arange(half_dim, dtype=time.dtype, device=device) * -emb)  # (half_dim,) 各维频率
        emb = scale * time.unsqueeze(1) * emb.unsqueeze(0)  # (N, half_dim) 数值乘以各维频率
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (N, dim) 拼接 sin 与 cos
        return emb


class TimestepEmbedding(nn.Module):
    """时间步嵌入 MLP：把正弦编码后的时间向量进一步映射为模型可用的条件向量。"""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = None,
    ):
        """初始化时间步嵌入。

        参数:
            in_channels: 输入向量维度（正弦嵌入的维度）
            time_embed_dim: 中间隐层维度
            out_dim: 输出维度，未指定时与中间隐层维度相同
        """
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)  # 第一层变换
        self.act = nn.SiLU()  # SiLU 激活函数
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim  # 未指定输出维度时保持维度不变

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)  # 输出层

    def forward(self, sample):
        """两层 MLP 前向：Linear -> SiLU -> Linear。

        参数:
            sample: 输入张量，形状 (N, in_channels)

        返回:
            嵌入张量，形状 (N, out_dim)
        """
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class VoxCPMLocDiT(nn.Module):
    """基于 Transformer 主干的（去噪）扩散模型：作为条件流匹配（CFM）的速度场估计器。"""

    def __init__(
        self,
        config: MiniCPM4Config,
        in_channels: int = 64,
    ):
        """初始化局部 DiT。

        参数:
            config: MiniCPM4 配置对象（要求 vocab_size 为 0，直接输入连续向量而非词嵌入）
            in_channels: 输入/输出隐变量通道数（等于 AudioVAE 的 latent_dim）
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.config = config

        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)  # 当前状态 x_t 的输入投影
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)  # 前缀条件（上文）的输入投影
        self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)  # 输出投影：预测速度场的输出层

        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)  # 时间步的正弦编码
        self.time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )  # 当前时刻 t 的 MLP 嵌入
        self.delta_time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )  # ODE 步长（delta t）的 MLP 嵌入
        # CFG 蒸馏：引导强度 w 的嵌入，使单次前向即可复现 cond/uncond 双分支的 CFG 组合速度场
        self.cfg_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )

        assert config.vocab_size == 0, "vocab_size must be 0 for local DiT"
        self.decoder = MiniCPMModel(config)  # 复用 MiniCPM4 Transformer 作为 DiT 主干

    def init_cfg_embedding_zero(self):
        """零初始化 cfg 嵌入 MLP。

        保证蒸馏学生从「教师条件分支」的行为起步（w 嵌入初始输出为 0，
        不改变原模型的前向结果）。
        """
        nn.init.zeros_(self.cfg_mlp.linear_1.weight)
        nn.init.zeros_(self.cfg_mlp.linear_1.bias)
        nn.init.zeros_(self.cfg_mlp.linear_2.weight)
        nn.init.zeros_(self.cfg_mlp.linear_2.bias)

    def forward(
        self,
        state: torch.Tensor,
        mu: torch.Tensor,
        current_time: torch.Tensor,
        cond: torch.Tensor,
        time_step: torch.Tensor,
        w: torch.Tensor | None = None,
    ):
        """DiT 前向：预测流匹配速度场 v(x_t, t | 条件)。

        参数:
            state: 当前状态（噪声插值点）张量，形状 (N, C, T)
            mu: 全局条件隐变量嵌入，形状 (N, C)
            current_time: 当前扩散时间步，形状 (N,)
            cond: 前缀（上文）条件张量，形状 (N, C, T')
            time_step: ODE 步长 (t - r)，形状 (N,)，用于平均速度模式（未来可能启用）
            w: 可选的 CFG 引导强度，形状 (N,)。蒸馏学生传入后将 w 嵌入叠加到
                时间条件上，用单次前向复现 CFG 组合速度场；None 即普通前向
        """
        state = self.in_proj(state.transpose(1, 2).contiguous())  # (N, C, T) -> (N, T, C) 后投影到隐层维度 -> (N, T, hidden_size)

        cond = self.cond_proj(cond.transpose(1, 2).contiguous())  # 前缀条件同样转为 (N, T', hidden_size)
        prefix = cond.size(1)  # 记录前缀条件长度 T'，便于之后剥离其对应输出

        current_time = self.time_embeddings(current_time).to(state.dtype)  # 当前时间步正弦编码 -> (N, hidden_size)
        current_time = self.time_mlp(current_time)  # 当前时间步 MLP 嵌入
        time_step = self.time_embeddings(time_step).to(state.dtype)  # 步长正弦编码 -> (N, hidden_size)
        time_step = self.delta_time_mlp(time_step)  # 步长 MLP 嵌入
        current_time = current_time + time_step  # 条件注入：将当前时刻与步长的嵌入相加，得到时间条件向量
        if w is not None:
            w = self.time_embeddings(w).to(state.dtype)  # 引导强度 w 的正弦编码 -> (N, hidden_size)
            w = self.cfg_mlp(w)  # w 的 MLP 嵌入
            current_time = current_time + w  # 与时间条件同路注入首 token（蒸馏单前向 CFG 的关键通路）

        state = torch.cat([(mu + current_time).unsqueeze(1), cond, state], dim=1)  # 拼接序列 [全局条件 token(1) | 前缀条件(T') | 当前状态(T)]，全局条件通过 mu+时间嵌入相加注入首 token
        hidden, _ = self.decoder(state, is_causal=False)  # 非因果 Transformer 编码（当前状态可关注全部条件前缀）
        hidden = hidden[:, prefix + 1 :, :]  # 剥离全局条件 token 与前缀条件，只保留当前状态对应位置的输出
        hidden = self.out_proj(hidden)  # (N, T, out_channels) 投影回隐变量维度

        return hidden.transpose(1, 2).contiguous()  # 恢复通道在前：(N, out_channels, T)
