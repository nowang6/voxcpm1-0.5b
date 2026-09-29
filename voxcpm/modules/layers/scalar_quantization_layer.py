import torch
import torch.nn as nn


class ScalarQuantizationLayer(nn.Module):
    """标量量化层：将连续隐变量投影到低维空间并做 tanh 压缩，再按固定步长进行标量量化。

    训练时使用直通估计（straight-through estimator），让梯度绕过不可导的取整操作；
    推理时直接输出量化结果，把连续隐变量离散化，便于压缩表示与下游建模。
    """

    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        """初始化标量量化层。

        参数:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            latent_dim: 中间量化空间的维度
            scale: 量化级数，tanh 输出区间 (-1, 1) 被划分为约 2*scale 个量化台阶
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.scale = scale

        self.in_proj = nn.Linear(in_dim, latent_dim)  # 输入投影：压缩到量化空间
        self.out_proj = nn.Linear(latent_dim, out_dim)  # 输出投影：从量化空间恢复到输出维度

    def forward(self, hidden):
        """对输入隐变量做标量量化。

        参数:
            hidden: 输入张量，形状 (B, T, in_dim)

        返回:
            量化并反投影后的张量，形状 (B, T, out_dim)
        """
        hidden = self.in_proj(hidden)  # (B, T, latent_dim) 投影到量化空间
        hidden = torch.tanh(hidden)  # 压缩到 (-1, 1)，便于按固定 scale 均匀量化

        if self.training:
            quantized = torch.round(hidden * self.scale) / self.scale  # 放大、取整、再缩回，得到量化网格点
            hidden = hidden + (quantized - hidden).detach()  # 直通估计：前向输出量化值，反向梯度直通未量化值
        else:
            hidden = torch.round(hidden * self.scale) / self.scale  # 推理时直接量化

        return self.out_proj(hidden)  # (B, T, out_dim) 投影回输出维度
