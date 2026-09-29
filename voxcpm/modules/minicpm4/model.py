from .config import MiniCPM4Config
import torch
import torch.nn as nn
from typing import List, Tuple
import math
from .cache import StaticKVCache


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    """RMSNorm 计算：在 float32 下按均方根归一化，再乘以可学习缩放权重。

    参数:
        hidden: 输入隐状态张量
        weight: 可学习缩放权重
        eps: 数值稳定小量

    返回:
        归一化后的张量（与输入同形状、同 dtype）
    """
    old_dtype = hidden.dtype  # 记录原始 dtype，计算完成后转回
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)  # float32 下求最后一维的均方，防止低精度下溢出
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)  # 除以均方根（乘 rsqrt）并恢复原 dtype
    return hidden * weight  # 逐元素乘可学习缩放权重


class MiniCPMRMSNorm(nn.Module):
    """RMSNorm 层（与 T5LayerNorm 等价）。"""

    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm 与 T5LayerNorm 等价。

        参数:
            hidden_size: 隐层维度
            eps: 数值稳定小量，默认 1e-6
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放权重，初始化为全 1
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """对输入隐状态做 RMSNorm 归一化。"""
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


def rotate_half(x):
    """旋转输入隐藏维度的一半（用于旋转位置编码 RoPE 的实现）。"""
    x1, x2 = x.chunk(2, dim=-1)  # 把最后一维对半分成 x1、x2
    return torch.cat((-x2, x1), dim=-1)  # 取负并交换顺序，对应二维旋转


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """将旋转位置编码（RoPE）应用到查询与键张量上。

    参数:
        q: 查询张量，形状 (batch_size, num_heads, seq_len, head_dim)
        k: 键张量，形状 (batch_size, num_key_value_heads, seq_len, head_dim)
        cos: 余弦缓存，形状 (seq_len, head_dim)
        sin: 正弦缓存，形状 (seq_len, head_dim)

    返回:
        应用了 RoPE 的 (q, k)，形状分别与输入相同
    """
    orig_dtype = q.dtype  # 记录原始 dtype（旋转在 float32 下计算以保证精度）
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # RoPE 公式：q' = q*cos + rotate_half(q)*sin
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)  # 恢复原始 dtype


class MiniCPMLongRoPE(nn.Module):
    """带动态 NTK 缩放的旋转位置编码（LongRoPE）：短/长上下文各用一套频率缩放系数以支持长度外推（致谢 Reddit 用户 /u/bloc97 与 /u/emozilla）。"""

    def __init__(self, config: MiniCPM4Config):
        """初始化 LongRoPE。

        参数:
            config: MiniCPM4 配置对象
        """
        super().__init__()
        self.config = config
        self.dim = config.kv_channels if config.kv_channels else config.hidden_size // config.num_attention_heads  # 每个注意力头的维度
        self.base = config.rope_theta  # RoPE 基频 theta
        self.max_position_embeddings = config.max_position_embeddings

        self.short_factor = config.rope_scaling.short_factor  # 短上下文（原始长度内）的频率缩放系数
        self.long_factor = config.rope_scaling.long_factor  # 长上下文（超原始长度）的频率外推系数
        self.original_max_position_embeddings = config.rope_scaling.original_max_position_embeddings

        scale = self.max_position_embeddings / self.original_max_position_embeddings  # 相对训练时原始长度的外扩倍数
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))  # 动态 NTK 缩放因子：随外扩倍数对数增长
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))  # 各维逆频率：1 / theta^(2i/d)
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册为非持久化缓冲区（不随模型权重保存）

        self.max_seq_len_cached = 0

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)  # cos 缓存，避免重复计算
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)  # sin 缓存，避免重复计算

        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)  # 初始化时预构建最大长度对应的缓存

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """设置 cos/sin 缓存：按给定序列长度构建各位置的旋转角度缓存表。"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)  # 位置索引 0..seq_len-1

        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)  # 超过原始训练长度：使用长上下文外推系数
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)  # 原始长度内：使用短上下文缩放系数

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device), self.inv_freq.to(device=device).to(dtype)
        )  # 位置 x（缩放后的逆频率），得到各位置各维的旋转角度

        # 创建embeddings
        emb = torch.cat((freqs, freqs), dim=-1)  # 复制拼接以匹配 head_dim（RoPE 中前后两半共享频率）

        self.cos_cached = emb.cos().to(dtype) * self.scaling_factor  # 计算 cos 并按 NTK 缩放因子缩放
        self.sin_cached = emb.sin().to(dtype) * self.scaling_factor  # 计算 sin 并按 NTK 缩放因子缩放

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据位置 id 查询缓存的 cos/sin。

        参数:
            position_ids: Tensor(seq_len) 或 Tensor(batch_size, seq_len)
        返回:
            Tensor(seq_len, head_dim), Tensor(seq_len, head_dim)
        """
        cos = self.cos_cached[position_ids]  # 按位置索引查表
        sin = self.sin_cached[position_ids]

        return cos, sin


class MiniCPMAttention(nn.Module):
    """多头自注意力（支持分组查询注意力 GQA），提供整段前向与单步解码两条路径。"""

    def __init__(self, config: MiniCPM4Config, layer_idx: int):
        """初始化注意力层。

        参数:
            config: MiniCPM4 配置
            layer_idx: 当前层编号
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )  # 每头维度：优先取 kv_channels，否则按 hidden_size/头数 计算
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # GQA 分组数：每多少个查询头共享一组 KV
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = 10000.0  # RoPE 基频（此处为默认值 10000）

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)  # 查询投影（无 bias）
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 键投影（GQA：KV 头数更少）
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 值投影
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)  # 输出投影

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """整段前向（训练或预填充阶段）。

        参数:
            hidden_states: 输入隐状态，形状 (batch_size, seq_len, hidden_size)
            position_emb: 旋转位置编码 (cos, sin) 元组，可为 None（禁用 RoPE 时）
            is_causal: 是否使用因果注意力掩码

        返回:
            注意力输出，形状 (batch_size, seq_len, num_heads*head_dim)
            本层 (K, V) 元组（供外部 KV 缓存使用）
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)  # Q 投影
        key_states = self.k_proj(hidden_states)  # K 投影
        value_states = self.v_proj(hidden_states)  # V 投影

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # 拆分多头：(B, L, H, D) -> (B, H, L, D)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 对 Q/K 应用旋转位置编码 RoPE

        # ref: https://github.com/pytorch/pytorch/issues/163597
        # there is a bug in MPS for non-contiguous tensors, so we need to make them contiguous
        query_states = query_states.contiguous()  # MPS 后端对非连续张量有 bug，需先转为连续
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=is_causal,  # 因果掩码：每个位置只能关注自身及之前的 token
            enable_gqa=True,  # 查询头多于 KV 头时自动广播（分组查询注意力）
        )

        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, H, L, D) -> (B, L, H, D)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)  # 合并多头 -> (B, L, H*D)

        attn_output = self.o_proj(attn_output)  # 输出投影

        past_key_value = (key_states, value_states)  # 返回本层 K/V，供 KV 缓存使用
        return attn_output, past_key_value

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        position_id: int,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """单步解码前向（配合静态 KV 缓存，一次只处理一个 token）。

        参数:
            hidden_states: 当前 token 的隐状态，形状 (batch_size, hidden_size)
            position_emb: 当前位置的旋转位置编码 (cos, sin)
            position_id: 当前写入 KV 缓存的位置下标
            kv_cache: 本层的 (K, V) 缓存张量

        返回:
            注意力输出，形状 (batch_size, num_heads*head_dim)
        """
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)  # Q 投影（序列长度为 1）
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 1, H, D) -> (B, H, 1, D)
        key_states = key_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 应用当前位置的 RoPE

        key_cache, value_cache = kv_cache

        key_cache[:, :, position_id, :] = key_states  # 将当前 K 写入静态缓存的对应位置
        value_cache[:, :, position_id, :] = value_states  # 将当前 V 写入静态缓存的对应位置

        # Use an explicit broadcastable mask shape for SDPA. A 1D mask can
        # trigger a CPU-side dimension bug in some PyTorch versions.
        attn_mask = (torch.arange(key_cache.size(2), device=key_cache.device) <= position_id).view(1, 1, 1, -1)  # 因果掩码：仅允许关注 [0, position_id] 的历史位置

        # ref: https://github.com/pytorch/pytorch/issues/163597
        # there is a bug in MPS for non-contiguous tensors, so we need to make them contiguous
        query_states = query_states.contiguous()  # MPS 后端对非连续张量有 bug，需先转为连续
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_cache,
            value_cache,
            attn_mask=attn_mask,  # 使用显式掩码（is_causal 不适用于 query 与 key 长度不等的情况）
            enable_gqa=True,
        )  # query 长度为 1，key/value 为整段缓存

        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, H, 1, D) -> (B, 1, H, D)
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)  # 合并多头 -> (B, H*D)

        attn_output = self.o_proj(attn_output)

        return attn_output


class MiniCPMMLP(nn.Module):
    """前馈网络（SwiGLU 式 FFN）：down(silu(gate(x)) * up(x))。"""

    def __init__(self, config):
        """初始化前馈网络。

        参数:
            config: MiniCPM4 配置
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # 门控分支
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # 上分支
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # 降维输出分支
        self.act_fn = nn.SiLU()  # SiLU 激活函数

    def forward(self, x):
        """FFN 前向：门控分支激活后与上分支逐元素相乘，再投影回隐层维度。"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMDecoderLayer(nn.Module):
    """单个 Transformer 解码器层：RMSNorm -> 自注意力 -> 残差（可选 muP 缩放）-> RMSNorm -> FFN -> 残差。"""

    def __init__(self, config: MiniCPM4Config, layer_idx: int):
        """初始化解码器层。

        参数:
            config: MiniCPM4 配置
            layer_idx: 层编号
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniCPMAttention(config=config, layer_idx=layer_idx)

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力前 RMSNorm（pre-norm）
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # FFN 前 RMSNorm

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.use_mup = config.use_mup

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        解码器层整段前向。

        参数:
            hidden_states (`torch.FloatTensor`): 层输入，形状 `(batch, seq_len, embed_dim)`
            position_emb: 旋转位置编码 (cos, sin)
            is_causal (`bool`): 是否使用因果注意力掩码

        返回:
            层输出与本层 (K, V) 缓存元组
        """
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)  # 注意力前 RMSNorm
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_emb=position_emb,
            is_causal=is_causal,
        )

        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))  # muP 残差缩放：scale_depth/sqrt(层数)，稳定深层训练
        else:
            hidden_states = residual + hidden_states  # 普通残差连接

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # FFN 前 RMSNorm

        hidden_states = self.mlp(hidden_states)
        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))  # muP 残差缩放
        else:
            hidden_states = residual + hidden_states

        return hidden_states, present_key_value

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        position_id: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """解码器层单步前向（配合静态 KV 缓存逐 token 解码）。

        参数:
            hidden_states: 当前 token 隐状态，形状 (batch, hidden_size)
            position_emb: 当前位置的旋转位置编码 (cos, sin)
            position_id: 当前写入位置下标
            kv_cache: 本层 (K, V) 缓存

        返回:
            形状 (batch, hidden_size) 的层输出
        """
        residual = hidden_states  # 保存残差
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn.forward_step(
            hidden_states=hidden_states,
            position_emb=position_emb,
            position_id=position_id,
            kv_cache=kv_cache,
        )

        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))  # muP 残差缩放
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))  # muP 残差缩放
        else:
            hidden_states = residual + hidden_states

        return hidden_states


class MiniCPMModel(nn.Module):
    """
    Transformer 解码器，由 *config.num_hidden_layers* 层堆叠而成，每层为 [`MiniCPMDecoderLayer`]。

    参数:
        config: MiniCPM4 配置
    """

    def __init__(self, config: MiniCPM4Config):
        """初始化 MiniCPM 主干模型。

        参数:
            config: MiniCPM4 配置
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.config = config

        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 有词表时使用词嵌入
        else:
            self.embed_tokens = nn.Identity()  # vocab_size=0 时直接透传连续输入（本项目中 DiT/编码器输入为连续向量）

        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )  # 堆叠 num_hidden_layers 个解码器层

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终 RMSNorm
        if config.no_rope:
            self.rope_emb = None  # 可选完全关闭 RoPE
        else:
            self.rope_emb = MiniCPMLongRoPE(config)

        self.kv_cache = None  # 静态 KV 缓存（setup_cache 时创建）

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        整段前向（训练或预填充阶段）。

        参数:
            inputs_embeds: 输入嵌入，形状 (batch_size, seq_length, hidden_size)
            is_causal: 是否使用因果注意力掩码
        返回:
            hidden_states: 形状 (batch_size, seq_length, hidden_size)
            next_decoder_cache: 每层的 (K, V) 缓存列表，元素形状为 (batch_size, num_heads, seq_length, head_dim)
        """
        if self.rope_emb is not None:
            position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device)  # 生成位置 id 0..L-1
            position_emb = self.rope_emb(position_ids)  # 查表得到 (cos, sin)
        else:
            position_emb = None
        hidden_states = inputs_embeds

        next_decoder_cache = []  # 收集各层 KV 供外部缓存

        for decoder_layer in self.layers:

            hidden_states, this_cache = decoder_layer(
                hidden_states,
                position_emb,
                is_causal,
            )  # 逐层前向
            next_decoder_cache.append(this_cache)
        hidden_states = self.norm(hidden_states)  # 最终 RMSNorm
        return hidden_states, next_decoder_cache

    def forward_step(
        self,
        inputs_embeds: torch.Tensor,
        position_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步前向（逐 token 解码，需先调用 setup_cache）。

        参数:
            inputs_embeds: 当前 token 的输入嵌入，形状 (batch_size, hidden_size)
            position_id: 当前位置下标（用于 RoPE 与 KV 缓存写入）
        返回:
            hidden_states: 形状 (batch_size, hidden_size) 的输出
        """
        assert self.kv_cache is not None, "KV cache is not setup"  # 必须先设置 KV 缓存

        if self.rope_emb is not None:
            position_emb = self.rope_emb(position_id)  # 当前位置的位置编码 (cos, sin)
        else:
            position_emb = None
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer.forward_step(
                hidden_states,
                position_emb,
                position_id,
                self.kv_cache.get_layer_cache(i),  # 取第 i 层静态缓存
            )

        hidden_states = self.norm(hidden_states)  # 最终 RMSNorm
        return hidden_states

    def setup_cache(self, batch_size: int, max_length: int, device, dtype: torch.dtype):
        """创建静态 KV 缓存（推理前调用）。

        参数:
            batch_size: 批大小
            max_length: 缓存支持的最大序列长度
            device: 缓存所在设备
            dtype: 缓存数据类型
        """
        self.kv_cache = StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            dim_kv_head=(
                self.config.hidden_size // self.config.num_attention_heads
                if self.config.kv_channels is None
                else self.config.kv_channels
            ),  # 每个 KV 头的维度
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
