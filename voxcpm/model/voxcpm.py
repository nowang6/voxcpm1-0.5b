"""
VoxCPM1 (0.5B)：免分词器（Tokenizer-free）的语音生成模型。

本文件只包含核心模型类（结构搭建、训练前向、编译优化与加载入口）；
配置定义在 config.py，生成/推理逻辑在 generation.py（Mixin），
权重加载在 loader.py。

从 OpenBMB/VoxCPM 抽取而来，仅保留单一架构（voxcpm v1 / VoxCPM-0.5B）
且不含 LoRA：只支持全参数微调。
"""

import sys

import torch
import torch.nn as nn
from einops import rearrange
from transformers import LlamaTokenizerFast

from ..modules.audiovae import AudioVAE
from ..modules.layers import ScalarQuantizationLayer
from ..modules.locdit import UnifiedCFM, VoxCPMLocDiT
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPMModel
from .config import VoxCPMConfig, VoxCPMDitConfig, VoxCPMEncoderConfig  # noqa: F401  (re-export，保持旧导入路径兼容)
from .generation import VoxCPMGenerationMixin
from .loader import load_from_local
from .utils import (
    get_dtype,
    mask_multichar_chinese_tokens,
    pick_runtime_dtype,
    resolve_runtime_device,
)


class VoxCPMModel(VoxCPMGenerationMixin, nn.Module):
    """VoxCPM1 (0.5B) 核心模型：文本-语义 LM + 残差声学 LM + 局部编码器 + 局部 DiT 流匹配解码器。

    生成与推理相关方法见 VoxCPMGenerationMixin（generation.py）。
    """

    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
        device: str | None = None,
    ):
        """初始化模型：按配置搭建各子模块并分配设备与计算精度。

        Args:
            config: VoxCPM 整体配置。
            tokenizer: 文本分词器（会被包装为按字切分中文多字词的形式）。
            audio_vae: 音频变分自编码器（AudioVAE），用于音频与隐变量之间的互转。
            device: 运行设备；``None`` 表示自动选择。
        """
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        # 解析实际运行设备（None/auto 时按 CUDA -> MPS -> CPU 顺序自动选择）
        self.device = resolve_runtime_device(device, config.device)
        self.config.device = self.device
        # 按设备选择安全的计算精度（例如 MPS 上低精度会导致扩散自回归输出劣化，强制 float32）
        resolved_dtype = pick_runtime_dtype(self.device, self.config.dtype)
        if resolved_dtype != self.config.dtype:
            print(
                f"[voxcpm] adjusted dtype {self.config.dtype} -> {resolved_dtype} for device {self.device}",
                file=sys.stderr,
            )
            self.config.dtype = resolved_dtype
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}", file=sys.stderr)

        # Text-Semantic LM
        # 文本-语义语言模型：统一处理文本 token 与音频 patch 的自回归建模
        self.base_lm = MiniCPMModel(config.lm_config)
        # 为 batch=1、最大长度 max_length 的自回归生成预分配 KV 缓存
        self.base_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # 包装分词器：将多字中文词切分为单字，保证中文分词粒度一致
        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101  # 音频段起始的特殊 token id
        self.audio_end_token = 102  # 音频段结束的特殊 token id
        self.last_successful_seed = None  # 记录最近一次成功生成所用的随机种子

        # Residual Acoustic LM
        # 残差声学语言模型：以基础 LM 的输出为条件，补充建模声学细节
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0  # 不使用词表（纯隐变量输入）
        self.residual_lm = MiniCPMModel(residual_lm_config)
        self.residual_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # Local Encoder
        # 局部编码器：将音频 patch 特征编码为与 LM 同维的嵌入
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0  # 不使用词表
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        # 局部 DiT 解码器：基于流匹配（CFM）把 LM 隐状态渲染为音频 patch 特征
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0  # 不使用词表
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiT(decoder_config, in_channels=config.feat_dim),
            mean_mode=config.dit_mean_mode,
        )

        # Projection layers
        # 投影层：对齐各子模块之间的隐藏维度
        # 标量量化层（FSQ）：对音频位置的 LM 隐状态做离散量化，为残差声学 LM 提供更稳定的条件
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)  # 编码器 -> LM 维度
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)  # 基础 LM -> DiT 维度
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)  # 残差 LM -> DiT 维度

        # Stop Predictor
        # 停止预测头：根据 LM 隐状态判断生成是否应结束（二分类：继续/停止）
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)  # 输出 2 个 logits（继续/停止）
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")  # 逐位置交叉熵，稍后自行加权求平均

        # Audio VAE
        # 音频 VAE：编码器把波形压成隐变量，解码器把隐变量还原为波形
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size  # VAE 每次解码的 chunk 大小（patch 数）
        self.sample_rate = audio_vae.sample_rate  # VAE 采样率（16000 Hz）

    def optimize(self, disable: bool = False):
        """使用 torch.compile 编译热点子模块以加速推理。

        仅在 CUDA 设备且已安装 triton 时生效；失败时优雅回退（打印警告并保持原模型）。

        Args:
            disable: 为 True 时直接跳过编译，原样返回模型。

        Returns:
            VoxCPMModel: 自身（便于链式调用）。
        """
        if disable:
            return self
        try:
            if self.device != "cuda":
                raise ValueError("VoxCPMModel can only be optimized on CUDA device")
            try:
                import triton  # noqa: F401
            except ImportError:
                raise ValueError("triton is not installed")
            # reduce-overhead 模式利用 CUDA Graph 消除逐次调用开销；
            # fullgraph=True 要求整图编译以获得最大收益
            # 逐 token 解码的热点路径：两个 LM 的单步前向
            self.base_lm.forward_step = torch.compile(self.base_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.residual_lm.forward_step = torch.compile(
                self.residual_lm.forward_step, mode="reduce-overhead", fullgraph=True
            )
            # 保留未编译的局部编码器引用：prefill（整段前向）时形状多变，用原始版本更稳妥
            self._feat_encoder_raw = self.feat_encoder
            self.feat_encoder = torch.compile(self.feat_encoder, mode="reduce-overhead", fullgraph=True)
            self.feat_decoder.estimator = torch.compile(
                self.feat_decoder.estimator, mode="reduce-overhead", fullgraph=True
            )
        except Exception as e:
            print(f"Warning: torch.compile disabled - {e}", file=sys.stderr)
        return self

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        progress: float = 0.0,
        sample_generate: bool = False,
    ):
        """训练前向：计算流匹配扩散损失（loss/diff）与停止预测损失（loss/stop）。

        序列布局为「文本段 + 音频段」拼接：文本位置由 text_mask 标记（audio_mask 为 0），
        音频 patch 位置由 audio_mask 标记（text_mask 为 0），二者互斥。

        Args:
            text_tokens: 文本 token id，形状 (B, T)。
            text_mask: 文本位置掩码，形状 (B, T)，文本位置为 1、音频位置为 0。
            audio_feats: 音频 patch 特征，形状 (B, T, P, D)，P=patch_size、D=隐变量维度。
            audio_mask: 音频位置掩码，形状 (B, T)，音频位置为 1、文本位置为 0。
            loss_mask: 损失掩码，形状 (B, T)，仅对有效目标位置计损失（如排除提示段）。
            position_ids: 位置 id（当前未使用）。
            labels: 停止预测标签，形状 (B, T)，1 表示该位置后应停止生成。
            progress: 训练进度（0~1），用于调整流匹配损失的课程（progressive training）。
            sample_generate: 是否额外执行一次流匹配采样得到 feat_pred（供验证试听）。

        Returns:
            dict: 包含 ``loss/diff``（流匹配损失）、``loss/stop``（停止损失）、
                ``feat_gt``（真实音频特征，(B, D, T*P)）与 ``feat_pred``（采样特征或 None）。
        """
        del position_ids  # not used yet

        # 统一把各输入搬到目标设备与精度
        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=self._dtype())
        audio_feats = audio_feats.to(self.device, dtype=self._dtype())
        audio_mask = audio_mask.to(self.device, dtype=self._dtype())
        loss_mask = loss_mask.to(self.device, dtype=self._dtype())
        labels = labels.to(self.device, dtype=torch.long)

        B, T, P, D = audio_feats.shape  # (batch, 序列长度, patch 大小, 特征维度)
        feat_embed = self.feat_encoder(audio_feats)  # (B, T, hidden)
        feat_embed = self.enc_to_lm_proj(feat_embed)  # (B, T, lm_hidden) 对齐到 LM 维度

        # muP 训练时按 sqrt(hidden) 缩放嵌入；未启用 muP 则不缩放
        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.base_lm.embed_tokens(text_tokens) * scale_emb  # (B, T, lm_hidden)
        # 按 mask 融合两种嵌入：文本位置取文本嵌入、音频位置取音频嵌入
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

        # 基础 LM 因果前向（teacher forcing，一次看完整序列）
        enc_outputs, _ = self.base_lm(inputs_embeds=combined_embed, is_causal=True)
        enc_outputs = enc_outputs.to(self._dtype())
        # 音频位置走 FSQ 标量量化，文本位置保持连续隐状态（量化的离散性利于声学建模）
        enc_outputs = self.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        # 右移一位：t 位置的预测条件只包含 t-1 及之前的信息，保证因果性
        lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)

        # 残差声学 LM：输入为量化后的隐状态（文本位置）加上原始音频嵌入（音频位置）
        residual_inputs = enc_outputs + audio_mask.unsqueeze(-1) * feat_embed
        residual_outputs, _ = self.residual_lm(inputs_embeds=residual_inputs, is_causal=True)
        residual_outputs = residual_outputs.to(self._dtype())
        # 同样右移一位，保持因果性
        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        # 双路条件相加后投影到 DiT 维度，作为流匹配的速度场估计条件
        dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")  # (B*T, dit_hidden)，逐 patch 独立求解

        # Keep diffusion inputs in the same dtype as the model (e.g., bfloat16)
        # 扩散输入与模型保持同一精度（如 bfloat16）
        target_dtype = self._dtype()

        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")  # 目标特征 (B*T, P, D)
        # 条件特征 = 上一个 patch（右移一位）：首个位置用零填充，保证以已生成的历史为条件
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        # 把 patch 级 loss_mask 展开到 patch 内每一帧，并变形为 (B*T, P, 1)
        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        # 流匹配（CFM）条件流匹配损失：预测速度场 v(x_t, t | cond)
        diff_loss = self.feat_decoder.compute_loss(
            feat_gt.transpose(1, 2).contiguous(),  # (B*T, D, P)，转置为通道优先
            dit_hidden,
            cond=feat_cond.transpose(1, 2).contiguous(),
            tgt_mask=loss_seq_mask.transpose(1, 2).contiguous(),
            progress=progress,
        )

        # 停止预测头：由右移后的 LM 隐状态预测每个位置的停止概率
        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))  # (B, T, 2)
        stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)  # 逐位置交叉熵 (B, T)
        # 仅对有效位置（loss_mask=1）求平均，min=1 防止除零
        denom = torch.clamp(loss_mask.sum(), min=1.0)
        stop_loss = (stop_losses * loss_mask).sum() / denom

        feat_pred = None
        if sample_generate:
            # 可选：用流匹配 ODE 求解实际采样一个特征序列（验证时生成样本音频用）
            feat_cond_for_sample = feat_cond.transpose(1, 2).contiguous()
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond_for_sample,
                n_timesteps=(
                    self.config.dit_config.cfm_config.inference_cfg_rate
                    if hasattr(self.config.dit_config.cfm_config, "inference_cfg_rate")
                    else 10
                ),
            )
            feat_pred = rearrange(feat_pred_seq.transpose(1, 2), "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        # 真实特征同样重排为 (B, D, T*P)，与 feat_pred 对齐方便对比
        feat_gt_tensor = rearrange(feat_gt, "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        return {
            "loss/diff": diff_loss,
            "loss/stop": stop_loss,
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
        }

    def _dtype(self):
        """返回配置字符串对应的 torch 数据类型对象。"""
        return get_dtype(self.config.dtype)

    @classmethod
    def from_local(
        cls,
        path: str,
        optimize: bool = True,
        training: bool = False,
        device: str | None = None,
    ):
        """从本地目录加载模型（配置、分词器、AudioVAE 与主模型权重）。

        具体加载流程见 loader.load_from_local。

        Args:
            path: 模型目录，需包含 config.json、分词器文件、AudioVAE 权重与主模型权重。
            optimize: 是否启用 torch.compile 优化（仅推理模式生效）。
            training: 是否以训练模式加载（冻结 AudioVAE、不做 eval、保留 float32 精度设置）。
            device: 运行设备，``None`` 表示自动选择。

        Returns:
            VoxCPMModel: 加载完成的模型实例。
        """
        return load_from_local(cls, path, optimize=optimize, training=training, device=device)
