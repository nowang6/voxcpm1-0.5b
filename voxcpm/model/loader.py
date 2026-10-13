"""
VoxCPM1 (0.5B) 模型加载：从本地目录读取配置、分词器与权重。

从 OpenBMB/VoxCPM 抽取而来，仅保留单一架构（voxcpm v1 / VoxCPM-0.5B）。
"""

import os
import sys

import torch
from transformers import LlamaTokenizerFast

try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True  # 标记 safetensors 可用，权重加载时优先使用
except ImportError:
    SAFETENSORS_AVAILABLE = False

from ..modules.audiovae import AudioVAE
from .config import VoxCPMConfig
from .utils import get_dtype


def _load_checkpoint_state(safetensors_path: str, fallback_path: str, label: str) -> dict:
    """加载一份 checkpoint 的 state_dict：优先 safetensors，回退 torch 序列化文件。

    Args:
        safetensors_path: safetensors 权重路径（优先）。
        fallback_path: torch 序列化（.pth / .bin）权重路径（回退）。
        label: 权重名称，用于日志与报错信息（如 "AudioVAE"、"model"）。

    Returns:
        dict: 权重的 state_dict（自动解包 checkpoint 的 "state_dict" 键）。
    """
    if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
        print(f"Loading {label} from safetensors: {safetensors_path}", file=sys.stderr)
        return load_file(safetensors_path, device="cpu")
    if os.path.exists(fallback_path):
        print(f"Loading {label} from pytorch: {fallback_path}", file=sys.stderr)
        checkpoint = torch.load(
            fallback_path,
            map_location="cpu",
            weights_only=True,
        )
        return checkpoint.get("state_dict", checkpoint)
    raise FileNotFoundError(f"{label} checkpoint not found. Expected either {safetensors_path} or {fallback_path}")


def load_from_local(
    model_cls: type,
    path: str,
    optimize: bool = True,
    training: bool = False,
    device: str | None = None,
):
    """从本地目录加载模型（配置、分词器、AudioVAE 与主模型权重）。

    Args:
        model_cls: 模型类（通常传 VoxCPMModel 或其子类）。
        path: 模型目录，需包含 config.json、分词器文件、AudioVAE 权重与主模型权重。
        optimize: 是否启用 torch.compile 优化（仅推理模式生效）。
        training: 是否以训练模式加载（冻结 AudioVAE、不做 eval、保留 float32 精度设置）。
        device: 运行设备，``None`` 表示自动选择。

    Returns:
        加载完成的模型实例（类型即 model_cls）。
    """
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as _cfg_f:
        config = VoxCPMConfig.model_validate_json(_cfg_f.read())
    tokenizer = LlamaTokenizerFast.from_pretrained(path)

    # AudioVAE：按配置构建并加载权重（优先 safetensors，回退 .pth）
    audio_vae_config = getattr(config, "audio_vae_config", None)
    audio_vae = AudioVAE(config=audio_vae_config) if audio_vae_config else AudioVAE()
    vae_state_dict = _load_checkpoint_state(
        os.path.join(path, "audiovae.safetensors"),
        os.path.join(path, "audiovae.pth"),
        label="AudioVAE",
    )

    model = model_cls(config, tokenizer, audio_vae, device=device)
    if not training:
        # 推理模式：主模型整体转为低精度（如 bfloat16）以加速
        model = model.to(get_dtype(model.config.dtype))
    else:  # training mode: freeze the audio codec, train everything else
        # 训练模式：冻结音频编解码器（AudioVAE），其余参数全部可训练
        for name, param in model.named_parameters():
            if "audio_vae" in name:  # freeze VAE weights
                param.requires_grad = False
    # 将随机初始化的 prefix 参数迁移到运行设备/精度（与模型主体一致），确保保存/加载口径统一
    model.base_prefix.data = model.base_prefix.data.to(device=model.device, dtype=get_dtype(model.config.dtype))
    model.residual_prefix.data = model.residual_prefix.data.to(device=model.device, dtype=get_dtype(model.config.dtype))
    model.audio_vae = model.audio_vae.to(torch.float32)  # VAE 编解码保持 float32 以保证数值稳定

    # 主模型权重同样优先 safetensors，其次回退到 pytorch_model.bin
    model_state_dict = _load_checkpoint_state(
        os.path.join(path, "model.safetensors"),
        os.path.join(path, "pytorch_model.bin"),
        label="model",
    )

    # AudioVAE 权重单独存放，这里加 audio_vae. 前缀合并进完整 state_dict 后统一加载
    for kw, val in vae_state_dict.items():
        model_state_dict[f"audio_vae.{kw}"] = val

    # cfg_mlp 是 CFG 蒸馏新增模块：旧权重（预训练/非蒸馏产物）缺失其 4 个键是合法的，
    # 零初始化补齐后行为与原模型完全一致（w 嵌入输出恒 0）；其余缺失/多余键仍视为加载错误。
    # P-Tuning 说话人前缀（base_prefix/residual_prefix）同理：作为新增可学习参数，
    # 在预训练权重中不存在是合法的，加载后使用其自身的初始化值。
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    allowed_missing = {k for k in missing if "cfg_mlp" in k or "base_prefix" in k or "residual_prefix" in k}
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"Error(s) in loading state_dict: missing={bad_missing[:5]} unexpected={list(unexpected)[:5]}")
    if allowed_missing:
        model.feat_decoder.estimator.init_cfg_embedding_zero()
    if training:
        return model
    # 推理模式：搬到目标设备、切 eval，并按需启用 torch.compile 优化
    return model.to(model.device).eval().optimize(disable=not optimize)
