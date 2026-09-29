#!/usr/bin/env python3
"""VoxCPM1-0.5B 全参数微调脚本。

用法：
    python train.py --config conf/finetune_example.yaml
    python train.py --pretrained_path /data/models/VoxCPM-0.5B --train_manifest data/train.jsonl ...

任何命令行参数都会覆盖 YAML 配置文件中的同名配置项。

多卡训练（Multi-GPU）：
    torchrun --nproc_per_node=N train.py --config conf/finetune_example.yaml
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

# 避免 tokenizers 并行 fork 与 DataLoader 多进程 worker 冲突导致死锁
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tensorboardX import SummaryWriter
from safetensors.torch import save_file, load_file

from voxcpm.model import VoxCPMModel
from voxcpm.training import (
    Accelerator,
    BatchProcessor,
    TrainingTracker,
    build_dataloader,
    load_audio_text_datasets,
)

try:
    import yaml
except ImportError:
    yaml = None  # pyyaml 未安装时 --config 选项不可用


def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 训练所需的全部超参（data / 优化器 / 日志与保存三组）。
        未显式传入的参数值为 None，由 load_config 用默认值或 YAML 配置补齐。
    """
    parser = argparse.ArgumentParser(description="VoxCPM1-0.5B full-parameter fine-tuning")
    parser.add_argument("--config", type=str, default="", help="YAML config file; CLI args override it")
    # data
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=None, help="Must match the AudioVAE encoder rate (16000)")
    parser.add_argument("--max_batch_tokens", type=int, default=None,
                        help="If >0, drop samples longer than max_batch_tokens / batch_size to avoid OOM")
    # optimization
    parser.add_argument("--batch_size", type=int, default=None)  # 每卡 batch 大小
    parser.add_argument("--grad_accum_steps", type=int, default=None)  # 梯度累积步数，用于等效扩大 batch
    parser.add_argument("--num_workers", type=int, default=None)  # DataLoader 加载数据的子进程数
    parser.add_argument("--num_iters", type=int, default=None)  # 总训练步数（optimizer step 数）
    parser.add_argument("--learning_rate", type=float, default=None)  # 峰值学习率
    parser.add_argument("--weight_decay", type=float, default=None)  # 权重衰减系数
    parser.add_argument("--warmup_steps", type=int, default=None)  # 学习率预热步数
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Gradient clipping norm; 0 disables")
    parser.add_argument("--lambda_diff", type=float, default=None, help="Weight for loss/diff")  # 流匹配扩散损失权重
    parser.add_argument("--lambda_stop", type=float, default=None, help="Weight for loss/stop")  # 停止预测损失权重
    parser.add_argument("--freeze_prefixes", type=str, nargs="*", default=None,
                        help="Param-name prefixes to freeze (audio_vae is always frozen). "
                             "e.g. --freeze_prefixes base_lm. fsq_layer. to train only the acoustic side")
    # logging / saving
    parser.add_argument("--log_interval", type=int, default=None)  # 每隔多少步打印一次训练日志
    parser.add_argument("--valid_interval", type=int, default=None)  # 每隔多少步做一次验证
    parser.add_argument("--save_interval", type=int, default=None)  # 每隔多少步保存一次 checkpoint
    parser.add_argument("--save_path", type=str, default=None)  # checkpoint 保存目录
    parser.add_argument("--tensorboard", type=str, default=None)  # TensorBoard 日志目录
    parser.add_argument("--num_val_batches", type=int, default=None, help="Validation batches per validation round")
    parser.add_argument("--num_audio_samples", type=int, default=None, help="Sample audios logged to TensorBoard")
    parser.add_argument("--seed", type=int, default=None)  # 随机种子，保证训练可复现
    return parser.parse_args()


def load_config(args) -> dict:
    """将默认值、YAML 配置与命令行参数合并为最终训练配置。

    合并优先级（从低到高）：内置默认值 < YAML 配置文件 < 显式传入的命令行参数。

    Args:
        args: parse_args() 返回的命令行参数命名空间。

    Returns:
        dict: 完整训练配置，其中额外包含 ``lambdas`` 字段，
        即训练循环使用的各项损失权重 ``{"loss/diff": ..., "loss/stop": ...}``。
    """
    defaults = {
        "pretrained_path": "",
        "train_manifest": "",
        "val_manifest": "",
        "sample_rate": 16_000,
        "max_batch_tokens": 0,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "num_workers": 2,
        "num_iters": 10_000,
        "learning_rate": 1e-5,
        "weight_decay": 1e-2,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "lambda_diff": 1.0,
        "lambda_stop": 1.0,
        "freeze_prefixes": [],
        "log_interval": 10,
        "valid_interval": 500,
        "save_interval": 1000,
        "save_path": "checkpoints/finetune_all",
        "tensorboard": "",
        "num_val_batches": 10,
        "num_audio_samples": 2,
        "seed": 42,
    }

    cfg = dict(defaults)
    if args.config:
        if yaml is None:
            raise RuntimeError("--config requires pyyaml")
        # 读取 YAML 配置文件并覆盖默认值（必须是顶层映射结构）
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        if not isinstance(yaml_cfg, dict):
            raise ValueError(f"Config file {args.config} must contain a top-level mapping.")
        cfg.update(yaml_cfg)

    # CLI overrides (only the ones explicitly passed)
    # 命令行覆盖：仅当参数被显式传入（值非 None）时才覆盖
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value

    if not cfg["pretrained_path"] or not cfg["train_manifest"]:
        raise ValueError("pretrained_path and train_manifest are required (CLI or YAML config)")

    # lambdas used by the training loop
    # 供训练循环使用的各项损失权重（流匹配损失 + 停止预测损失）
    cfg["lambdas"] = {"loss/diff": cfg["lambda_diff"], "loss/stop": cfg["lambda_stop"]}
    return cfg


# --------------------------------------------------------------------------- #
# Checkpointing（checkpoint 保存与恢复）
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    step: int,
    pretrained_path: str,
):
    """将非 VAE 权重保存为 model.safetensors，同时保存优化器/调度器状态。

    保存出的文件夹可直接被 ``VoxCPM(voxcpm_model_path=...)`` 加载使用。

    Args:
        model: （可能被 DDP 包装的）训练中的模型。
        optimizer: 优化器，其状态用于断点续训。
        scheduler: 学习率调度器，其状态用于断点续训。
        save_dir: checkpoint 根目录，每个 step 保存为子文件夹。
        step: 当前训练步数。
        pretrained_path: 预训练模型目录，用于复制配置文件使 checkpoint 自包含。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    folder = save_dir / f"step_{step:07d}"  # 按 step 命名的 checkpoint 子目录
    folder.mkdir(parents=True, exist_ok=True)

    # 剥掉 DDP 的 .module 包装，拿到原始模型
    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()
    # AudioVAE 在训练中始终冻结，剔除其权重以减小 checkpoint 体积
    state_dict = {k: v for k, v in full_state.items() if not k.startswith("audio_vae.")}
    save_file(state_dict, folder / "model.safetensors")

    # Copy config files from pretrained path so the folder is self-contained
    pretrained_dir = Path(pretrained_path)
    for fname in [
        "config.json",
        "audiovae.pth",
        "audiovae.safetensors",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]:
        src = pretrained_dir / fname
        if src.exists():
            shutil.copy2(src, folder / fname)

    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")
    # 记录当前步数，用于断点续训时恢复训练进度
    with open(folder / "training_state.json", "w", encoding="utf-8") as f:
        json.dump({"step": int(step)}, f)

    # Update (or create) a `latest` folder by copying the most recent checkpoint
    latest_link = save_dir / "latest"
    try:
        if latest_link.exists():
            shutil.rmtree(latest_link)
        shutil.copytree(folder, latest_link)
    except Exception:
        print(f"Warning: failed to update latest checkpoint at {latest_link}", file=sys.stderr)


def load_checkpoint(model, optimizer, scheduler, save_dir: Path, rank: int = 0) -> int:
    """若存在最新的 checkpoint 则加载（模型/优化器/调度器状态），并返回续训起始步。

    Args:
        model: （可能被 DDP 包装的）训练中的模型。
        optimizer: 待恢复状态的优化器。
        scheduler: 待恢复状态的学习率调度器。
        save_dir: checkpoint 根目录，从其中的 ``latest`` 子目录加载。
        rank: 当前进程的分布式 rank，仅 rank 0 打印日志。

    Returns:
        int: 续训起始步；无 checkpoint 时返回 0（从头开始）。
    """
    latest_folder = save_dir / "latest"
    if not latest_folder.exists():
        return 0  # 没有任何 checkpoint，从第 0 步开始训练

    # 剥掉 DDP 的 .module 包装，拿到原始模型
    unwrapped = model.module if hasattr(model, "module") else model

    model_path = latest_folder / "model.safetensors"
    if model_path.exists():
        state_dict = load_file(str(model_path))
        unwrapped.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"Loaded model weights from {model_path}", file=sys.stderr)

    optimizer_path = latest_folder / "optimizer.pth"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"Loaded optimizer state from {optimizer_path}", file=sys.stderr)

    scheduler_path = latest_folder / "scheduler.pth"
    if scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"Loaded scheduler state from {scheduler_path}", file=sys.stderr)

    state_path = latest_folder / "training_state.json"
    if state_path.exists():
        # 读取训练进度（步数）作为续训起点
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        resume_step = int(state.get("step", 0))
        if rank == 0:
            print(f"Resuming from step {resume_step}", file=sys.stderr)
        return resume_step

    return 0


# --------------------------------------------------------------------------- #
# Validation（验证）
# --------------------------------------------------------------------------- #
def validate(model, val_loader, batch_processor, accelerator, tracker, lambdas, num_val_batches):
    """在验证集上计算验证损失（流匹配损失 diff + 停止预测损失 stop）。

    Args:
        model: （分布式包装的）训练模型。
        val_loader: 验证集 DataLoader。
        batch_processor: 将原始 batch 加工成模型 forward 输入的处理器。
        accelerator: 分布式/混合精度加速器封装。
        tracker: 训练日志记录器。
        lambdas: 各项损失的权重字典 ``{"loss/diff": ..., "loss/stop": ...}``。
        num_val_batches: 本轮验证使用的 batch 数量上限。
    """
    model.eval()
    total_losses = []
    sub_losses = {}
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= num_val_batches:
                break
            processed = batch_processor(batch)
            # 与训练一致使用 bfloat16 自动混合精度做前向（验证时 progress 固定为 0.0）
            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                )
            total = 0.0
            # 累加各项带权重的子损失（loss/diff 与 loss/stop）
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    total = total + lambdas.get(key, 1.0) * value
                    sub_losses.setdefault(key, []).append(value.detach())
            total_losses.append(total.detach())
            num_batches += 1

    if total_losses:
        # 对所有 batch 取平均，再跨进程 all_reduce 求全局平均
        mean_total_loss = torch.stack(total_losses).mean()
        accelerator.all_reduce(mean_total_loss)

        val_metrics = {"loss/total": mean_total_loss.item()}
        for key, values in sub_losses.items():
            mean_sub_loss = torch.stack(values).mean()
            accelerator.all_reduce(mean_sub_loss)
            val_metrics[key] = mean_sub_loss.item()

        tracker.log_metrics(val_metrics, split="val")

    model.train()  # 恢复训练模式


def generate_sample_audio(model, val_ds, val_texts, writer, step, accelerator, sample_rate, num_samples, tracker):
    """为若干固定的验证样本生成音频，并写入 TensorBoard 供试听。

    Args:
        model: （分布式包装的）训练模型。
        val_ds: 验证数据集（用于确定可用样本数）。
        val_texts: 验证集文本列表，作为生成目标文本。
        writer: TensorBoard SummaryWriter。
        step: 当前训练步数，作为日志的 global_step。
        accelerator: 分布式/混合精度加速器封装。
        sample_rate: 音频采样率（写入 TensorBoard 时使用）。
        num_samples: 期望生成的样本音频数量。
        tracker: 训练日志记录器。
    """
    import numpy as np

    log = tracker.print
    unwrapped = accelerator.unwrap(model)
    num_samples = min(num_samples, len(val_ds))
    log(f"[Audio] Generating {num_samples} sample audios at step {step}")

    prev_training = unwrapped.training  # 记录原始模式，结束后恢复
    try:
        unwrapped.eval()  # 生成音频需切换到推理模式（关闭 dropout 等）
        for i in range(num_samples):
            text = val_texts[i] if val_texts and i < len(val_texts) else "Hello, this is a test."
            try:
                with torch.no_grad():
                    # CUDA 上使用 bfloat16 自动混合精度生成，CPU 上则不加 autocast
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if torch.cuda.is_available()
                        else contextlib.nullcontext()
                    )
                    with autocast_ctx:
                        # 固定推理步数 10、CFG 强度 2.0 与随机种子 42，便于跨步对比生成质量
                        generated = unwrapped.generate(target_text=text, inference_timesteps=10, cfg_value=2.0, seed=42)
                if generated is None or len(generated) == 0:
                    log(f"[Audio] Sample {i}: empty output, skip")
                    continue
                gen_audio_np = generated.cpu().float().numpy().flatten()
                max_val = np.abs(gen_audio_np).max()
                if max_val > 0:
                    gen_audio_np = gen_audio_np / max_val * 0.9  # 峰值归一化到 0.9，避免削波
                writer.add_audio(f"val_sample_{i}/generated", gen_audio_np, global_step=step, sample_rate=sample_rate)
                log(f"[Audio] Sample {i}: duration={len(gen_audio_np) / sample_rate:.2f}s, text='{text[:40]}...'")
            except Exception as e:
                log(f"[Warning] Failed to generate audio for sample {i}: {e}")
    finally:
        unwrapped.train() if prev_training else unwrapped.eval()  # 恢复进入函数前的训练/推理模式


# --------------------------------------------------------------------------- #
# Main（训练主流程）
# --------------------------------------------------------------------------- #
def train(cfg: dict):
    """微调主流程：初始化分布式与模型、准备数据与优化器，然后执行训练循环。

    Args:
        cfg: load_config() 生成的完整训练配置字典。
    """
    accelerator = Accelerator(amp=True, seed=cfg["seed"])  # 初始化加速器（AMP 混合精度 + 固定随机种子）
    lambdas = cfg["lambdas"]
    is_main = accelerator.rank == 0  # 仅主进程负责建目录、写日志、保存 checkpoint

    save_dir = Path(cfg["save_path"])
    tb_dir = Path(cfg["tensorboard"]) if cfg["tensorboard"] else save_dir / "logs"
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.barrier()  # 等所有进程都完成目录创建后再继续

    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main else None
    tracker = TrainingTracker(writer=writer, log_file=str(save_dir / "train.log"), rank=accelerator.rank)

    # ---------------- model ---------------- #
    if is_main:
        print(f"Loading pretrained model from {cfg['pretrained_path']}", file=sys.stderr)
    # 以训练模式加载预训练模型（from_local 内部已冻结 audio_vae，且不启用 torch.compile）
    base_model = VoxCPMModel.from_local(cfg["pretrained_path"], optimize=False, training=True)
    tokenizer = base_model.text_tokenizer

    # Full-parameter finetune: audio_vae is already frozen in from_local().
    # Optionally freeze extra modules via freeze_prefixes.
    # 全参数微调：audio_vae 已在 from_local() 中冻结；
    # 可通过 freeze_prefixes 追加冻结其他模块（实现部分参数微调）
    freeze_prefixes = ["audio_vae."] + list(cfg["freeze_prefixes"])
    n_frozen = n_total = 0
    for name, param in base_model.named_parameters():
        n_total += 1
        if any(name.startswith(pref) for pref in freeze_prefixes):
            param.requires_grad = False  # 命中冻结前缀的参数不参与梯度更新
            n_frozen += 1
    if is_main:
        # 统计并打印可训练参数量占比
        n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        n_total_params = sum(p.numel() for p in base_model.parameters())
        tracker.print(
            f"[freeze] prefixes={freeze_prefixes} | Frozen {n_frozen}/{n_total} tensors | "
            f"Trainable: {n_trainable / 1e6:.2f}M / {n_total_params / 1e6:.2f}M "
            f"({100.0 * n_trainable / n_total_params:.2f}%)"
        )

    # 校验配置的采样率与 AudioVAE 编码器要求的采样率一致（否则隐变量帧率不匹配）
    expected_sr = base_model.audio_vae.sample_rate
    assert cfg["sample_rate"] == expected_sr, (
        f"sample_rate mismatch: config says {cfg['sample_rate']}, but the AudioVAE encoder expects {expected_sr}."
    )

    # ---------------- data ---------------- #
    train_ds, val_ds = load_audio_text_datasets(
        train_manifest=cfg["train_manifest"],
        val_manifest=cfg["val_manifest"],
        sample_rate=cfg["sample_rate"],
    )

    def tokenize(batch):
        """将一批文本编码为 token id 序列（供 BatchProcessor 在训练时拼接使用）。"""
        return {"text_ids": [tokenizer(text) for text in batch["text"]]}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_texts = None
    if val_ds is not None:
        val_texts = list(val_ds["text"])  # 保留验证集原始文本，供验证阶段生成样本音频
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    num_train_samples = len(train_ds)

    # Optional: filter samples by estimated token count to avoid OOM
    # 可选：按估算 token 数过滤过长样本，避免显存溢出（OOM）
    if cfg["max_batch_tokens"] and cfg["max_batch_tokens"] > 0:
        from voxcpm.training.data import compute_sample_lengths

        # AudioVAE 每秒产生的隐变量帧数 = 采样率 / hop_length
        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        # 单个样本允许的最大 token 数 = batch 总 token 预算 / batch 大小
        max_sample_len = max(cfg["max_batch_tokens"] // cfg["batch_size"], 1)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and is_main:
            tracker.print(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={cfg['max_batch_tokens']})."
            )
        train_ds = train_ds.select(keep_indices)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        sample_rate=cfg["sample_rate"],
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            sample_rate=cfg["sample_rate"],
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        device=accelerator.device,
    )

    # audio_vae stays attached to the model so that sample-audio generation
    # works during validation; forward() never touches it.
    # audio_vae 仍挂在模型上，保证验证阶段能生成样本音频；
    # forward() 内部不会用到它。
    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    # ---------------- optimizer ---------------- #
    # 仅优化 requires_grad=True（即未被冻结）的参数
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    # 余弦退火学习率调度，前置 warmup_steps 步线性预热
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=cfg["num_iters"],
    )

    # All ranks load the same checkpoint to keep model and optimizer state in sync.
    # 所有进程加载同一份 checkpoint，保证各 rank 的模型与优化器状态一致
    start_step = load_checkpoint(model, optimizer, scheduler, save_dir, rank=accelerator.rank)
    accelerator.barrier()
    if start_step > 0 and is_main:
        tracker.print(f"Resuming training from step {start_step}")

    # ---------------- training loop ---------------- #
    grad_accum_steps = max(int(cfg["grad_accum_steps"]), 1)
    data_epoch = 0
    train_iter = iter(train_loader)

    def get_next_batch():
        """取下一个 batch；数据集耗尽时自动开启新 epoch 并重洗数据。"""
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(data_epoch)  # 设置 epoch 使 DistributedSampler 每轮重新洗牌
            train_iter = iter(train_loader)
            return next(train_iter)

    with tracker.live():
        for step in range(start_step, cfg["num_iters"]):
            tracker.step = step
            optimizer.zero_grad(set_to_none=True)  # 清空梯度（置 None 可节省内存）

            # Gradient accumulation
            # 梯度累积：将一个 optimizer step 拆成 grad_accum_steps 个 micro step
            loss_dict = {}
            for micro_step in range(grad_accum_steps):
                batch = get_next_batch()
                processed = batch_processor(batch)

                is_last_micro_step = micro_step == grad_accum_steps - 1
                # 非最后一个 micro step 时不做梯度同步（DDP no_sync），减少通信开销
                sync_context = contextlib.nullcontext() if is_last_micro_step else accelerator.no_sync()

                with sync_context:
                    # bfloat16 自动混合精度前向
                    with accelerator.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            processed["text_tokens"],
                            processed["text_mask"],
                            processed["audio_feats"],
                            processed["audio_mask"],
                            processed["loss_mask"],
                            processed["position_ids"],
                            processed["labels"],
                            progress=step / max(1, cfg["num_iters"]),
                        )

                    total_loss = 0.0
                    # 按权重累加各项子损失，并除以累积步数以保持梯度尺度正确
                    for key, value in outputs.items():
                        if key.startswith("loss/"):
                            weight = lambdas.get(key, 1.0)
                            total_loss = total_loss + value * weight / grad_accum_steps
                            loss_dict[key] = value.detach()

                    accelerator.backward(total_loss)

            # unscale / clip / step
            # 梯度反缩放 / 裁剪 / 更新参数
            scaler = getattr(accelerator, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer)  # 若使用 GradScaler，先还原真实梯度再裁剪
            effective_max_norm = cfg["max_grad_norm"] if cfg["max_grad_norm"] > 0 else 1e9  # 0 表示禁用裁剪（等价超大阈值）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in unwrapped_model.parameters() if p.requires_grad], max_norm=effective_max_norm
            )

            accelerator.step(optimizer)  # 更新模型参数
            accelerator.update()
            scheduler.step()  # 更新学习率

            # 按日志间隔打印训练指标（损失、学习率、epoch、梯度范数）
            if step % cfg["log_interval"] == 0 or step == cfg["num_iters"] - 1:
                loss_values = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
                loss_values["lr"] = float(optimizer.param_groups[0]["lr"])
                # 估算已消耗的 epoch 数：步数 × 累积步 × 单卡 batch × 卡数 / 样本总数
                epoch = (step * grad_accum_steps * cfg["batch_size"] * accelerator.world_size) / max(
                    1, num_train_samples
                )
                loss_values["epoch"] = float(epoch)
                loss_values["grad_norm"] = float(grad_norm)
                tracker.log_metrics(loss_values, split="train")

            # 按验证间隔执行验证，并可选地生成样本音频写回 TensorBoard
            if val_loader is not None and (step % cfg["valid_interval"] == 0 or step == cfg["num_iters"] - 1):
                validate(
                    model,
                    val_loader,
                    batch_processor,
                    accelerator,
                    tracker,
                    lambdas,
                    num_val_batches=cfg["num_val_batches"],
                )
                if writer is not None and accelerator.rank == 0 and cfg["num_audio_samples"] > 0:
                    try:
                        generate_sample_audio(
                            model,
                            val_ds,
                            val_texts,
                            writer,
                            step,
                            accelerator,
                            cfg["sample_rate"],
                            cfg["num_audio_samples"],
                            tracker,
                        )
                    except Exception as e:
                        tracker.print(f"[Warning] Sample audio generation failed: {e}")

            # 按保存间隔保存 checkpoint（仅主进程执行）
            if (step % cfg["save_interval"] == 0 or step == cfg["num_iters"] - 1) and is_main:
                save_checkpoint(model, optimizer, scheduler, save_dir, step, cfg["pretrained_path"])

    if is_main:
        # 训练结束后保存最终 checkpoint（以 num_iters 命名）
        save_checkpoint(model, optimizer, scheduler, save_dir, cfg["num_iters"], cfg["pretrained_path"])
    if writer:
        writer.close()


if __name__ == "__main__":
    train(load_config(parse_args()))
