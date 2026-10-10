#!/usr/bin/env python3
"""LocDiT CFG 蒸馏训练：从预计算的 DiT 条件（v2 块打包 LMDB）训练蒸馏学生。

只训练 DiT 估计器（及其 cfg 嵌入）；VoxCPMModel 的其余部分完全不参与训练。
教师 = 原始权重的独立冻结副本（target 静止），学生 = 全网络 + 零初始化 cfg_mlp，
单前向回归教师的 CFG-Zero* 组合速度场。详见 cfg_distill.md 与方案文档。

用法:
  python scripts/train_voxcpm_distill.py \
      --lmdb_dir /data/datasets/tts_1000h/dit_conditions/train_v2 \
      --w 2.0 --batch_size 256 --num_iters 20000 \
      --save_path /data/models/voxcpm1-0.5b-distill/cfg
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from voxcpm.modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiT
from voxcpm.modules.minicpm4 import MiniCPM4Config


# ------------------------------------------------------------------ #
# 预计算 DiT 条件数据集（v2 块打包格式，见 pack_dit_conditions.py 的产物）
# ------------------------------------------------------------------ #
class BlockConditionDataset(Dataset):
    """v2 块打包数据集：每 1024 个连续 patch 打成一个 key 的裸 bf16 字节。

    每读一个块服务 ``block`` 个样本，配合 shuffle 后按块聚集的 sampler，
    把 ~5000 次/秒的随机 key 读降为每 ~1024 样本一次 2.6 MB 顺序读；
    每个 worker 缓存最近一个块。
    """

    def __init__(self, root: str):
        import lmdb

        root = Path(root)
        meta = json.loads((root / "__meta__.json").read_text())
        self.block = meta["block"]
        self.total = meta["total_patches"]
        self.shapes = {k: tuple(v) for k, v in meta["shapes"].items()}

        self.parts = []
        for part_dir in sorted(root.glob("part_*")):
            env = lmdb.open(str(part_dir), readonly=True, lock=False, readahead=False)
            with env.begin() as txn:
                lo = int(txn.get(b"__part_lo__").decode())
                hi = int(txn.get(b"__part_hi__").decode())
                assert txn.get(b"__done__") == b"1", f"{part_dir} incomplete"
            self.parts.append((lo, hi, env))

        # 全局扁平索引 -> (分片 env, 块 key, 块内偏移)
        self.n_blocks = (self.total + self.block - 1) // self.block
        self._env_of_block = []
        for b in range(self.n_blocks):
            pid = b * self.block
            for lo, hi, env in self.parts:
                if lo <= pid < hi:
                    self._env_of_block.append(env)
                    break
        self._cache_key = None
        self._cache = None

    def __len__(self):
        return self.total

    def _load_block(self, b: int):
        key = f"blk_{b:08d}".encode()
        if self._cache_key == key:
            return self._cache
        env = self._env_of_block[b]
        with env.begin() as txn:
            blob = txn.get(key)
            n = int(txn.get(key + b"_n").decode())
        segs, off = {}, 0
        for name, shape in self.shapes.items():
            cnt = n * math.prod(shape)
            # bytearray 拷贝一份可写缓冲，frombuffer 才能安全 reshape
            seg = torch.frombuffer(bytearray(blob[off: off + cnt * 2]), dtype=torch.bfloat16)
            segs[name] = seg.reshape(n, *shape)
            off += cnt * 2
        self._cache_key = key
        self._cache = segs
        return segs

    def __getitem__(self, idx: int):
        b, off = idx // self.block, idx % self.block
        segs = self._load_block(b)
        return segs["mu"][off], segs["x1"][off], segs["cond"][off], segs["mask"][off]


class BlockShuffleSampler(torch.utils.data.Sampler):
    """块间乱序 + 块内乱序：全局随机性的同时保持严格的局部访问（每 1024 样本一次块读）。"""

    def __init__(self, n_blocks, block, n_samples, seed: int):
        self.n_blocks, self.block, self.n_samples, self.seed = n_blocks, block, n_samples, seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        for b in torch.randperm(self.n_blocks, generator=g).tolist():
            lo, hi = b * self.block, min((b + 1) * self.block, self.n_samples)
            offs = torch.randperm(hi - lo, generator=g).tolist()
            for o in offs:
                yield lo + o

    def __len__(self):
        return self.n_samples


def collate_fn(batch):
    mu, x1, cond, mask = zip(*batch)
    return (
        torch.stack(mu),
        torch.stack(x1),
        torch.stack(cond),
        torch.stack(mask),
    )


# ------------------------------------------------------------------ #
# 从预训练 config 构建学生 DiT
# ------------------------------------------------------------------ #
def build_student_dit(pretrained_path: str, device: str, dtype: torch.dtype):
    with open(os.path.join(pretrained_path, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    lm_cfg = cfg["lm_config"]
    dit_cfg = cfg["dit_config"]

    decoder_config = MiniCPM4Config(**lm_cfg)
    decoder_config.hidden_size = dit_cfg["hidden_dim"]
    decoder_config.intermediate_size = dit_cfg["ffn_dim"]
    decoder_config.num_attention_heads = dit_cfg["num_heads"]
    decoder_config.num_hidden_layers = dit_cfg["num_layers"]
    decoder_config.kv_channels = dit_cfg.get("kv_channels")
    decoder_config.vocab_size = 0

    estimator = VoxCPMLocDiT(decoder_config, in_channels=cfg["feat_dim"])
    student = UnifiedCFM(
        in_channels=cfg["feat_dim"],
        cfm_params=CfmConfig(**dit_cfg.get("cfm_config", {})),
        estimator=estimator,
        mean_mode=False,
    )
    return student


def load_dit_weights(student: UnifiedCFM, weight_path: str, device: str):
    """加载 DiT 权重：剥前缀 + 硬校验。

    完整模型 checkpoint 的键带 ``feat_decoder.estimator.`` 前缀，而
    ``student.estimator`` 期望裸键名——不剥离时 ``strict=False`` 会静默
    加载 NOTHING，学生从随机初始化训练（实测发生过：missing=55
    unexpected=377，三轮训练全部作废）。蒸馏产物（save_student）已是
    裸键名，直接加载。
    """
    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(weight_path)
    else:  # .bin / .pth / .ckpt
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("state_dict", ckpt)
    valid = set(student.estimator.state_dict().keys())
    stripped = {k[len("feat_decoder.estimator."):]: v for k, v in sd.items() if k.startswith("feat_decoder.estimator.")}
    if stripped:
        sd = stripped
    sd = {k: v for k, v in sd.items() if k in valid}
    missing, unexpected = student.estimator.load_state_dict(sd, strict=False)
    print(f"loaded DiT weights from {weight_path}: matched={len(sd)} missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    non_cfg_missing = [k for k in missing if "cfg_mlp" not in k]
    if len(sd) == 0 or non_cfg_missing:
        raise RuntimeError(
            f"DiT weight load failed: matched={len(sd)} missing non-cfg keys={non_cfg_missing[:5]} — refusing to train from random init."
        )


def save_student(student: UnifiedCFM, path: Path):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    save_file(student.estimator.state_dict(), path / "model.safetensors")


def save_latest(student, opt, sched, step, path: Path):
    """滚动 checkpoint（权重 + 优化器 + 调度器 + 步数）：进程被杀后可无损续训。

    先写 .tmp 再 ``os.replace`` 原子换名，写一半被杀不会损坏已有文件。
    """
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": student.estimator.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
        },
        tmp,
    )
    os.replace(tmp, path)


# ------------------------------------------------------------------ #
# 训练主流程
# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default="/data/models/VoxCPM-0.5B")
    p.add_argument("--lmdb_dir", default="/data/datasets/cfg-distill/train_v2")
    p.add_argument("--w", type=float, default=2.0, help="蒸馏目标使用的 CFG 引导强度（=推理时 cfg_value）")
    p.add_argument("--init_dit", default="", help="可选的 DiT checkpoint（如上一轮蒸馏产物），默认取预训练权重")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_iters", type=int, default=20000)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--ckpt_interval", type=int, default=200,
                   help="滚动 latest.pt 的保存间隔（重启后自动续训）")
    p.add_argument("--learning_rate", type=float, default=5e-5,
                   help="DiT 主干学习率（蒸馏只需在教师附近小步微调）")
    p.add_argument("--cfg_lr_mult", type=float, default=10.0,
                   help="cfg 嵌入参数的学习率倍数（零初始化，需要更快起步）")
    p.add_argument("--train_cfg_only", action="store_true",
                   help="冻结 DiT 主干、只训 cfg 嵌入（教师=学生自身）。默认关闭："
                        "走「冻结教师副本 + 学生全网络」，目标静止且学生全容量")
    p.add_argument("--t_sampling", default="uniform", choices=["uniform", "log-norm"],
                   help="t 采样分布。uniform 覆盖全 t 区间（推理从 t=1 积分，必须覆盖大 t）；"
                        "log-norm 匹配基模型 CFM 训练分布但大 t 区无监督（rollout 会崩）")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", default="/data/models/voxcpm1-0.5b-distill/cfg")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    student = build_student_dit(args.pretrained_path, device, dtype).to(device, dtype)
    student.estimator.init_cfg_embedding_zero()  # 学生初始行为 = 教师条件分支
    # 教师永远取原始 base 权重的冻结副本：蒸馏目标唯一且静止。--init_dit 只热启动
    # 学生——若让它同时改变教师，已收敛学生会突然失配新目标（自身权重的组合场
    # 从未被训练约束），loss 会从 ~0.03 跳到 0.2+。
    base_sd = os.path.join(args.pretrained_path, "model.safetensors")
    if not os.path.exists(base_sd):
        base_sd = os.path.join(args.pretrained_path, "pytorch_model.bin")
    load_dit_weights(student, base_sd, device)
    import copy

    teacher = copy.deepcopy(student.estimator)
    for pm in teacher.parameters():
        pm.requires_grad_(False)
    teacher.eval()
    student.teacher_estimator = teacher
    if args.init_dit:  # 学生从蒸馏产物热启动（教师保持 base 不变）
        load_dit_weights(student, args.init_dit, device)
    student.train()

    # ---- 数据：v2 块打包（__meta__.json 存在性校验） ----
    if not (Path(args.lmdb_dir) / "__meta__.json").exists():
        raise FileNotFoundError(f"{args.lmdb_dir} 不是 v2 块打包格式（缺 __meta__.json）")
    ds = BlockConditionDataset(args.lmdb_dir)
    sampler = BlockShuffleSampler(ds.n_blocks, ds.block, len(ds), args.seed)
    print(f"dataset v2: {len(ds)} patches, {ds.n_blocks} blocks", file=sys.stderr)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        shuffle=False,
        sampler=sampler,
    )

    # ---- 优化器：分层 LR（教师副本已在加载段挂好，恒为 base 权重） ----
    # 主干从教师初始化、只需小步微调；零初始化的 cfg 嵌入需要快得多 的学习率。
    # 冻结教师副本 + 学生全网络：回归目标静止，学生全容量（两个历史问题一次
    # 解决：共享权重的自蒸馏目标随更新移动会发散；cfg-only 又欠拟合）。
    cfg_params = [pm for nm, pm in student.named_parameters() if "cfg_mlp" in nm]
    base_params = [pm for nm, pm in student.named_parameters() if "cfg_mlp" not in nm]
    assert cfg_params, "cfg_mlp params not found -- is init_cfg_embedding_zero() called?"
    if args.train_cfg_only:
        for pm in base_params:
            pm.requires_grad_(False)
        opt = torch.optim.AdamW(cfg_params, lr=args.learning_rate * args.cfg_lr_mult,
                                weight_decay=args.weight_decay)
        trainables = cfg_params
    else:
        opt = torch.optim.AdamW(
            [
                {"params": base_params, "lr": args.learning_rate},
                {"params": cfg_params, "lr": args.learning_rate * args.cfg_lr_mult},
            ],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        trainables = base_params + cfg_params
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))
        * (0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, args.num_iters))))),
    )

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(save_dir / "train.log", "a")

    def log(msg):
        print(msg, file=sys.stderr)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"w={args.w} num_samples={len(ds)} batch={args.batch_size} iters={args.num_iters} "
        f"lr={args.learning_rate} cfg_lr_mult={args.cfg_lr_mult} t_sampling={args.t_sampling} "
        f"train_cfg_only={args.train_cfg_only}")

    # ---- 自动续训：latest.pt 存在则恢复（数据顺序不恢复，反正每轮 shuffle） ----
    step = 0
    latest_path = save_dir / "latest.pt"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)
        student.estimator.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        # 调度器不 load 旧 state_dict：先 load 再快进会把 last_epoch 翻倍
        # （99800*2/200000 -> cosine 末尾 lr≈0，训练空转）。改为按当前
        # --num_iters 的新周期从头重放到 step，lr 即为新配置该步的正确值。
        for _ in range(step):
            sched.step()
        log(f"resumed from {latest_path} at step={step} (lr={sched.get_last_lr()[0]:.2e})")

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.dtype == "bfloat16"))
    init_loss = None  # 本进程首个迭代的 loss（断点续训后重新赋值，发散保护基准）
    t0 = time.time()
    data_iter = iter(loader)

    pbar = tqdm(total=args.num_iters, desc="distill/cfg")
    while step < args.num_iters:
        try:
            mu, x1, cond, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            mu, x1, cond, mask = next(data_iter)

        mu = mu.to(device, dtype=dtype, non_blocking=True)
        x1 = x1.to(device, dtype=dtype, non_blocking=True)
        cond = cond.to(device, dtype=dtype, non_blocking=True)
        mask = mask.to(device, dtype=dtype, non_blocking=True)

        with autocast_ctx:
            loss = student.compute_cfg_distill_loss(
                x1, mu, cond=cond, tgt_mask=mask, w=args.w, t_sampling=args.t_sampling
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, args.max_grad_norm)
        opt.step()
        sched.step()

        # 发散保护：loss 超过本进程首步 loss 的 10 倍即停（保留 final 权重）
        loss_val = loss.item()
        if init_loss is None:
            init_loss = loss_val
        if loss_val > max(1.0, 10.0 * init_loss):
            log(f"DIVERGED at step={step}: loss={loss_val:.4f} >> init={init_loss:.4f}. Stopping.")
            save_student(student, save_dir / "final")
            break

        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % args.log_interval == 0:
            lr = sched.get_last_lr()[0]
            ips = (step + 1) / (time.time() - t0)
            log(f"step={step} loss={loss.item():.5f} lr={lr:.2e} ips={ips:.1f}")

        if step > 0 and step % args.ckpt_interval == 0:
            save_latest(student, opt, sched, step, latest_path)

        if step > 0 and step % args.save_interval == 0:
            save_student(student, save_dir / f"step_{step:07d}")

        step += 1

    save_student(student, save_dir / "final")
    log(f"done. elapsed={(time.time() - t0) / 60:.1f}min")
    log_f.close()
    pbar.close()


if __name__ == "__main__":
    main()
