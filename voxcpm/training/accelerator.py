"""轻量级训练加速器：管理分布式进程组（torchrun + NCCL）、DDP 模型包装、
AMP 混合精度（GradScaler 梯度缩放）以及分布式数据加载器的构建。"""
from __future__ import annotations

import contextlib
import os
import random
import typing

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel


class Accelerator:
    """
    极简训练加速器：在使用 ``torchrun`` 启动时初始化分布式进程组，
    并提供 AMP（自动混合精度）与 DDP（分布式数据并行）相关的辅助方法。

    Args:
        amp: 是否启用 AMP 混合精度（配合 GradScaler 做梯度缩放）。
        seed: 随机种子，保证多卡下模型初始化一致。
    """

    def __init__(self, amp: bool = False, seed: int = 42):
        # 从环境变量读取全局进程数（world_size）；未用 torchrun 启动时默认为 1（单进程）
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        # 多进程且进程组尚未初始化时，基于环境变量建立 NCCL 后端的分布式进程组
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")

        # 全局 rank 与节点内 local_rank：用于选择 GPU、判断是否主进程（rank 0）
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.amp = amp

        # Set random seed to ensure model initialization consistency
        # 设置随机种子，保证模型初始化一致
        self._set_seed(seed)

        # 未启用 AMP 时的空操作梯度缩放器，使训练循环无需分支判断
        class DummyScaler:
            def step(self, optimizer):
                # 直接执行优化器更新（无梯度缩放）
                optimizer.step()

            def scale(self, loss):
                # 原样返回损失，不做缩放
                return loss

            def unscale_(self, optimizer):
                # 无需反缩放，原样返回
                return optimizer

            def update(self):
                pass

        # 启用 AMP 且 CUDA 可用时使用 GradScaler（防 fp16 梯度下溢），否则用空实现
        self.scaler = torch.amp.GradScaler("cuda") if (amp and torch.cuda.is_available()) else DummyScaler()
        # 绑定本进程使用的 CUDA 设备上下文（无 CUDA 时为 None）
        self.device_ctx = torch.cuda.device(self.local_rank) if torch.cuda.is_available() else None
        self._ddp_model = None  # For no_sync support
        # 保存 DDP 包装后的模型引用，供梯度累积时的 no_sync 使用

    def _set_seed(self, seed: int):
        """设置随机种子，保证多 GPU 下模型初始化一致。"""
        # 分别固定 PyTorch / NumPy / Python 内置随机源
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            # 固定所有可见 GPU 上的随机种子
            torch.cuda.manual_seed_all(seed)

    def __enter__(self):
        """进入上下文：激活本进程绑定的 CUDA 设备上下文。"""
        if self.device_ctx is not None:
            self.device_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文：释放本进程绑定的 CUDA 设备上下文。"""
        if self.device_ctx is not None:
            self.device_ctx.__exit__(exc_type, exc_value, traceback)

    def barrier(self):
        """同步屏障：阻塞等待所有进程都到达此处后再继续。"""
        # 仅分布式环境已初始化时才需要同步
        if dist.is_initialized():
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.AVG):
        """跨所有进程对张量做 all-reduce 归约（默认对各 rank 求平均 AVG），返回该张量。"""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    # ------------------------------------------------------------------ #
    # Model helpers
    # 模型相关辅助方法
    # ------------------------------------------------------------------ #
    def prepare_model(self, model: torch.nn.Module, **kwargs):
        """把模型移到当前设备；多卡时再用 DDP（分布式数据并行）包装。

        Args:
            model: 待准备的模型。
            **kwargs: 透传给 DistributedDataParallel 的额外参数。

        Returns:
            位于正确设备、必要时已被 DDP 包装的模型。
        """
        if hasattr(model, "device"):  # make sure the matrix will be moved to the correct device
            # 同步更新模型自身记录的 device 属性，确保内部生成的矩阵被放到正确设备
            model.device = self.device
        model = model.to(self.device)
        if self.world_size > 1:
            # 多卡：用 DDP 包装模型，并指定本进程负责的 GPU
            model = DistributedDataParallel(model, device_ids=[self.local_rank], **kwargs)
            self._ddp_model = model  # Save DDP model reference for no_sync support
        return model

    @contextlib.contextmanager
    def no_sync(self):
        """
        梯度累积期间跳过梯度同步的上下文管理器。

        仅在每个累积周期的最后一个 micro-batch 之外使用：前面的 micro-batch
        反向传播时不做 DDP 梯度 all-reduce，最后一步统一同步，等效扩大批大小。
        """
        if self._ddp_model is not None:
            # DDP 模型：进入 no_sync 模式，本次 backward 不触发梯度同步
            with self._ddp_model.no_sync():
                yield
        else:
            # 单卡/未包装：无需任何处理
            yield

    @property
    def device(self):
        """当前进程使用的计算设备：CUDA 可用时为本机 rank 对应的 GPU，否则为 CPU。"""
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # AMP helpers
    # AMP 混合精度辅助方法
    # ------------------------------------------------------------------ #
    def autocast(self, *args, **kwargs):
        """返回 CUDA autocast 上下文；self.amp 为 False 时混合精度不生效。"""
        return torch.amp.autocast("cuda", enabled=self.amp, *args, **kwargs)

    def backward(self, loss: torch.Tensor):
        """对损失反向传播；启用 AMP 时先经 GradScaler 缩放，避免 fp16 梯度下溢。"""
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer):
        """执行优化器更新：GradScaler 会先反缩放梯度，必要时跳过不安全的更新。"""
        self.scaler.step(optimizer)

    def update(self):
        """每个优化步结束后更新 GradScaler 的缩放因子。"""
        self.scaler.update()

    # ------------------------------------------------------------------ #
    # Data helpers
    # 数据加载辅助方法
    # ------------------------------------------------------------------ #
    def prepare_dataloader(
        self,
        dataset: typing.Iterable,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = False,
    ) -> torch.utils.data.DataLoader:
        """构建数据加载器；多卡时挂载分布式采样器（DistributedSampler）。

        Args:
            dataset: 数据集（需支持 len 与按索引取样本）。
            batch_size: 每个进程（每卡）的批大小。
            num_workers: 数据加载的工作进程数。
            shuffle: 是否打乱数据（多卡时改由分布式采样器负责打乱）。
            collate_fn: 自定义批拼接（collate）函数。
            drop_last: 是否丢弃最后一个不足 batch_size 的不完整批。

        Returns:
            可迭代的 torch.utils.data.DataLoader。
        """
        if self.world_size > 1:
            # 多卡：用分布式采样器把数据按 world_size 切分，保证各卡样本互不重叠
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            # 打乱逻辑已交给采样器，DataLoader 层面关闭 shuffle
            shuffle = False
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            # pin_memory=True：使用页锁定内存，加速 CPU -> GPU 的数据拷贝
            pin_memory=True,
        )

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        """剥掉 DDP 包装，返回底层原始模型（未被包装时原样返回）。"""
        return model.module if hasattr(model, "module") else model
