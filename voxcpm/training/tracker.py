"""训练指标跟踪器：负责训练过程中的日志打印（仅主进程）、指标记录（TensorBoard
writer）、日志文件写入，以及断点续训时全局步数（step）状态的保存与恢复。"""
from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path
from typing import Dict, Optional


class TrainingTracker:
    """
    受 minimcpm-audio 训练流程启发的轻量级训练指标跟踪器。

    维护当前全局步数（global step）；打印只感知 rank 的日志消息（仅 rank 0 输出）；
    可选地通过传入的 writer 写入 TensorBoard；并把进度写入日志文件，便于事后查看。

    Args:
        writer: TensorBoard SummaryWriter（或任何带 add_scalar 的对象），可为 None。
        log_file: 日志文件路径；不传则不写文件。
        rank: 当前进程的分布式 rank，仅 rank 0 打印与写日志。
    """

    def __init__(
        self,
        *,
        writer=None,
        log_file: Optional[str] = None,
        rank: int = 0,
    ):
        self.writer = writer
        self.log_file = Path(log_file) if log_file else None
        # 确保日志文件的父目录存在
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        # 当前全局训练步数（随 checkpoint 保存/恢复）
        self.step = 0
        # Record the time of the last log to calculate the interval
        # 记录上一次打日志的时间，用于计算两次日志之间的间隔
        self._last_log_time: float | None = None

    # ------------------------------------------------------------------ #
    # Logging helpers
    # 日志辅助方法
    # ------------------------------------------------------------------ #
    def print(self, message: str):
        """打印消息：仅 rank 0 输出到 stderr，并同时追加写入日志文件（可选）。"""
        # 仅主进程打印，避免多卡时重复输出
        if self.rank == 0:
            print(message, flush=True, file=sys.stderr)
            # 同步追加到日志文件，便于事后排查
            if self.log_file:
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(message + "\n")

    def log_metrics(self, metrics: Dict[str, float], split: str):
        """记录一组指标：打印一行汇总日志，并写入 TensorBoard。

        Args:
            metrics: 指标名到数值的映射（如 loss、learning_rate 等）。
            split: 指标所属阶段（如 train / validation）。
        """
        if self.rank == 0:
            # 计算距上次记录日志的时间间隔，便于观察日志频率
            now = time.time()
            dt_str = ""
            if self._last_log_time is not None:
                dt = now - self._last_log_time
                dt_str = f", log interval: {dt:.2f}s"
            self._last_log_time = now

            formatted = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
            self.print(f"[{split}] step {self.step}: {formatted}{dt_str}")
        # 数值型指标写入 TensorBoard，标签为 {split}/{key}
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{key}", value, self.step)

    def done(self, split: str, message: str):
        """输出某阶段结束的消息（如 epoch 完成）。"""
        self.print(f"[{split}] {message}")

    # ------------------------------------------------------------------ #
    # State dict
    # 状态字典（随检查点保存/恢复全局步数）
    # ------------------------------------------------------------------ #
    def state_dict(self):
        """返回可随检查点保存的状态（当前全局步数）。"""
        return {"step": self.step}

    def load_state_dict(self, state):
        """从检查点状态恢复全局步数。"""
        self.step = int(state.get("step", 0))

    # ------------------------------------------------------------------ #
    # Context manager compatibility (for parity with minicpm-audio code)
    # 上下文管理器兼容接口（与 minicpm-audio 代码保持一致）
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def live(self):
        """空实现的上下文管理器，仅为与 minicpm-audio 训练循环保持接口一致。"""
        yield
