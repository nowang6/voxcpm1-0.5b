from typing import List, Tuple
import torch


class StaticKVCache:
    """静态 KV 缓存：预分配固定大小的键值缓存张量，用于自回归逐步解码。

    缓存张量形状为 (2, num_layers, batch_size, num_kv_heads, max_length, dim_kv_head)，
    第 0 维分别存放 K（下标 0）与 V（下标 1），序列维按 max_length 预分配。
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        dim_kv_head: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int = 8192,
    ):
        """初始化静态 KV 缓存。

        参数:
            num_layers: Transformer 层数
            num_kv_heads: 每层 KV 头数（GQA）
            dim_kv_head: 每个 KV 头的维度
            batch_size: 批大小
            device: 缓存所在设备
            dtype: 缓存数据类型
            max_length: 预分配的最大序列长度（默认 8192）
        """
        self.max_length = max_length
        self.num_layers = num_layers

        self.kv_cache = torch.zeros(
            2,  # 维度 0：下标 0 存 K，下标 1 存 V
            num_layers,
            batch_size,
            num_kv_heads,
            max_length,
            dim_kv_head,
            device=device,
            dtype=dtype,
        )
        self.current_length = 0  # 当前已写入的序列长度（写入位置指针）

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 (K, V) 缓存视图，形状均为 (batch_size, num_kv_heads, max_length, dim_kv_head)。"""
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self) -> int:
        """写入一个新 token：返回当前写入位置，并将长度指针前移一位。"""
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")

        ret = self.current_length
        self.current_length += 1
        return ret

    def fill_caches(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]):
        """用预填充（prefill）阶段各层的 KV 缓存初始化静态缓存。

        参数:
            kv_caches: 长度为 num_layers 的列表，每个元素为该层的 (K, V) 张量
        """
        self.current_length = kv_caches[0][0].size(2)  # 以第一层 K 的序列长度作为当前已写入长度
        self.kv_cache.zero_()  # 清零整块缓存，避免残留数据
        for i in range(self.num_layers):
            self.kv_cache[0, i, :, :, : self.current_length, :] = kv_caches[i][0]  # 写入第 i 层的 K
            self.kv_cache[1, i, :, :, : self.current_length, :] = kv_caches[i][1]  # 写入第 i 层的 V
