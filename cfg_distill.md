# VoxCPM CFG 蒸馏文档


端侧推理加速：把 LocDiT 生成**每个 audio patch 的前向次数从 20 次降到 10 次**（DiT 部分 4× 提速）。

## 背景与原理

推理时每个 patch 的生成 = `10 步 Euler × CFG 双分支（有条件 + 无条件）= 20 次 DiT 前向`。

冻结教师 DiT 做 cond/uncond 双前向，用 CFG-Zero* 最优缩放 `s_star` 组合成目标速度场；学生 DiT（同一套权重 + 零初始化的 `cfg_mlp` w 嵌入）单次前向回归该目标 | 2 → 1 | 计算量减半


## 第一步：DiT 条件预计算

蒸馏训练只动 DiT，而 LM 栈（feat_encoder / base_lm / fsq / residual_lm / 投影层）是冻结的，因此其输出可以**一次性缓存**：把全量训练语料过一遍冻结 LM 栈，按 patch 粒度缓存 4 个张量到 LMDB——

- `mu`:  `(1024,)`   DiT 条件（`lm_to_dit_proj(lm_hidden) + res_to_dit_proj(residual_hidden)`）
- `x1`:  `(64, 2)`   真值音频特征（feat_dim × patch_size）
- `cond`:(64, 2)     前一 patch 特征（shift 条件）
- `mask`:(1, 2)      loss 权重掩码

LMDB 里的 KV 布局（key 均为字符串，value 为 `torch.save` 序列化的张量 bytes）：

```
"%012d_mu" / "%012d_x1" / "%012d_cond" / "%012d_mask"   # 每 patch 4 条，d 为全局 patch 序号
"__len__"   # patch 游标（已写入 patch 数）
"__src__"   # 源样本游标（已消费 manifest 行数）
```

位于：/data/datasets/tts_1000h/dit_conditions/train_v2

## 第二步：CFG 蒸馏

