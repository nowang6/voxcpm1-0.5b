# VoxCPM CFG 蒸馏文档


端侧推理加速：把 LocDiT 生成**每个 audio patch 的前向次数从 20 次降到 10 次**（DiT 部分 2× 提速）。

## 背景与原理

推理时每个 patch 的生成 = `10 步 Euler × CFG 双分支（有条件 + 无条件）= 20 次 DiT 前向`。

冻结教师 DiT 做 cond/uncond 双前向，用 CFG-Zero* 最优缩放 `s_star` 组合成目标速度场；学生 DiT（同一套权重 + 零初始化的 `cfg_mlp` w 嵌入）单次前向回归该目标 | 2 → 1 | 计算量减半

**已于 2026-10 完成全流程验证与六版横评，定案配方见下。**

## 代码改动（本仓库）

- `voxcpm/modules/locdit/local_dit.py`：`VoxCPMLocDiT` 新增 `cfg_mlp`（w 嵌入）与 `init_cfg_embedding_zero()`（零初始化，学生初始行为 = 教师条件分支）；forward 加 `w` 参数，与时间嵌入同路注入首 token
- `voxcpm/modules/locdit/unified_cfm.py`：`teacher_estimator`（冻结教师副本）、`compute_cfg_distill_loss`（教师 CFG-Zero* 组合场 → 学生单前向）、`distill_solver`（单前向/步 Euler）、forward 的 `use_distill`/`distill_use_dt` 接线
- `voxcpm/model/generation.py`：`feat_decoder(...)` 调用透传 `use_distill_solver`/`distill_use_dt` 属性（默认关，零影响）
- `voxcpm/model/loader.py`：允许旧权重缺 `cfg_mlp` 4 键（零初始化补齐），其余缺失/多余仍报错
- `scripts/train_voxcpm_distill.py`：蒸馏训练（v2 块数据集、块级 shuffle、发散保护、断点续训、权重加载硬校验）
- `scripts/test_distill_inference.py`：基线 vs 蒸馏推理对比（RTF/badcase/试听）
- `scripts/gen_cv3_audio.py`：CV3-Eval 评测音频生成（支持蒸馏接线）
- 依赖：`uv add lmdb`

## 第一步：DiT 条件预计算（已完成，无需重跑）

蒸馏只动 DiT，LM 栈（feat_encoder / base_lm / fsq / residual_lm / 投影层）冻结，其输出一次性缓存：把全量训练语料（`tts_1000h`，1100h）过冻结 LM 栈，按 patch 粒度缓存 4 个张量：

- `mu`:  `(1024,)`   DiT 条件（`lm_to_dit_proj(lm_hidden) + res_to_dit_proj(residual_hidden)`）
- `x1`:  `(64, 2)`   真值音频特征（feat_dim × patch_size）
- `cond`:(64, 2)     前一 patch 特征（shift 条件）
- `mask`:(1, 2)      loss 权重掩码

**数据位置**：`/data/datasets/cfg-distill/train_v2`（112,049,826 patches / 268GB）。
v2 块打包格式：每 1024 个连续 patch 一个 key（`blk_%08d`），value 为裸 bf16 字节 `[mu(P,1024) | x1(P,64,2) | cond(P,64,2) | mask(P,1,2)]`，16 分片 `part_00..15`，`__meta__.json` 描述布局。读取端 `BlockConditionDataset`（块读取 + 单块 LRU）+ `BlockShuffleSampler`（块间乱序 + 块内乱序），实测 23 it/s（B=256）。

预计算与打包脚本在兄弟仓库 `/home/niwang/code/VoxCPM-0.5B/scripts/`（`precompute_dit_conditions.py` / `pack_dit_conditions.py`），全量重跑约 13.5h + 6h，一般不需要。

## 第二步：CFG 蒸馏训练（定案配方：单周期 140k 步）

**核心结论：一次完整长周期训练即可达到/超过任何多阶段组合，无需分段。**
「更多数据」由总步数实现（140k × 256 ≈ 35.8M patches，34% epoch）；「低 lr 精炼」由长 cosine 尾部自动实现（最后 40k 步 lr 从 ~9e-6 平滑降到 0）。分阶段训练的 lr 重启会扰动已收敛的分布内精度（实测 zh 3.48 → 3.64），再靠精炼轮补救不如不分段。

```bash
cd /home/niwang/code/voxcpm1-0.5b && source .venv/bin/activate
python scripts/train_voxcpm_distill.py \
    --lmdb_dir /data/datasets/cfg-distill/train_v2 \
    --w 2.0 --batch_size 256 --num_iters 140000 --num_workers 16 \
    --learning_rate 5e-5 --cfg_lr_mult 10 --warmup_steps 500 \
    --t_sampling uniform \
    --save_path /data/models/voxcpm1-0.5b-distill/cfg_single
```

- 单卡 RTX 5060 Ti 16GB，磁盘空闲时约 1.8-2 小时（16T 机械盘随机块读为瓶颈，训练时避免 tar/网盘等并行 IO）
- 中断重跑同一命令自动从 `latest.pt` 续训
- 健康信号：首步 loss ≈ **0.177**（真 base 权重的 cond-vs-combined 失配；若 ≈0.04 说明权重没加载上，立即停）
- 定案超参（勿改）：冻结 base 教师副本 + 学生全网络（非 `--train_cfg_only`）、`t_sampling uniform`（推理从 t=1 积分，大 t 必须有监督）、w=2.0、cfg_mlp 零初始化 ×10 lr

产物：`<save_path>/final/model.safetensors`（estimator 裸键名，含 cfg_mlp）。

## 推理接入

```python
from safetensors.torch import load_file
sd = load_file("<save_path>/final/model.safetensors")
tts.tts_model.feat_decoder.estimator.load_state_dict(sd, strict=False)  # 必须 missing=0
tts.tts_model.feat_decoder.use_distill_solver = True   # 走 distill_solver（每步单前向）
tts.tts_model.feat_decoder.distill_use_dt = False      # 必须为 False：CFG 蒸馏学生恒以 dt=0 训练
# 之后正常 generate(cfg_value=2.0, inference_timesteps=10)  # cfg_value 须等于训练 w
```

范例：`scripts/test_distill_inference.py`（RTF 对比 + 试听）、`scripts/gen_cv3_audio.py`（评测生成）。

## 评测结果（CV3-Eval zero-shot，1124 条，seed=42）

工具：`/home/niwang/code/voxcpm-cv3-eval`（生成用本仓库 `gen_cv3_audio.py`，评测用其 `eval_all.sh`；CER/WER + SIM + DNSMOS）。

| 子集 | 基线 | 20k | 100k(两阶段) | 200k | refine(三阶段) | **single(定案)** |
|---|---:|---:|---:|---:|---:|---:|
| zh (500) | 3.40 | 3.48 | 3.64 | 3.66 | 3.42 | **3.41** |
| en (500) | 4.49 | 5.10 | 4.94 | 5.13 | 5.56 | 5.12 |
| hard_zh (60) | 8.20 | 10.00 | 9.44 | 11.59 | 12.23 | 10.76 |
| hard_en (64) | 8.10 | 12.21 | 10.93 | 12.31 | 7.81* | 13.07 |

- **single 为综合最优**（zh+en 均值 4.27 并列第一，zh 3.41 六版最优，且配方最简）；仅英文占比高的场景可选 100k 版本（en 4.94，权重 `cfg/final_100k_best/`）
- *hard 子集仅 60/64 条，单样本即可撬动 ~1.5 点，跨版本翻转剧烈，不作主判据
- hard 子集的残余退化（+1~3）根因是训练数据（纯 TTS 无 prompt）与克隆推理分布不匹配，加训/降 lr 均无法补平；治本需按推理布局重打包 prompt 数据（历史实验证明简单混合无效）
- 端到端速度收益有限（RTF ~1.04×）：克隆场景 DiT 序列仅 5 token，前向占比小；收益在 DiT 计算量减半与端侧部署

各版本权重：`/data/models/voxcpm1-0.5b-distill/{cfg,cfg_refine,cfg_single}/`；逐条评分与 wav：`voxcpm-cv3-eval/results/distill_cmp/`（6 组）。

## 踩坑记录

- **权重加载硬校验**（承自兄弟仓库第五轮教训）：完整模型 checkpoint 键带 `feat_decoder.estimator.` 前缀，`strict=False` 静默加载 NOTHING 会让学生从随机初始化训练（假 loss 正常下降）。`load_dit_weights` 剥前缀 + 校验非 cfg_mlp 键全覆盖，否则 raise。首步 loss 0.177 是加载正确的信号。
- **调度器 resume 翻倍 bug**：`sched.load_state_dict` 恢复 `last_epoch` 后又快进 step 次会把 `last_epoch` 翻倍，cosine 直冲末尾 lr≈0，训练空转（loss 不降、权重不动）。修复：不 load 调度器状态，按当前 `--num_iters` 新周期从头重放到 step。教训：续训后第一件事核对 lr 数值是否符合新周期该步的预期。
- **`--init_dit` 教师语义**：教师副本必须恒取 base 权重。若让 init 权重污染教师，已收敛学生（只学过拟合 base 组合场）会突失配「自身权重的组合场」（该量从未被训练约束），loss 从 0.03 跳到 0.7。
- **lr 重启扰动**：cosine 周期切换时 lr 从 0 跳回中段，冲击已收敛的分布内精度（zh 3.48→3.64），200k 版本的全面回退即由此+过拟合叠加。单周期长调度无此问题。
- **训练 loss 低 ≠ rollout 正确**：验证蒸馏质量必须过 ASR 客观评测（听感不敏感）；评估场误差时注意 mask 归一化（padding 补齐 patch 占 ~26% 且不参与训练，不剔除会让 MSE 虚高 3 倍）。
- **机械盘 IO 争抢**：训练瓶颈是 268GB LMDB 的随机块读（16T HDD）。百度网盘同步 / tar 打包等并行 IO 会把训练饿死到 1-2 it/s（GPU 0%），磁盘 util 高但吞吐低即是此症。训练前确认盘空闲；`--num_workers 16` 比 8 略优。
- 评测工具 `.venv` 被清理后用 `uv venv .venv --python 3.10 && uv sync` 重建（uv.lock 固定依赖，ASR 模型缓存在 ~/.cache 不受影响）。
