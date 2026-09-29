# voxcpm1-0.5b

从 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 中单独抽取的 **VoxCPM1-0.5B** 推理 + 全参数微调代码。

与上游相比的简化：

- 只支持 voxcpm v1 架构（VoxCPM-0.5B），删除了 voxcpm1.5 / voxcpm2 相关代码（`voxcpm2.py`、`audio_vae_v2.py`、`local_dit_v2.py`）
- 删除 LoRA 相关代码，只保留全参数微调
- 删除 denoiser（ZipEnhancer）、ref_audio 训练路径、argbind，CLI 改为 argparse + YAML
- 模型权重路径默认 `/data/models/VoxCPM-0.5B`

## 环境

```bash
uv sync          # 安装依赖到 .venv
source .venv/bin/activate
```

## 推理

```bash
# 无参考音频 (zero-shot)
python infer.py --text "你好，欢迎使用 VoxCPM 语音合成模型。" --output output.wav

# 声音克隆（提供 prompt 音频及其文本）
python infer.py --text "需要合成的目标文本。" \
    --prompt_wav data/ref.wav --prompt_text "参考音频对应的文本。" \
    --output cloned.wav --seed 42

# 微调后的 checkpoint 目录可直接作为 --model_path
python infer.py --model_path checkpoints/finetune_all/latest --text "..."
```

Python API:

```python
from voxcpm import VoxCPM

voxcpm = VoxCPM(voxcpm_model_path="/data/models/VoxCPM-0.5B", optimize=False)
wav = voxcpm.generate(text="你好世界", seed=42)   # np.ndarray, float32
```

## 训练（全参数微调）

1. 准备 jsonl 清单，每行：

```json
{"text": "对应的转写文本", "audio": "/absolute/path/to/x.wav"}
```

音频会自动重采样到 16 kHz（AudioVAE 编码器采样率）。

2. 编辑 `conf/finetune_example.yaml`（数据路径、迭代数等）。

3. 启动训练：

```bash
python train.py --config conf/finetune_example.yaml

# 单条参数覆盖 YAML：
python train.py --config conf/finetune_example.yaml --num_iters 100 --learning_rate 5e-5

# 多卡（DDP）：
torchrun --nproc_per_node=2 train.py --config conf/finetune_example.yaml
```

说明：

- `audio_vae`（音频编解码器）始终冻结，其余参数默认全部参与训练；如需冻结 LM 保 WER，可设
  `freeze_prefixes: ["base_lm.", "fsq_layer.", "stop_proj.", "stop_actn.", "stop_head.", "enc_to_lm_proj.", "lm_to_dit_proj."]`
- TensorBoard：`tensorboard --logdir checkpoints/finetune_all/logs`（含 loss 曲线与试听音频）
- 训练中断后重新运行同一命令会自动从 `checkpoints/<save_path>/latest` 恢复（模型 + optimizer + scheduler + step）
- checkpoint 目录（`model.safetensors` + config/tokenizer/audiovae 文件）可直接用于 `infer.py --model_path`

## 代码结构

```
voxcpm/
├── tts.py             # VoxCPM 推理 pipeline（加载权重、generate API）
├── model/
│   ├── voxcpm.py           # VoxCPMModel：forward（训练）+ 自回归推理
│   └── utils.py            # tokenizer 包装、dtype/device/seed 工具
├── modules/
│   ├── minicpm4/           # MiniCPM4 LM backbone（base_lm / residual_lm）
│   ├── locenc/             # 局部音频编码器
│   ├── locdit/             # Local DiT + CFM（流匹配）
│   ├── audiovae/           # AudioVAE 16kHz 音频编解码器
│   └── layers/             # 标量量化层（FSQ）
├── training/
│   ├── accelerator.py      # 单卡/DDP + AMP
│   ├── data.py             # jsonl 清单加载、collate、BatchProcessor
│   ├── packers.py          # text+audio -> 训练序列打包
│   └── tracker.py          # 日志/TensorBoard
└── utils/text_normalize.py # 文本规整（可选）
train.py                    # 全参数微调入口
infer.py                    # 推理入口
conf/finetune_example.yaml  # 训练配置示例
```
