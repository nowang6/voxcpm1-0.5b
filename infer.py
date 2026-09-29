#!/usr/bin/env python3
"""VoxCPM1-0.5B 推理脚本。

示例：
    # 零样本语音合成（zero-shot TTS）
    python infer.py --model_path /data/models/VoxCPM-0.5B \
        --text "你好，欢迎使用 VoxCPM 语音合成模型。" --output output.wav

    # 使用提示（音频）进行声音克隆
    python infer.py --model_path checkpoints/finetune_all/latest \
        --text "需要合成的目标文本。" \
        --prompt_wav ref.wav --prompt_text "参考音频对应的文本。" \
        --output cloned.wav --seed 42
"""
import argparse

import soundfile as sf
import torch

from voxcpm import VoxCPMTTS


def main():
    """推理入口：解析命令行参数，加载模型，合成音频并写出 wav 文件。"""
    parser = argparse.ArgumentParser(description="VoxCPM1-0.5B inference")
    parser.add_argument("--model_path", type=str, default="/data/models/VoxCPM-0.5B",
                        help="Model dir (pretrained or finetuned checkpoint folder)")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--prompt_wav", type=str, default=None, help="Prompt audio for voice cloning")
    parser.add_argument("--prompt_text", type=str, default=None, help="Transcript of the prompt audio")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--cfg_value", type=float, default=2.0)  # 无分类器引导（CFG）强度
    parser.add_argument("--inference_timesteps", type=int, default=10)  # 流匹配 ODE 求解步数
    parser.add_argument("--normalize", action="store_true", help="Text normalization (numbers, symbols, ...)")
    parser.add_argument("--retry_badcase", action=argparse.BooleanOptionalAction, default=True)  # 坏例重试开关
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--device", type=str, default=None)  # 运行设备，None 表示自动选择
    parser.add_argument("--seed", type=int, default=None)  # 随机种子，用于结果复现
    args = parser.parse_args()

    # 加载 VoxCPM 语音合成流水线；optimize=True 时启用 torch.compile 优化
    # （首次调用会先跑一次 dummy 生成做预热，编译耗时较长）

    voxcpm = VoxCPMTTS(
        voxcpm_model_path=args.model_path,
        optimize=not args.no_compile,
        device=args.device,
    )

    # 调用高层生成接口：内部完成文本分词、提示（音频）编码、自回归生成与 VAE 解码，
    # 返回一维 float32 波形数组（CPU 上的 numpy 数组）
    wav = voxcpm.generate(
        text=args.text,
        prompt_wav_path=args.prompt_wav,
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        retry_badcase=args.retry_badcase,
        seed=args.seed,
    )

    # 以模型原生采样率（16 kHz）写出 wav 文件
    sf.write(args.output, wav, voxcpm.tts_model.sample_rate)
    duration = len(wav) / voxcpm.tts_model.sample_rate  # 音频时长（秒）
    print(f"Saved {args.output} ({duration:.2f}s, {voxcpm.tts_model.sample_rate} Hz)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 释放显存，避免进程驻留占用 GPU


if __name__ == "__main__":
    main()
