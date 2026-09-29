#!/usr/bin/env python3
"""用本仓库的 VoxCPM 为 CV3-Eval zero-shot 测试集生成音频(评测客观指标用)。

与 voxcpm-cv3-eval/gen_audio.py 的输出结构完全一致,评测侧直接用其
eval_all.sh;差异仅在生成端使用本仓库 API(VoxCPMTTS),并支持蒸馏权重
(--dit-weights 走 distill_solver 单前向推理)。

示例:
  # 基线
  python scripts/gen_cv3_audio.py --model /data/models/VoxCPM-0.5B \
      --cv3-root ~/code/CV3-Eval --out-root <decode-root>/baseline
  # 蒸馏
  python scripts/gen_cv3_audio.py --model /data/models/VoxCPM-0.5B \
      --dit-weights /data/models/voxcpm1-0.5b-distill/cfg/final/model.safetensors \
      --cv3-root ~/code/CV3-Eval --out-root <decode-root>/distill

支持断点续跑:已存在的 wav 会跳过。
"""
import argparse
import time
import traceback
from pathlib import Path

import soundfile as sf

ALL_LANGS = ["zh", "en", "hard_zh", "hard_en"]


def load_scp(path: Path):
    items = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        items.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/models/VoxCPM-0.5B")
    p.add_argument("--cv3-root", required=True, help="CV3-Eval 仓库根目录")
    p.add_argument("--out-root", required=True, help="生成音频输出根目录")
    p.add_argument("--langs", nargs="+", default=ALL_LANGS)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--timesteps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42, help="固定种子保证两组可比")
    p.add_argument("--normalize", action="store_true", help="文本归一化(CV3 评测不建议,默认关)")
    p.add_argument("--dit-weights", default="", help="蒸馏 DiT 权重(safetensors,estimator 裸键名);设置后走蒸馏推理路径")
    p.add_argument("--no-prompt", action="store_true", help="design 模式:不使用 prompt(纯 TTS)")
    args = p.parse_args()

    cv3_root = Path(args.cv3_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    from voxcpm import VoxCPMTTS

    tts = VoxCPMTTS(args.model, optimize=False)  # 蒸馏要换权重,关 torch.compile(两组一致)
    sr = tts.tts_model.sample_rate
    print(f"model sample_rate={sr}", flush=True)

    if args.dit_weights:
        from safetensors.torch import load_file

        sd = load_file(args.dit_weights)
        missing, unexpected = tts.tts_model.feat_decoder.estimator.load_state_dict(sd, strict=False)
        print(f"loaded distilled DiT from {args.dit_weights}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        assert len(missing) == 0 and len(unexpected) == 0, "蒸馏权重加载不完整"
        tts.tts_model.feat_decoder.use_distill_solver = True
        # CFG 蒸馏(阶段 1)学生恒以 dt=0 训练,推理必须传 dt=0
        tts.tts_model.feat_decoder.distill_use_dt = False
        print("distill inference enabled (use_dt=False)", flush=True)

    for lang in args.langs:
        dump = cv3_root / "data/zero_shot" / lang
        prompt_wavs = dict(load_scp(dump / "prompt_wav.scp"))
        prompt_texts = dict(load_scp(dump / "prompt_text"))
        texts = load_scp(dump / "text")

        out_dir = out_root / lang / "wavs"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_done = n_skip = n_fail = 0
        t_start = time.time()
        for i, (utt, text) in enumerate(texts, 1):
            out_path = out_dir / f"{utt}.wav"
            if out_path.exists():
                n_skip += 1
                continue
            prompt_wav = Path(prompt_wavs[utt]).expanduser()
            if not prompt_wav.is_absolute():
                prompt_wav = cv3_root / prompt_wav
            try:
                t0 = time.time()
                kwargs = dict(
                    text=text,
                    cfg_value=args.cfg,
                    inference_timesteps=args.timesteps,
                    normalize=args.normalize,
                    seed=args.seed,
                )
                if not args.no_prompt:
                    kwargs["prompt_wav_path"] = str(prompt_wav)
                    kwargs["prompt_text"] = prompt_texts[utt]
                wav = tts.generate(**kwargs)
                sf.write(out_path, wav, sr)
                n_done += 1
                print(
                    f"[{lang} {i}/{len(texts)}] {utt} "
                    f"{len(wav) / sr:.1f}s 耗时{time.time() - t0:.1f}s "
                    f"(done={n_done} skip={n_skip} fail={n_fail})",
                    flush=True,
                )
            except Exception:
                n_fail += 1
                print(f"[{lang}] {utt} 生成失败:", flush=True)
                traceback.print_exc()

        print(
            f"== {lang} 完成: 新生成 {n_done}, 跳过 {n_skip}, 失败 {n_fail}, "
            f"总耗时 {(time.time() - t_start) / 60:.1f} 分钟 ==",
            flush=True,
        )


if __name__ == "__main__":
    main()
