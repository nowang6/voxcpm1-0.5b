#!/usr/bin/env python3
"""CFG 蒸馏推理对比：基线（教师 + CFG 双前向 Euler） vs 蒸馏学生（单前向/步）。

基线：原始权重，solve_euler + CFG（每步 2 次 DiT 前向）。
蒸馏：加载蒸馏后的 cfg_mlp 权重，distill_solver（每步 1 次前向），
CFG 已烘焙进 w 嵌入（推理 cfg_value 须等于训练时的 w）。

用法:
  python scripts/test_distill_inference.py \
      --model_path /data/models/VoxCPM-0.5B \
      --dit_weights /data/models/voxcpm1-0.5b-distill/cfg/final/model.safetensors
"""
import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def run_case(name, tts, text, timesteps, cfg_value, out_dir, sample_rate, seed=42):
    """跑一个推理 case：计时、落盘 wav、返回统计。"""
    t0 = time.time()
    wav = tts.generate(
        text=text,
        cfg_value=cfg_value,
        inference_timesteps=timesteps,
        seed=seed,
    )
    elapsed = time.time() - t0
    dur = len(wav) / sample_rate
    rtf = elapsed / dur
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    path = out_dir / f"{name}.wav"
    sf.write(path, wav, sample_rate)
    print(
        f"{name:>18}: {timesteps:>2} steps | gen {elapsed:6.2f}s | audio {dur:5.2f}s "
        f"| RTF {rtf:5.3f} | peak {peak:.3f} -> {path}"
    )
    return {"name": name, "elapsed": elapsed, "dur": dur, "rtf": rtf, "peak": peak}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/data/models/VoxCPM-0.5B")
    p.add_argument("--dit_weights", default="/data/models/voxcpm1-0.5b-distill/cfg/final/model.safetensors")
    p.add_argument("--text", default="VoxCPM蒸馏测试：把每步两次前向压缩为单次前向，推理速度提升一倍。")
    p.add_argument("--out_dir", default="/data/models/voxcpm1-0.5b-distill/test_out")
    p.add_argument("--cfg_value", type=float, default=2.0, help="须等于蒸馏训练时的 w")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from voxcpm import VoxCPMTTS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading base model from {args.model_path} ...")
    tts = VoxCPMTTS(args.model_path, optimize=False)  # 必须关 torch.compile，否则热身后换权重行为不可控
    sample_rate = tts.tts_model.sample_rate

    # ---- 基线：教师权重 + CFG euler（每步 2 次前向） ----
    baseline = run_case("baseline_10step", tts, args.text, 10, args.cfg_value, out_dir, sample_rate, args.seed)

    # ---- 加载蒸馏 DiT 权重并接线蒸馏求解器 ----
    from safetensors.torch import load_file

    sd = load_file(args.dit_weights)
    missing, unexpected = tts.tts_model.feat_decoder.estimator.load_state_dict(sd, strict=False)
    print(f"loaded distilled DiT from {args.dit_weights}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(f"distilled weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    tts.tts_model.feat_decoder.use_distill_solver = True
    # CFG 蒸馏（阶段 1）学生恒以 dt=0 训练，推理必须传 dt=0（dt 嵌入未训练）
    tts.tts_model.feat_decoder.distill_use_dt = False

    d10 = run_case("distill_10step", tts, args.text, 10, args.cfg_value, out_dir, sample_rate, args.seed)

    print("\n=== summary（每 patch 的 DiT 前向次数） ===")
    print(f"baseline 10 steps x CFG(2x) = 20 forwards/patch: RTF {baseline['rtf']:.3f}")
    print(f"distill  10 steps x 1 fwd   = 10 forwards/patch: RTF {d10['rtf']:.3f}  ({baseline['rtf'] / d10['rtf']:.2f}x)")


if __name__ == "__main__":
    main()
