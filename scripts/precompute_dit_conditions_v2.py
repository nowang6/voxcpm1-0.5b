#!/usr/bin/env python3
"""Pre-compute DiT conditions for CLONE-scenario distillation, writing the
block-packed v2 format directly (no intermediate per-patch LMDB).

Input manifest rows carry an optional ``ref_audio`` (same-speaker prompt);
packed sequences are then [103, ref, 104, text, 101, target, 102] with loss
mask = target only. Patches whose loss mask is all zero (the ref segment)
are NOT stored -- training only samples loss-carrying patches.

Output: <out_dir>/part_00 (v2 layout, see pack_dit_conditions.py) + __meta__.json
Resume: scans the largest written block and continues from its end.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from einops import rearrange
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DT = torch.bfloat16
SUFFIXES = ("mu", "x1", "cond", "mask")


def pack_block(patches):
    out = []
    for si in range(4):
        stacked = torch.stack([p[si] for p in patches]).detach().cpu().contiguous().to(DT)
        out.append(stacked.view(torch.uint8).numpy().tobytes())
    return b"".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default="/data/models/VoxCPM-0.5B")
    p.add_argument("--manifest", default="/data/datasets/tts_1000h/manifests/all_1100h_prompt.jsonl")
    p.add_argument("--out_dir", default="/data/datasets/tts_1000h/dit_conditions/train_prompt_v2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = p.parse_args()

    import lmdb
    import soundfile as sf

    from voxcpm.model import VoxCPMModel
    from voxcpm.training import BatchProcessor
    from voxcpm.training.data import HFVoxCPMDataset

    model = VoxCPMModel.from_local(args.pretrained_path, optimize=False, training=True, device="cuda")
    model.eval()
    if args.dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    model.audio_vae = model.audio_vae.to(torch.float32)
    model = model.to(model.device)
    model.config.dtype = args.dtype

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_samples and i >= args.max_samples:
                break
            rows.append(json.loads(line))
    rows.sort(key=lambda r: float(r.get("duration", 0.0)))
    print(f"manifest rows: {len(rows)} (with ref: {sum(1 for r in rows if r.get('ref_audio'))})", flush=True)

    class _LazyWavDataset(torch.utils.data.Dataset):
        def __init__(self, rows, tokenizer):
            self.rows = rows
            self.text_ids = [tokenizer(r["text"]) for r in rows]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            r = self.rows[idx]
            wav, sr = sf.read(r["audio"], dtype="float32")
            sample = {
                "text_ids": self.text_ids[idx],
                "audio_array": wav,
                "audio_sampling_rate": sr,
                "dataset_id": 0,
                "is_prompt": False,
                "ref_audio_array": HFVoxCPMDataset._SENTINEL,
            }
            if r.get("ref_audio"):
                rwav, rsr = sf.read(r["ref_audio"], dtype="float32")
                if rwav.ndim > 1:
                    rwav = rwav.mean(axis=1)
                sample["ref_audio_array"] = rwav
                sample["ref_audio_sampling_rate"] = rsr
            return sample

        @staticmethod
        def collate_fn(batch):
            return HFVoxCPMDataset.collate_fn(batch)

    out_dir = Path(args.out_dir)
    part_dir = out_dir / "part_00"
    part_dir.mkdir(parents=True, exist_ok=True)

    block = args.block
    # resume: find the highest written block; its end patch index = (gb+1)*block
    start_patch = 0
    env = lmdb.open(str(part_dir), map_size=500 << 30, writemap=False)
    with env.begin() as txn:
        if txn.get(b"__done__"):
            print("already done", flush=True)
            return
        cur = txn.cursor()
        if cur.last():
            k = cur.key()
            if k.startswith(b"blk_"):
                start_patch = (int(k[4:12]) + 1) * block
    if start_patch:
        print(f"resuming from patch {start_patch}", flush=True)
    # rows sorted by duration; skip by cumulative patch count
    cum = 0
    skipped_rows = 0
    for r in rows:
        need = math.ceil(max(float(r.get("duration", 0.0)), 1e-6) * 25)
        if cum + need > start_patch:
            break
        cum += need
        skipped_rows += 1
    if skipped_rows:
        rows = rows[skipped_rows:]
        print(f"skipped {skipped_rows} rows (~{cum} patches)", flush=True)

    with open(out_dir / "__meta__.json", "w") as f:
        json.dump({
            "total_patches": None,  # filled on completion
            "block": block,
            "dtype": "bfloat16",
            "shapes": {"mu": [1024], "x1": [64, 2], "cond": [64, 2], "mask": [1, 2]},
            "layout": "[mu(P,1024) | x1(P,64,2) | cond(P,64,2) | mask(P,1,2)] bf16 raw bytes",
            "source_manifest": args.manifest,
        }, f, indent=2)

    dataset = _LazyWavDataset(rows, model.text_tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_LazyWavDataset.collate_fn,
        drop_last=False,
    )

    batch_processor = BatchProcessor(
        config=model.config, audio_vae=model.audio_vae, dataset_cnt=1, device=model.device,
    )

    txn = env.begin(write=True)
    buffer = []
    n_blocks = 0
    total_patches = start_patch
    n_written = 0
    gb = start_patch // block
    commit_every = 32  # blocks

    def flush(force=False):
        nonlocal buffer, n_blocks, txn, gb, total_patches
        while len(buffer) >= block or (force and buffer):
            take = buffer[:block]
            del buffer[:block]
            lo = gb * block
            blob = pack_block(take)
            txn.put(f"blk_{gb:08d}".encode(), blob)
            txn.put(f"blk_{gb:08d}_n".encode(), str(len(take)).encode())
            gb += 1
            n_blocks += 1
            if n_blocks % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)

    pbar = tqdm(loader, desc="precompute/prompt_v2", dynamic_ncols=True)
    for batch in pbar:
        processed = batch_processor(batch)
        with torch.no_grad():
            out = model.compute_dit_conditions(
                processed["text_tokens"], processed["text_mask"],
                processed["audio_feats"], processed["audio_mask"],
                processed["loss_mask"],
            )
        n = out["mu"].shape[0]
        # keep only loss-carrying patches (drops the ref segment, mask all-zero)
        keep = out["mask"].float().sum(dim=(1, 2)) > 0
        for j in torch.nonzero(keep).flatten().tolist():
            buffer.append((out["mu"][j], out["x1"][j], out["cond"][j], out["mask"][j]))
            total_patches += 1
            n_written += 1
        flush()
        if pbar.n % 50 == 0:
            pbar.set_postfix(written=n_written, blocks=gb)

    flush(force=True)
    txn.put(b"__done__", b"1")
    txn.put(b"__part_lo__", b"0")
    txn.put(b"__part_hi__", str(total_patches).encode())
    txn.commit()
    env.sync()
    env.close()

    meta = json.loads((out_dir / "__meta__.json").read_text())
    meta["total_patches"] = total_patches
    (out_dir / "__meta__.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {total_patches} patches in {gb} blocks -> {part_dir}", flush=True)


if __name__ == "__main__":
    main()
