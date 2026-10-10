#!/usr/bin/env python3
"""Pre-compute frozen LocDiT conditions (dit_hidden / feat_gt / feat_cond /
loss_seq_mask) for distillation training.

The LM stack (feat_encoder, base_lm, fsq_layer, residual_lm, projections) is
frozen during distillation, so its outputs can be cached once and reused for
all training iterations. This turns distillation training into a pure-DiT
small-batch workload.

Each LM "patch" (one audio patch, patch_size frames) becomes one sample:
  - mu:      (H_DiT,)        lm_to_dit_proj(lm_hidden) + res_to_dit_proj(res_hidden)
  - x1:      (D, P)          ground-truth audio features (transposed)
  - cond:    (D, P)          previous patch features (shifted, zero at t=0)
  - mask:    (P,)            loss weight mask
"""
import argparse
import io
import json
import math
import os
import sys
from pathlib import Path

import torch
from einops import rearrange
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_lmdb_tensor(txn, key: str, tensor: torch.Tensor):
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu(), buf)
    txn.put(key.encode(), buf.getvalue())


def _infer_src_from_patch_count(rows, batch_size, patch_count, config):
    """Infer how many source rows produced ``patch_count`` DiT patches.

    Only used for legacy lmdb files written before the ``__src__`` cursor
    existed. Reproduces the duration-sorted batching of the original run and
    walks the cumulative patch count until it reaches ``patch_count``.
    """
    audio_fps = 25          # AudioVAE frames per second (VoxCPM default)
    # Each source sample yields ceil(duration * audio_fps) DiT patches
    # (verified: lmdb patch totals match sum(ceil(duration*25)) exactly).
    cum = 0
    full_batches, tail = divmod(len(rows), batch_size)
    boundaries = []
    for b in range(full_batches):
        cum += sum(
            math.ceil(max(rows[b * batch_size + j].get("duration", 0.0), 1e-6) * audio_fps)
            for j in range(batch_size)
        )
        boundaries.append(cum)
    if tail:
        cum += sum(
            math.ceil(max(rows[full_batches * batch_size + j].get("duration", 0.0), 1e-6) * audio_fps)
            for j in range(tail)
        )
        boundaries.append(cum)

    # boundaries[b] = cumulative patches through batch b (0-based). The batch
    # whose cumulative count first reaches patch_count contains the breakpoint;
    # resume from the START of that batch so its partially-written tail is
    # rewritten under fresh (higher) keys.
    for b, c in enumerate(boundaries):
        if c >= patch_count:
            return b * batch_size
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", default="/data/models/VoxCPM-0.5B")
    p.add_argument("--train_manifest", default="/data/datasets/tts_1000h/manifests/all_1100h.jsonl")
    p.add_argument("--val_manifest", default="")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    p.add_argument("--out_dir", default="/data/datasets/tts_1000h/dit_conditions")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = p.parse_args()

    import lmdb
    import json as _json

    from voxcpm.model import VoxCPMModel
    from voxcpm.training import BatchProcessor
    from voxcpm.training.data import HFVoxCPMDataset

    model = VoxCPMModel.from_local(
        args.pretrained_path, optimize=False, training=True, device="cuda"
    )
    model.eval()
    if args.dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    # AudioVAE is float-only; from_local keeps it fp32 in training mode, but the
    # blanket .to(bfloat16) above would corrupt its conv weights. Restore fp32.
    model.audio_vae = model.audio_vae.to(torch.float32)
    model = model.to(model.device)  # ensure everything on cuda (from_local returns cpu model in training mode)
    model.config.dtype = args.dtype

    # Manual manifest read (avoids load_dataset split-generation on huge files).
    manifest = args.val_manifest if (args.split == "val" and args.val_manifest) else args.train_manifest
    rows = []
    with open(manifest, "r", encoding="utf-8") as _f:
        for i, line in enumerate(_f):
            if args.max_samples > 0 and i >= args.max_samples:
                break
            rows.append(_json.loads(line))

    # Sort rows by audio duration so each batch has near-uniform length -> far
    # less LM padding. (Shuffling is irrelevant: we cache every patch once.)
    rows.sort(key=lambda r: float(r.get("duration", 0.0)))

    # ---- Resume: decide the source-row cursor BEFORE building the dataset ----
    # The authoritative resume cursor is __src__ = number of SOURCE rows
    # already consumed; rows are stably sorted by duration, so skipping the
    # first __src__ rows reproduces the exact batching of the interrupted run.
    out_path = Path(args.out_dir) / args.split
    out_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_path), map_size=1 << 42, writemap=True)
    with env.begin() as txn:
        existing_src = txn.get(b"__src__")
        existing_len = txn.get(b"__len__")
    total_written = int(existing_len.decode()) if existing_len else 0
    if existing_src is not None:
        start_src = int(existing_src.decode())
    else:
        # Legacy lmdb written before __src__ existed: infer the source cursor
        # from the duration-sorted manifest by matching the cumulative patch
        # count. Patch counts are model-determined and reproducible, so this
        # lands exactly (or within one partial batch) on the true breakpoint.
        start_src = _infer_src_from_patch_count(
            rows, args.batch_size, total_written, model.config
        )
    if start_src:
        print(
            f"Resuming: {total_written} samples present, skipping {start_src} source rows.",
            file=sys.stderr,
        )
        rows = rows[start_src:]

    # Lazy-decode dataset: each __getitem__ loads one wav via soundfile on the
    # DataLoader worker (num_workers>0 parallelises decoding), so the main
    # process never pre-decodes the whole corpus.
    class _LazyWavDataset(torch.utils.data.Dataset):
        def __init__(self, rows, tokenizer):
            self.rows = rows
            self.text_ids = [tokenizer(r["text"]) for r in rows]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            import soundfile as sf

            r = self.rows[idx]
            wav, sr = sf.read(r["audio"], dtype="float32")
            return {
                "text_ids": self.text_ids[idx],
                "audio_array": wav,
                "audio_sampling_rate": sr,
                "dataset_id": 0,
                "is_prompt": False,
                "ref_audio_array": HFVoxCPMDataset._SENTINEL,
            }

        @staticmethod
        def collate_fn(batch):
            return HFVoxCPMDataset.collate_fn(batch)

    loader = torch.utils.data.DataLoader(
        _LazyWavDataset(rows, model.text_tokenizer),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_LazyWavDataset.collate_fn,
        drop_last=False,
    )

    batch_processor = BatchProcessor(
        config=model.config,
        audio_vae=model.audio_vae,
        dataset_cnt=1,
        device=model.device,
    )
    txn = env.begin(write=True)
    commit_every = 100  # batches per commit; keeps cursors queryable mid-run
    src_consumed = start_src  # absolute source rows consumed (incl. skipped)
    since_commit = 0  # counts batches written in the CURRENT transaction
    for batch in tqdm(loader, desc=f"precompute/{args.split}"):
        processed = batch_processor(batch)
        n_src = processed["text_tokens"].shape[0]  # source samples in this batch

        with torch.no_grad():
            # AudioVAE runs in fp32 (it is float-only); LM stack under bf16 autocast.
            out = model.compute_dit_conditions(
                processed["text_tokens"],
                processed["text_mask"],
                processed["audio_feats"],
                processed["audio_mask"],
                processed["loss_mask"],
            )

        n = out["mu"].shape[0]
        assert out["x1"].float().abs().sum() > 0, (
            f"batch all-zero audio feats (text_mask sum={processed['text_mask'].sum().item()}, "
            f"audio_mask sum={processed['audio_mask'].sum().item()})"
        )
        for j in range(n):
            write_lmdb_tensor(txn, f"{total_written:012d}_mu", out["mu"][j])
            write_lmdb_tensor(txn, f"{total_written:012d}_x1", out["x1"][j])
            write_lmdb_tensor(txn, f"{total_written:012d}_cond", out["cond"][j])
            write_lmdb_tensor(txn, f"{total_written:012d}_mask", out["mask"][j])
            total_written += 1
        src_consumed += n_src
        since_commit += 1

        if since_commit >= commit_every:
            txn.put(b"__len__", str(total_written).encode())
            txn.put(b"__src__", str(src_consumed).encode())
            txn.commit()
            env.sync()
            txn = env.begin(write=True)
            since_commit = 0

    txn.put(b"__len__", str(total_written).encode())
    txn.put(b"__src__", str(src_consumed).encode())
    txn.commit()
    env.sync()
    env.close()
    print(f"Done. wrote {total_written} DiT samples to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
