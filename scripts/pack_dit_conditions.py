#!/usr/bin/env python3
"""Rewrite the per-patch DiT-condition LMDB into a compact, block-packed v2.

Old layout: 4 torch.save values per patch (pickle/zip overhead ~4x) with
random per-patch reads during training -> disk-bound (~5 s/it, GPU idle).

New layout: one key per block of consecutive patches; each value is raw
bf16 bytes laid out as [mu_all | x1_all | cond_all | mask_all] where each
segment is (P, ...) with P = patches in block. Raw bytes shrink ~4x and a
whole block (default 1024 patches ~2.6 MB) is a single sequential read.

The output is sharded into `--parts` independent LMDBs (train_v2/part_XX)
so N worker processes can write in parallel (LMDB allows a single writer
per environment). Training code reads all parts via the same dataset.

Usage:
  python scripts/pack_dit_conditions.py --workers 16
"""
import argparse
import io
import json
import os
from pathlib import Path

import lmdb
import torch
from multiprocessing import Process

SRC = "/data/datasets/tts_1000h/dit_conditions/train"
DST = "/data/datasets/tts_1000h/dit_conditions/train_v2"
SUFFIXES = ("mu", "x1", "cond", "mask")
DT = torch.bfloat16


def load_range(src_path: str, lo: int, hi: int):
    """Sequentially scan patches [lo, hi) with a cursor (sorted keys make this
    a linear sweep; per-key random gets + readahead=False were ~50x slower)."""
    env = lmdb.open(src_path, readonly=True, lock=False, readahead=True)
    got = {s: [] for s in SUFFIXES}
    with env.begin() as txn:
        cur = txn.cursor()
        ok = cur.set_range(f"{lo:012d}_".encode())
        while ok:
            k = cur.key()
            if not (len(k) >= 13 and k[:12].isdigit()):
                break  # ran into a meta key ("__len__" etc.) past the patch range
            idx = int(k[:12])
            if idx >= hi:
                break
            got[k[13:].decode()].append(
                torch.load(io.BytesIO(cur.value()), weights_only=True)
            )
            ok = cur.next()
    env.close()
    return [tuple(got[s][i] for s in SUFFIXES) for i in range(hi - lo)]


def load_patch(txn, idx: int):
    base = f"{idx:012d}"
    return tuple(
        torch.load(io.BytesIO(txn.get(f"{base}_{s}".encode())), weights_only=True)
        for s in SUFFIXES
    )


def pack_block(patches):
    """patches: list of (mu, x1, cond, mask) -> one raw-bytes blob."""
    out = []
    for si in range(4):
        stacked = torch.stack([p[si] for p in patches]).contiguous().to(DT)
        out.append(stacked.view(torch.uint8).numpy().tobytes())
    return b"".join(out)


def worker(rank: int, bounds, block: int, block_per_commit: int = 64):
    part_lo, part_hi = bounds[rank], bounds[rank + 1]  # block-aligned patch ids
    part_dir = Path(DST) / f"part_{rank:02d}"
    part_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(part_dir), map_size=int(part_hi - part_lo) * 3000 + (1 << 30), writemap=False)
    with env.begin() as txn:
        done_flag = txn.get(b"__done__")
    start_block = part_lo // block
    end_block = (part_hi + block - 1) // block  # include a short final block
    if done_flag:
        print(f"[w{rank}] already done", flush=True)
        return

    # Fine-grained resume: blocks already committed (e.g. after the __done__
    # crash) are skipped, so a restart only re-reads the uncommitted tail.
    with env.begin() as txn:
        cur = txn.cursor()
        if cur.last():
            k = cur.key()
            if k.startswith(b"blk_"):
                start_block = max(start_block, int(k[4:12]) + 1)

    src_path = SRC

    n_local = 0
    txn = env.begin(write=True)
    for gb in range(start_block, end_block):
        lo = gb * block
        hi = min((gb + 1) * block, part_hi)
        patches = load_range(src_path, lo, hi)
        blob = pack_block(patches)
        txn.put(f"blk_{gb:08d}".encode(), blob)
        txn.put(f"blk_{gb:08d}_n".encode(), str(hi - lo).encode())
        n_local += 1
        if n_local % block_per_commit == 0:
            txn.commit()
            txn = env.begin(write=True)
    # finish the still-open write transaction with the completion markers
    # (opening a second write txn on the same thread deadlocks -> lmdb.Error)
    txn.put(b"__done__", b"1")
    txn.put(b"__part_lo__", str(part_lo).encode())
    txn.put(b"__part_hi__", str(part_hi).encode())
    txn.commit()
    env.sync()
    env.close()
    print(f"[w{rank}] wrote {n_local} blocks ({part_lo}..{part_hi})", flush=True)


def main():
    global SRC, DST
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    p.add_argument("--block", type=int, default=1024, help="patches per block")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    SRC, DST = args.src, args.dst

    src = lmdb.open(SRC, readonly=True, lock=False)
    with src.begin() as txn:
        total = int(txn.get(b"__len__").decode())
    src.close()
    print(f"total patches: {total}, block={args.block}, parts={args.workers}")

    Path(DST).mkdir(parents=True, exist_ok=True)
    with open(Path(DST) / "__meta__.json", "w") as f:
        json.dump(
            {
                "total_patches": total,
                "block": args.block,
                "dtype": "bfloat16",
                "shapes": {"mu": [1024], "x1": [64, 2], "cond": [64, 2], "mask": [1, 2]},
                "layout": "[mu(P,1024) | x1(P,64,2) | cond(P,64,2) | mask(P,1,2)] bf16 raw bytes",
            },
            f,
            indent=2,
        )

    procs = []
    # block-aligned part boundaries so no block straddles two parts
    n_blocks = (total + args.block - 1) // args.block
    bounds = [n_blocks * r // args.workers * args.block for r in range(args.workers + 1)]
    bounds[-1] = total
    for r in range(args.workers):
        procs.append(Process(target=worker, args=(r, bounds, args.block)))
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    print("all parts done")


if __name__ == "__main__":
    main()
