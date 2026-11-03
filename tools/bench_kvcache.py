# -*- coding: utf-8 -*-
"""解码速度对比：KV cache 开 / 关。

用法：python tools/bench_kvcache.py [--ckpt out_v6/ckpt_best.pt] [--n 128]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def bench(model, idx, n, use_cache, warmup=8):
    toks = []

    def cb(t):
        toks.append(t)

    with torch.no_grad():
        model.generate(idx.clone(), max_new_tokens=warmup, temperature=0.9, top_k=50,
                       use_cache=use_cache, on_token=cb)
        toks.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(idx.clone(), max_new_tokens=n, temperature=0.9, top_k=50,
                       use_cache=use_cache, on_token=cb)
        torch.cuda.synchronize()
        dt = time.time() - t0
    return len(toks) / dt, dt * 1000 / max(1, len(toks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_v6/ckpt_best.pt")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = GPT(cfg).to("cuda").eval()
    m.load_state_dict(ck["model"])
    print(f"ckpt: {args.ckpt}  cfg: {cfg}")
    print(f"{'prompt':>8} {'路径':>10} {'tok/s':>9} {'ms/token':>9} {'加速':>7}")
    for plen in (64, 256, 1024):
        idx = torch.randint(0, cfg["vocab_size"], (1, plen), device="cuda")
        r = {}
        for use in (False, True):
            r[use] = bench(m, idx, args.n, use)
        name = {False: "重算", True: "KV cache"}
        for use in (False, True):
            sp, ms = r[use]
            sp0 = r[False][0]
            print(f"{plen:>8} {name[use]:>10} {sp:>9.1f} {ms:>9.2f} {sp/sp0:>6.1f}x")


if __name__ == "__main__":
    main()
