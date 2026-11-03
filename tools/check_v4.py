#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查打包结果：随机抽块解码、EOS 密度、byte-fallback 比例、特殊 token 统计。

用法：
    python tools/check_v4.py --data data_v4
    python tools/check_v4.py --data /tmp/pack_smoke --chunks 5
"""
import argparse
import collections
import json
import os
import random

import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT + "/data_v4")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sentencepiece as spm
    meta = json.load(open(os.path.join(args.data, "meta.json"), encoding="utf-8"))
    model = os.path.join(os.path.dirname(args.data.rstrip("/")), "data_v4", meta["tokenizer"])
    if not os.path.exists(model):
        cand = [p for p in (os.path.join(args.data, meta["tokenizer"]),
                            ROOT + "/data_v4/" + meta["tokenizer"]) if os.path.exists(p)]
        model = cand[0] if cand else model
    sp = spm.SentencePieceProcessor(model_file=model)
    vocab = meta["vocab"]
    eos_id = meta["eos_id"]
    print(f"meta: vocab={vocab} ctx={meta['ctx']} eos_id={eos_id} norm={meta.get('norm')}")
    print(f"分词器: {model}")
    for split in ("train", "val"):
        path = os.path.join(args.data, f"{split}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        n = len(arr) // meta["ctx"]
        eos = int((arr == eos_id).sum())
        print(f"\n[{split}] {len(arr)/1e6:.1f}M token, {n} 块, EOS {eos} 个 "
              f"(密度 1/{len(arr)/max(1, eos):.0f}), 平均每块 {eos/max(1,n):.2f} 个")
        # byte-fallback 的 piece 形如 <0xE4>，它们出现在词表末尾
        fb = int(((arr >= vocab - 256) & (arr != eos_id)).sum())
        print(f"          byte-fallback token {fb} ({100*fb/max(1,len(arr)):.2f}%)")
        rng = random.Random(args.seed)
        for _ in range(args.chunks):
            k = rng.randrange(max(1, n))
            ids = arr[k * meta["ctx"]:(k + 1) * meta["ctx"]].tolist()
            txt = sp.decode(ids)
            print(f"  --- 第 {k} 块（前 260 字） ---")
            print("  " + txt[:260].replace("\n", "\n  "))
        del arr


if __name__ == "__main__":
    main()
