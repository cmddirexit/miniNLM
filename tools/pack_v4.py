#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 打包：把清洗后的书编码成 uint16 token 流，按 1024 切块，EOS 留在流里。

与 v3 的 build_data.py pack 的区别：
  * 章节之间已经有 <|endoftext|> 标记 → 打包时自然落在 token 流中，不再「只写整块、把书尾 EOS 丢掉」；
  * 跨书连续缓冲，只在 split 结束时丢末尾不足一块的余数；
  * chunk 级去重保留；
  * meta.json 记录 EOS 统计，便于验证密度。

用法：
    python tools/pack_v4.py --spm data_v4/spm48k_v4.model --out data_v4
"""
import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np

ROOT = ROOT + ""
EOS = "<|endoftext|>"
CTX = 1024

_SP = None
_EOS = None


def init_worker(model_path):
    global _SP, _EOS
    import sentencepiece as spm
    _SP = spm.SentencePieceProcessor(model_file=model_path)
    _EOS = _SP.piece_to_id(EOS)


def tokenize_book(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = _SP.encode(text, out_type=int)
    return np.asarray(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=os.path.join(ROOT, "data_v4", "books"))
    ap.add_argument("--spm", default=os.path.join(ROOT, "data_v4", "spm48k_v4.model"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_v4"))
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit-books", type=int, default=0)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    eos_id = sp.piece_to_id(EOS)
    if eos_id <= 0:
        raise SystemExit(f"[!] 分词器里没有 {EOS}")
    vocab = sp.get_piece_size()
    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(l) for l in open(os.path.join(args.books, "_manifest_v4.jsonl"), encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    if args.limit_books:
        rows = rows[:args.limit_books]

    def is_val(r):
        h = int(hashlib.blake2b(r["id"].encode(), digest_size=8).hexdigest()[:8], 16)
        return h % 100 < int(args.val_frac * 100)

    splits = {"train": [r for r in rows if not is_val(r)], "val": [r for r in rows if is_val(r)]}
    print(f"分词器 {os.path.basename(args.spm)}  vocab={vocab}  eos={eos_id}  ctx={args.ctx}")
    print(f"书 train={len(splits['train'])} val={len(splits['val'])}")

    meta = {"vocab": vocab, "ctx": args.ctx, "eos_piece": EOS, "eos_id": int(eos_id),
            "tokenizer": os.path.basename(args.spm), "norm": "identity",
            "fullwidth_kept": True, "built": time.strftime("%Y-%m-%d %H:%M")}

    for split in ("train", "val"):
        books = splits[split]
        out_path = os.path.join(args.out, f"{split}.bin")
        buf = []
        buf_len = 0
        stats = {"chunks": 0, "dup_chunks": 0, "tokens": 0, "eos_tokens": 0,
                 "chunks_with_eos": 0, "books": len(books)}
        seen = set()
        t0 = time.time()
        with open(out_path, "wb") as fh, \
                Pool(max(1, args.workers), initializer=init_worker, initargs=(args.spm,)) as pool:
            jobs = [os.path.join(args.books, r["id"]) for r in books]
            for ids in pool.imap(tokenize_book, jobs, chunksize=1):
                buf.append(ids)
                buf_len += ids.size
                while buf_len >= args.ctx:
                    cat = np.concatenate(buf)
                    n = cat.size // args.ctx
                    chunk = cat[:n * args.ctx].reshape(n, args.ctx)
                    rest = cat[n * args.ctx:]
                    for k in range(n):
                        row = chunk[k]
                        if split == "train":
                            h = hashlib.blake2b(row.tobytes(), digest_size=8).digest()
                            if h in seen:
                                stats["dup_chunks"] += 1
                                continue
                            seen.add(h)
                        row.tofile(fh)
                        stats["chunks"] += 1
                        stats["tokens"] += args.ctx
                        neos = int((row == eos_id).sum())
                        stats["eos_tokens"] += neos
                        if neos:
                            stats["chunks_with_eos"] += 1
                    buf = [rest] if rest.size else []
                    buf_len = rest.size
        mb = os.path.getsize(out_path) / 1e6
        stats["MB"] = round(mb, 1)
        stats["sec"] = round(time.time() - t0, 1)
        stats["tokens_per_char"] = None
        print(f"  {split}: {stats['chunks']} 块 ({mb:.0f} MB, {stats['tokens']/1e6:.1f}M token)  "
              f"EOS {stats['eos_tokens']} 个，含 EOS 的块 {stats['chunks_with_eos']} "
              f"({100*stats['chunks_with_eos']/max(1,stats['chunks']):.1f}%)  "
              f"重复块 {stats['dup_chunks']}  {stats['sec']}s")
        meta[f"{split}_chunks"] = stats["chunks"]
        meta[f"{split}_tokens"] = stats["tokens"]
        meta[f"{split}_books"] = stats["books"]
        meta[f"{split}_eos"] = stats["eos_tokens"]
        meta[f"{split}_chunks_with_eos"] = stats["chunks_with_eos"]
        if split == "train":
            meta["dup_dropped"] = stats["dup_chunks"]

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nmeta.json:")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
