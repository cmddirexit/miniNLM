#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 分词器：identity 归一化（不再做 nmt_nfkc，全角标点原样保留）+ 词表大小对比。

流程：
  1. 从 data_v4/books 随机切片采样 ~300MB 文本（保留 <|endoftext|> 标记）；
  2. 分别训练 32k / 48k 的 BPE（byte_fallback）；
  3. 在同一份留出样本上比压缩率（字/token、字节/token、往返一致性、全角标点保留）。

用法：
    python tools/train_spm_v4.py                 # 训练 + 对比
    python tools/train_spm_v4.py --vocab 48000   # 只训一个词表
"""
import argparse
import json
import os
import random
import time

ROOT = ROOT + ""
BOOKS = os.path.join(ROOT, "data_v4", "books")
OUT = os.path.join(ROOT, "data_v4")
EOS = "<|endoftext|>"


def build_sample(books_dir, out_path, target_mb, seed=0, slice_chars=60000):
    rows = [json.loads(l) for l in open(os.path.join(books_dir, "_manifest_v4.jsonl"), encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    rng = random.Random(seed)
    target = target_mb * 1000 * 1000
    written = 0
    n = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        order = rows[:]
        rng.shuffle(order)
        for r in order:
            if written >= target:
                break
            p = os.path.join(books_dir, r["id"])
            try:
                with open(p, encoding="utf-8") as f:
                    f.seek(0)
                    t = f.read()
            except Exception:
                continue
            if len(t) > slice_chars * 2:
                start = rng.randrange(0, len(t) - slice_chars)
                t = t[start:start + slice_chars]
            t = t.strip()
            if not t:
                continue
            out.write(t + "\n")
            written += len(t.encode("utf-8"))
            n += 1
    print(f"[sample] {n} 本切片，{written/1e6:.1f} MB，{time.time()-t0:.0f}s -> {out_path}")
    return out_path


def train_one(sample, prefix, vocab, threads):
    import sentencepiece as spm
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=sample,
        model_prefix=prefix,
        vocab_size=vocab,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        user_defined_symbols=[EOS],
        bos_id=1, eos_id=2, unk_id=0, pad_id=3,
        normalization_rule_name="identity",     # 关键：不做 NFKC，全角标点原样保留
        add_dummy_prefix=False,
        remove_extra_whitespaces=True,
        split_by_whitespace=True,
        num_threads=threads,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        train_extremely_large_corpus=True,
        minloglevel=1,
    )
    print(f"[spm] {os.path.basename(prefix)} vocab={vocab} 用时 {time.time()-t0:.0f}s")
    return prefix + ".model"


def evaluate(model_path, held_out, tag):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    text = open(held_out, encoding="utf-8").read()
    t0 = time.time()
    ids = sp.encode(text, out_type=int)
    enc_sec = time.time() - t0
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    n_tok = len(ids)
    back = sp.decode(ids)
    # 去掉 EOS 标记与空白差异后比较往返一致性
    a = text.replace(EOS, "").strip()
    b = back.replace(EOS, "").strip()
    roundtrip = (a == b)
    eos_id = sp.piece_to_id(EOS)
    res = {
        "tag": tag, "vocab": sp.get_piece_size(), "tokens": n_tok,
        "chars_per_token": round(n_chars / n_tok, 4),
        "bytes_per_token": round(n_bytes / n_tok, 4),
        "enc_MBps": round(n_bytes / 1e6 / max(enc_sec, 1e-9), 2),
        "roundtrip": roundtrip, "eos_id": eos_id,
        "eos_tokens": ids.count(eos_id),
        "fullwidth_kept": ("，" in back and "。" in back and "，" in text),
        "unk": ids.count(0),
    }
    print(f"  [{tag}] vocab={res['vocab']} {res['chars_per_token']} 字/token  "
          f"{res['bytes_per_token']} B/token  往返{'一致' if roundtrip else '不一致'}  "
          f"eos={res['eos_id']} 出现 {res['eos_tokens']} 次  编码 {res['enc_MBps']} MB/s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=BOOKS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sample-mb", type=float, default=300)
    ap.add_argument("--vocab", nargs="*", type=int, default=[32000, 48000])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--reuse-sample", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sample = os.path.join(args.out, "_spm_sample.txt")
    held = os.path.join(args.out, "_spm_heldout.txt")
    if not (args.reuse_sample and os.path.exists(sample)):
        build_sample(args.books, sample, args.sample_mb, seed=0)
    if not os.path.exists(held):
        build_sample(args.books, held, 20, seed=99, slice_chars=20000)

    results = []
    for v in args.vocab:
        prefix = os.path.join(args.out, "spm%dk_v4" % (v // 1000))
        model = train_one(sample, prefix, v, args.threads)
        results.append(evaluate(model, held, "spm%dk_v4" % (v // 1000)))

    print("\n=== 对比 ===")
    base = results[0]
    for r in results:
        delta = (1 - r["bytes_per_token"] / base["bytes_per_token"]) * 100
        print(f"  {r['tag']}: {r['bytes_per_token']} B/token, 相对 {base['tag']} 省 "
              f"{delta:+.2f}% token 数, 往返{r['roundtrip']}")
    with open(os.path.join(args.out, "_spm_compare.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
