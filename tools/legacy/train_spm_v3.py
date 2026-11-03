#!/usr/bin/env python3
"""在新语料 data_v3 上训练 48k 词表，并与现有 32k 词表对比压缩率。

用法：python tools/train_spm_v3.py
输出：data_v3/spm48k.model / spm48k.vocab，并打印两套词表的实测对比。
"""
import json, os, random, re, hashlib, sys

import sentencepiece as spm
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = ROOT + "/data_v3"
BOOKS = os.path.join(DATA, "books")
SAMPLE_BYTES = 60 * 1024 * 1024
VOCAB = 48000


def load_books():
    out = []
    with open(os.path.join(BOOKS, "_manifest_final.jsonl"), encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def is_val(b):
    return int(hashlib.md5(b["id"].encode()).hexdigest()[:8], 16) % 20 == 0


def main():
    books = load_books()
    random.seed(0)
    total = sum(b["chars"] for b in books)
    print("书 %d 本 / %.2f 亿字" % (len(books), total / 1e8))

    # 按字数比例抽样，单本上限 4%
    cap = total * 0.04
    quota = min(1.0, SAMPLE_BYTES / total)
    parts = []
    got = 0
    for b in books:
        n = int(min(b["chars"], cap) * quota)
        if n < 2000:
            continue
        with open(os.path.join(BOOKS, b["id"]), encoding="utf-8", errors="replace") as f:
            t = f.read(n)
        parts.append(t)
        got += len(t)
    print("抽样 %d 本合计 %.1f MB" % (len(parts), got / 1e6))
    random.shuffle(parts)

    src = os.path.join(DATA, "_spm_sample48k.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    del parts

    model = os.path.join(DATA, "spm48k.model")
    print("训练 48k 词表...")
    spm.SentencePieceTrainer.train(
        input=src, model_prefix=model[:-6], vocab_size=VOCAB, model_type="bpe",
        character_coverage=0.9995, byte_fallback=True,
        user_defined_symbols=["<|endoftext|>"],
        bos_id=1, eos_id=2, unk_id=0, pad_id=3,
        num_threads=os.cpu_count() or 4, input_sentence_size=800000,
        shuffle_input_sentence=True,
    )
    os.remove(src)
    print("词表已存 %s" % model)

    # 对比：在 val 书里随机取 2MB 文本
    val = [b for b in books if is_val(b)]
    random.shuffle(val)
    text = ""
    for b in val[:40]:
        with open(os.path.join(BOOKS, b["id"]), encoding="utf-8", errors="replace") as f:
            text += f.read(60000)
        if len(text) > 2000000:
            break
    print("\n评测文本 %.2f M 字" % (len(text) / 1e6))

    for tag, path in [("32k(现有)", ROOT + "/data/spm32k.model"), ("48k(新)", model)]:
        sp = spm.SentencePieceProcessor(model_file=path)
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        n = len(pieces)
        lens = [len(p.replace("▁", "")) for p in pieces]
        one = sum(1 for x in lens if x == 1) / n
        two = sum(1 for x in lens if x == 2) / n
        print("%-10s vocab %d  token %d  %.3f 字/token  单字token %.1f%%  双字token %.1f%%"
              % (tag, sp.vocab_size(), len(ids), len(text) / len(ids), 100 * one, 100 * two))


if __name__ == "__main__":
    main()
