# -*- coding: utf-8 -*-
"""
stage2: 训练 sentencepiece BPE
stage3: 打包 uint16 训练集/验证集（含 chunk 级去重）
用法:
  python build_data.py tokenizer
  python build_data.py pack
"""
import os, sys, json, hashlib, random, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = ROOT + "/data"
CLEAN = os.path.join(DATA, "clean")
SPM_PREFIX = os.path.join(DATA, "spm32k")
VOCAB = 32000
CTX = 1024
EOS = "<|endoftext|>"


def train_tokenizer(n_mb=200, seed=0):
    import sentencepiece as spm
    from tqdm import tqdm

    recs = [json.loads(l) for l in open(os.path.join(CLEAN, "manifest.jsonl"), encoding="utf-8")]
    total = sum(r["chars"] for r in recs)
    tmp = os.path.join(DATA, "_spm_sample.txt")

    # 按比例从每本书抽样，凑到 n_mb 兆字节
    target = n_mb * 1000 * 1000
    rng = random.Random(seed)
    written = 0
    with open(tmp, "w", encoding="utf-8") as out:
        order = recs[:]
        rng.shuffle(order)
        for r in tqdm(order, desc="sample"):
            if written >= target:
                break
            need = int(target * r["chars"] / total) * 3  # 3 bytes/汉字 粗估
            need = max(need, 20000)
            t = open(os.path.join(CLEAN, r["file"]), encoding="utf-8").read()
            if len(t) > need:
                start = rng.randrange(0, len(t) - need)
                t = t[start:start + need]
            out.write(t)
            out.write("\n" + EOS + "\n")
            written += len(t.encode("utf-8"))
    print("sample bytes: %.1f MB" % (written / 1e6))

    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=tmp,
        model_prefix=SPM_PREFIX,
        vocab_size=VOCAB,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        user_defined_symbols=[EOS],
        bos_id=1, eos_id=2, unk_id=0, pad_id=3,
        num_threads=os.cpu_count() or 4,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=4096,
        train_extremely_large_corpus=False,
    )
    print("tokenizer trained in %.1fs -> %s.model" % (time.time() - t0, SPM_PREFIX))
    os.remove(tmp)


_SP = None
_EOS = None


def _init_worker():
    global _SP, _EOS
    import sentencepiece as spm
    _SP = spm.SentencePieceProcessor(model_file=SPM_PREFIX + ".model")
    _EOS = _SP.piece_to_id(EOS)


def _tokenize_book(args):
    path, = args
    sp, eos = _SP, _EOS
    t = open(path, encoding="utf-8").read()
    ids = sp.encode(t, out_type=int)
    ids.append(eos)
    return np.asarray(ids, dtype=np.uint16)


def pack(out_dir=None):
    import multiprocessing as mp
    from tqdm import tqdm

    OUT = out_dir or DATA
    os.makedirs(OUT, exist_ok=True)

    recs = [json.loads(l) for l in open(os.path.join(CLEAN, "manifest.jsonl"), encoding="utf-8")]
    recs.sort(key=lambda r: r["file"])

    # 验证集：按书名哈希选 ~5%，且整本留出
    val = [r for r in recs if int(hashlib.md5(r["src"].encode()).hexdigest()[:8], 16) % 20 == 0]
    train = [r for r in recs if r not in val]
    print("books: train=%d val=%d" % (len(train), len(val)))

    seen_chunks = set()
    stats = {"chunks": 0, "dup_chunks": 0, "tokens": 0}

    def flush(ids, fh, is_val):
        nonlocal stats
        n = len(ids) // CTX
        for k in range(n):
            chunk = ids[k * CTX:(k + 1) * CTX]
            h = hashlib.blake2b(chunk.tobytes(), digest_size=8).digest()
            if not is_val:
                if h in seen_chunks:
                    stats["dup_chunks"] += 1
                    continue
                seen_chunks.add(h)
            chunk.tofile(fh)
            stats["chunks"] += 1
            stats["tokens"] += CTX

    for split, books, fname in (("train", train, "train.bin"), ("val", val, "val.bin")):
        out_path = os.path.join(OUT, fname)
        with open(out_path, "wb") as fh, mp.Pool(os.cpu_count() or 4, initializer=_init_worker) as pool:
            job = [(os.path.join(CLEAN, r["file"]),) for r in books]
            for ids in tqdm(pool.imap(_tokenize_book, job, chunksize=1), total=len(job), desc=split):
                flush(ids, fh, split == "val")
        print("  %s -> %.1f MB" % (out_path, os.path.getsize(out_path) / 1e6))

    meta = {
        "vocab": VOCAB, "ctx": CTX,
        "train_chunks": stats["chunks"], "dup_dropped": stats["dup_chunks"],
        "val_books": len(val), "train_books": len(train),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tokenizer"
    if cmd == "tokenizer":
        train_tokenizer()
    elif cmd == "pack":
        pack(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
