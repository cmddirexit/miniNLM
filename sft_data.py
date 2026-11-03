# -*- coding: utf-8 -*-
"""SFT 数据构建：JSONL 对话 → (tokens.bin, mask.bin)

每行一条样本：
  {"messages":[{"role":"user","content":"问"},{"role":"assistant","content":"答"}, ...]}

渲染成纯文本（不往词表里加新词，用模型预训练时见过的中文写法）：
  \\n用户：问\\n助手：答<|endoftext|>

mask：只在「助手：」后面的回答、以及结尾的 <|endoftext|> 上是 1，
      问题部分是 0 —— 训练时这些位置的 loss 会被设成 -1 丢掉。
      （为什么要丢：模型不需要学「怎么提问」，只需要学「怎么回答」；
        不丢的话它会把一半算力花在模仿用户提问上。）
"""
import argparse, json, os, random, sys
import numpy as np
import sentencepiece as spm
ROOT = os.path.dirname(os.path.abspath(__file__))

USER, ASSIST, SYS = "\n用户：", "\n助手：", "\n系统："
EOS_PIECE = "<|endoftext|>"


def render(msgs):
    """把消息列表切成 [(文本片段, mask), ...]

    分段编码而不是「整段编码再对字符偏移」：推理时 prompt 也是单独编码的，
    这样训练看到的 token 序列和推理时完全一致（句首/句尾的分词边界不会漂）。
    """
    segs = []
    for m in msgs:
        role = (m.get("role") or "").strip()
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if role == "user":
            segs.append((USER + c, 0))
        elif role == "system":
            segs.append((SYS + c, 0))
        elif role == "assistant":
            segs.append((ASSIST, 0))            # 「助手：」是提示，不算 loss
            segs.append((c, 1))                 # 回答本体：算 loss
            segs.append((EOS_PIECE, 1))         # 结束符：让它学会什么时候停
    return segs


def encode_masked(sp, segs):
    ids, mask = [], []
    for text, m in segs:
        t = sp.encode(text)
        ids += t
        mask += [m] * len(t)
    return ids, mask


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[skip] {path}:{ln} JSON 解析失败: {e}")
                continue
            msgs = obj.get("messages") if isinstance(obj, dict) else obj
            if not isinstance(msgs, list) or not msgs:
                continue
            out.append(msgs)
    return out


def from_books(books_dir, n, sp=None, seed=0):
    """从已有小说语料里造「续写」指令样本：前半段当指令，后半段当答案。"""
    files = [os.path.join(books_dir, f) for f in os.listdir(books_dir)
             if f.endswith(".txt") and os.path.getsize(os.path.join(books_dir, f)) > 20000]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = rng.choice(files)
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                t = f.read()
        except Exception:
            continue
        if len(t) < 3000:
            continue
        s = rng.randrange(0, len(t) - 2000)
        chunk = t[s:s + 1400].strip()
        if len(chunk) < 1200:
            continue
        cut = 700
        head, tail = chunk[:cut].strip(), chunk[cut:].strip()
        out.append([
            {"role": "user", "content": "请接着下面这段往下写，保持同样的风格和人物口吻：\n\n" + head},
            {"role": "assistant", "content": tail},
        ])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", default=[], help="一个或多个 jsonl")
    ap.add_argument("--books", default="", help="可选：从小说目录再造续写样本")
    ap.add_argument("--books_n", type=int, default=200)
    ap.add_argument("--out_dir", default=ROOT + "/data_sft")
    ap.add_argument("--sp", default=ROOT + "/data_v4/spm48k_v4.model")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_tokens", type=int, default=2048, help="单条样本上限，超了截断")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.sp)
    eos_id = sp.piece_to_id(EOS_PIECE)
    assert eos_id >= 0, "词表里没有 <|endoftext|>"
    print(f"词表 {sp.vocab_size()}，EOS = {eos_id}")

    samples = []
    for p in args.inputs:
        d = load_jsonl(p)
        print(f"{p}: {len(d)} 条")
        samples += d
    if args.books:
        d = from_books(args.books, args.books_n, seed=args.seed)
        print(f"{args.books}: 造了 {len(d)} 条续写样本")
        samples += d

    rows = []
    for msgs in samples:
        segs = render(msgs)
        if not any(m for _, m in segs):
            continue                                # 没有助手回复的样本没法训练
        ids, mask = encode_masked(sp, segs)
        if len(ids) > args.max_tokens:
            ids, mask = ids[:args.max_tokens], mask[:args.max_tokens]
            if mask[-1] == 0:                        # 截断到问题中间就丢掉
                continue
        rows.append((ids, mask))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac)) if len(rows) > 4 else 0
    parts = {"val": rows[:n_val], "train": rows[n_val:]}

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"vocab": sp.vocab_size(), "eos_id": eos_id, "template": "\\n用户：Q\\n助手：A<|endoftext|>",
            "source": args.inputs, "books": args.books}
    for split, rs in parts.items():
        ids = np.concatenate([np.array(r[0], dtype=np.uint16) for r in rs]) if rs else np.zeros(0, np.uint16)
        msk = np.concatenate([np.array(r[1], dtype=np.uint8) for r in rs]) if rs else np.zeros(0, np.uint8)
        ids.tofile(os.path.join(args.out_dir, f"{split}.bin"))
        msk.tofile(os.path.join(args.out_dir, f"{split}_mask.bin"))
        meta[f"{split}_samples"] = len(rs)
        meta[f"{split}_tokens"] = int(len(ids))
        meta[f"{split}_answer_tokens"] = int(msk.sum())
        print(f"{split}: {len(rs)} 条  {len(ids):,} token（其中回答 {int(msk.sum()):,}，"
              f"占 {100*msk.mean() if len(msk) else 0:.1f}%）")

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"写到 {args.out_dir}")


if __name__ == "__main__":
    main()
