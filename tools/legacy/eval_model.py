#!/usr/bin/env python3
"""对训练好的检查点做评测：loss/ppl/bits-per-char、基线对比、生成质量。

用法：python tools/eval.py [--data_dir data_v2] [--out_dir out]
"""
import os, sys, math, json, argparse, collections
import numpy as np
import torch
import sentencepiece as spm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from model import GPT
from train import get_batch, forward_segments, evaluate
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV = "cuda"
PROMPTS = [
    ("标准开篇", "第1章 少年"),
    ("对话", "“你为什么要救我？”"),
    ("年代文", "1980年，江城。"),
    ("玄幻", "萧炎抬起头，看着面前的"),
    ("系统流", "【叮！系统提示："),
    ("悬疑", "他推开门，屋里一片漆黑，"),
    ("家族流", "爷爷说过，我们这一族"),
    ("战争", "枪声在雨夜里"),
    ("超短(压力测试)", "1"),
]


def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    cfg = ck["cfg"]
    model = GPT(cfg).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    return ck, model, cfg


def val_batches(data, block_size, bs, n_seg, nb, seed=0):
    g = torch.Generator().manual_seed(seed)
    span = block_size * n_seg
    out = []
    for _ in range(nb):
        ix = torch.randint(len(data) - span - 1, (bs,), generator=g)
        xs = torch.stack([torch.from_numpy(data[i:i + span].astype(np.int64)) for i in ix])
        ys = torch.stack([torch.from_numpy(data[i + 1:i + 1 + span].astype(np.int64)) for i in ix])
        out.append((xs.view(bs, n_seg, block_size).to(DEV), ys.view(bs, n_seg, block_size).to(DEV)))
    return out


@torch.no_grad()
def eval_fixed(model, batches, n_seg, mem0):
    tot = 0.0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = forward_segments(model, x, y, n_seg, mem0)
        tot += loss.item()
    return tot / len(batches)


@torch.no_grad()
def gen(model, sp, prompt, n=250, temp=0.8, top_k=50, rep=1.15, cfg=None):
    ids = sp.encode(prompt) if prompt else [4]
    idx = torch.tensor([ids], dtype=torch.long, device=DEV)
    out = model.generate(idx, n, temperature=temp, top_k=top_k, repetition_penalty=rep)
    new = out[0, len(ids):].tolist()
    return sp.decode(new), new


def quality(text, ids):
    byte = sum(1 for t in ids if 5 <= t <= 260)
    chars = [c for c in text if not c.isspace()]
    d3 = len(set(tuple(chars[i:i + 3]) for i in range(max(0, len(chars) - 2)))) / max(1, len(chars) - 2)
    return byte / max(1, len(ids)), d3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=ROOT + "/data_v2")
    ap.add_argument("--out_dir", default=ROOT + "/out")
    ap.add_argument("--spm", default=ROOT + "/data/spm32k.model")
    ap.add_argument("--nb", type=int, default=200)
    a = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=a.spm)
    val = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    train = np.memmap(os.path.join(a.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    print("val token %d / train token %d" % (len(val), len(train)))

    # 字数比：解码 200 段验证集统计
    g = np.random.default_rng(0)
    chars = 0; toks = 0
    for _ in range(200):
        i = int(g.integers(0, len(val) - 1024))
        ids = val[i:i + 1024].astype(np.int64).tolist()
        chars += len(sp.decode(ids).replace(" ", "").replace("\n", ""))
        toks += 1024
    cpt = chars / toks
    print("验证集 %.4f 字/token（含标点，去空白）" % cpt)

    # 基线
    uni = np.bincount(np.asarray(train[:50_000_000], dtype=np.int64), minlength=32000).astype(np.float64)
    p = (uni + 1) / (uni.sum() + 32000)
    uni_loss = float(-(p * np.log(p)).sum())
    print("基线：均匀 %.4f nats / 一元统计 %.4f nats" % (math.log(32000), uni_loss))

    res = {}
    for tag in ["ckpt_best.pt", "ckpt_last.pt"]:
        path = os.path.join(a.out_dir, tag)
        if not os.path.exists(path):
            continue
        ck, model, cfg = load(path)
        n_mem = cfg.get("n_mem", 0)
        bs, n_seg = 6, 2
        batches = val_batches(val, cfg["block_size"], bs, n_seg, a.nb)
        loss = eval_fixed(model, batches, n_seg, model.mem0 if n_mem else None)
        ppl = math.exp(loss)
        bpc = loss / cpt / math.log(2)
        print("\n%-14s step %-6d best_val %.4f  n_mem %d" % (tag, ck.get("step", -1), ck.get("best_val", 0), n_mem))
        print("  val_loss %.4f  ppl %.2f  %.3f bits/字  比一元好 %.1f%%"
              % (loss, ppl, bpc, 100 * (1 - loss / uni_loss)))
        res[tag] = (loss, ppl, bpc, model, cfg)

    # 生成
    tag = "ckpt_best.pt" if "ckpt_best.pt" in res else "ckpt_last.pt"
    loss, ppl, bpc, model, cfg = res[tag]
    print("\n=== 生成（%s, temp 0.8, top_k 50, rep 1.15, 250 token）===" % tag)
    for name, prompt in PROMPTS:
        text, ids = gen(model, sp, prompt, 250, 0.8)
        br, d3 = quality(text, ids)
        print("\n[%s] %s" % (name, prompt))
        print("  字节碎片 %.1f%%   distinct-3 %.2f" % (100 * br, d3))
        print("  " + text.replace("\n", "\n  ")[:400])

    print("\n=== 温度扫描（玄幻提示词）===")
    for t in [0.5, 0.7, 0.9, 1.1]:
        text, ids = gen(model, sp, "萧炎抬起头，看着面前的", 120, t)
        br, d3 = quality(text, ids)
        print("temp %.1f  字节碎片 %.1f%%  distinct-3 %.2f | %s" % (t, 100 * br, d3, text[:80].replace("\n", " ")))


if __name__ == "__main__":
    main()
