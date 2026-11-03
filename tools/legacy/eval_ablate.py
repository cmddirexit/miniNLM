#!/usr/bin/env python3
"""补充评测：记忆槽到底有没有用 + 长文生成会不会崩。"""
import os, sys, math, json
import numpy as np
import torch
import sentencepiece as spm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from model import GPT
from train import get_batch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV = "cuda"
D = ROOT + "/data_v2"


def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = GPT(ck["cfg"]).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    return ck, m


@torch.no_grad()
def main():
    ck, model = load(ROOT + "/out/ckpt_best.pt")
    cfg = ck["cfg"]
    T = cfg["block_size"]
    val = np.memmap(os.path.join(D, "val.bin"), dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(1)
    bs, nb = 6, 150
    span = T * 2

    l0 = l1_mem = l1_nomem = 0.0
    for _ in range(nb):
        ix = torch.randint(len(val) - span - 1, (bs,), generator=g)
        xs = torch.stack([torch.from_numpy(val[i:i + span].astype(np.int64)) for i in ix]).to(DEV)
        ys = torch.stack([torch.from_numpy(val[i + 1:i + 1 + span].astype(np.int64)) for i in ix]).to(DEV)
        x = xs.view(bs, 2, T); y = ys.view(bs, 2, T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mem0 = model.mem0.expand(bs, -1, -1)
            _, a, mem_new = model(x[:, 0], y[:, 0], mem=mem0)
            _, b, _ = model(x[:, 1], y[:, 1], mem=mem_new)
            zeros = torch.zeros_like(mem0)
            _, c, _ = model(x[:, 1], y[:, 1], mem=zeros)
        l0 += a.item(); l1_mem += b.item(); l1_nomem += c.item()
    l0 /= nb; l1_mem /= nb; l1_nomem /= nb
    print("段 0 loss            %.4f" % l0)
    print("段 1 带记忆          %.4f  (ppl %.1f)" % (l1_mem, math.exp(l1_mem)))
    print("段 1 记忆置零        %.4f  (ppl %.1f)" % (l1_nomem, math.exp(l1_nomem)))
    print("记忆带来的增益       %.4f nats = %.2f%% ppl 降低"
          % (l1_nomem - l1_mem, 100 * (1 - math.exp(l1_mem - l1_nomem))))

    # 长文生成
    sp = spm.SentencePieceProcessor(model_file=ROOT + "/data/spm32k.model")
    ids = sp.encode("第1章 少年")
    idx = torch.tensor([ids], dtype=torch.long, device=DEV)
    out = model.generate(idx, 2400, temperature=0.8, top_k=50, repetition_penalty=1.15)
    new = out[0, len(ids):].tolist()
    print("\n长文生成 2400 token，按 600 一段看：")
    for s in range(0, 2400, 600):
        seg = new[s:s + 600]
        txt = sp.decode(seg)
        chars = [c for c in txt if not c.isspace()]
        d3 = len(set(tuple(chars[i:i + 3]) for i in range(max(0, len(chars) - 2)))) / max(1, len(chars) - 2)
        byte = sum(1 for t in seg if 5 <= t <= 260) / max(1, len(seg))
        print("  %4d-%4d  字节碎片 %.1f%%  distinct-3 %.2f | %s"
              % (s, s + 600, 100 * byte, d3, txt[:60].replace("\n", " ")))


if __name__ == "__main__":
    main()
