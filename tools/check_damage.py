import os
# -*- coding: utf-8 -*-
"""诊断：SFT 之后模型在「原始小说验证集」上还有多好？顺便看几种采样设置"""
import sys, json
sys.path.insert(0, ROOT)
import numpy as np, torch
import sentencepiece as spm
from model import GPT
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPM = ROOT + "/data_v4/spm48k_v4.model"
val = np.memmap(ROOT + "/data_v4/val.bin", dtype=np.uint16, mode="r")


def load(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    cfg = dict(ck["cfg"]); cfg["dropout"] = 0.0
    m = GPT(cfg).to("cuda").eval()
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
    return m, ck, cfg


@torch.no_grad()
def ppl(m, cfg, n=20, bs=4, seg=2):
    tot = 0.0
    for k in range(n):
        g = torch.Generator(); g.manual_seed(1000 + k)
        ix = torch.randint(len(val) - cfg["block_size"] * seg - 1, (bs,), generator=g)
        xs = torch.stack([torch.from_numpy(val[i:i + cfg["block_size"] * seg].astype(np.int64)) for i in ix.tolist()])
        xs = xs.view(bs, seg, cfg["block_size"]).cuda()
        mem = m.mem0.expand(bs, -1, -1)
        s = 0.0
        for t in range(seg):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, mem = m(xs[:, t, :-1], xs[:, t, 1:], mem=mem)
            s += loss.item()
        tot += s / seg
    return tot / n


sp = spm.SentencePieceProcessor(model_file=SPM)
prompt = "\n用户：你好，你是谁？\n助手："
ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device="cuda")

for name, path in [("base", ROOT + "/out_v6/ckpt_best.pt"),
                   ("sft_C", ROOT + "/out_sft_c/ckpt_sft_best.pt"),
                   ("sft_A", ROOT + "/out_sft/ckpt_sft.pt")]:
    m, ck, cfg = load(path)
    l = ppl(m, cfg)
    print(f"\n=== {name} ({path})  小说验证集 loss {l:.4f}  ppl {np.exp(l):.1f}")
    for temp in (0.3, 0.75):
        out = m.generate(ids, 40, temperature=temp, top_k=40, eos_id=4, repetition_penalty=1.1)
        print(f"  temp={temp}: {sp.decode(out[0, ids.size(1):].tolist())!r}")
    out = m.generate(ids, 30, temperature=0.05, top_k=1, eos_id=4)
    print(f"  greedy  : {sp.decode(out[0, ids.size(1):].tolist())!r}")
    del m
    torch.cuda.empty_cache()
