# -*- coding: utf-8 -*-
"""CPU 上核对 preset 的参数结构（不占显存，可以在训练跑着的时候用）。

    python tools/check_presets.py                 # 只算参数
    python tools/check_presets.py --run           # 再跑一次 64 长度的小前向+反向
    python tools/check_presets.py --preset thin   # 只看某一个

参数口径与 train.py 一致（48k 词表 + GQA kv头2 + qk_norm + attn_res + n_mem32）。
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GPT
from train import PRESETS

VOCAB = 48000
COMMON = dict(n_head=8, block_size=1024, n_mem=32, attn_res=1, n_kv_head=2,
              qk_norm=1, z_loss=1e-4, dropout=0.0, vocab_size=VOCAB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="")
    p.add_argument("--run", action="store_true", help="真跑一次 64 token 的前向+反向")
    args = p.parse_args()

    names = [args.preset] if args.preset else list(PRESETS)
    print("%-8s %9s %9s %7s %7s %8s %6s" % ("preset", "total", "body", "emb%", "wide/deep", "FLOPs/tok", "tok/s*"))
    for n in names:
        cfg = dict(COMMON, **PRESETS[n])
        m = GPT(cfg)
        tot = sum(q.numel() for q in m.parameters())
        emb = m.wte.weight.numel()
        body = tot - emb
        L, d, T = cfg["n_layer"], cfg["n_embd"], cfg["block_size"]
        attn = 4 * T * d * L
        head = 2 * d * VOCAB
        flops = 2 * body + attn + head
        # 解码速度按 kernel 启动次数估算：实测 base(8 层) 约 110 tok/s、420 次启动/token
        tps = 110.0 * 8.0 / L
        print("%-8s %8.2fM %8.2fM %6.0f%% %7.1f %7.0fM %7.0f"
              % (n, tot / 1e6, body / 1e6, 100.0 * emb / tot, d / L, flops / 1e6, tps))
        if args.run:
            torch.manual_seed(0)
            idx = torch.randint(0, VOCAB, (2, 64))
            t0 = time.time()
            logits, loss, mem = m(idx, targets=idx)
            loss.backward()
            dt = time.time() - t0
            assert logits.shape == (2, 64, VOCAB), logits.shape
            assert mem.shape == (2, cfg["n_mem"], d), mem.shape
            assert torch.isfinite(loss), loss
            torch.manual_seed(0)
            out = list(m.generate(idx[:1, :8], 3, temperature=0.8, top_k=20))
            assert len(out) == 3
            print("         fwd+bwd ok  loss=%.4f  %.2fs   generate ok" % (loss.item(), dt))
    print("\n*tok/s 是「按层数线性折算」的粗估，实际还受 batch/占用影响")


if __name__ == "__main__":
    main()
