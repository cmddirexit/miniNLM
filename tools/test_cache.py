# -*- coding: utf-8 -*-
"""校验 KV cache / GQA / QK-norm / z-loss 改动的正确性。

  A. 旧检查点（32k 词表、MHA、无 QK-norm）：cache 路径与逐 token 重算路径的
     每步 logits 一致（atol 3e-3），采样结果一致。
  B. 新配置（GQA + QK-norm + z_loss + attn_res）：前后向可跑、梯度有限非零，
     cache/重算同样一致；含 prompt 长于 block_size 的滚动分段。
  C. generate(use_cache=True/False) 同 seed 同输出（prompt 长于 block_size）。

用法：python tools/test_cache.py [--ckpt out_v6/ckpt_best.pt] [--steps 12]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, sample_token  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 3e-3


def gen_recompute(model, idx, steps, temp=0.8, top_k=40):
    """参考实现：每步把当前段整个前向一遍（= 改造前的 generate 语义）。"""
    T = model.cfg["block_size"]
    K = model.n_mem
    mem = model.mem0.expand(idx.size(0), -1, -1).contiguous() if K else None
    seg_start = 0
    if K and idx.size(1) > T:                  # prompt 超一段：按 T 灌满并滚动 mem
        s = 0
        while idx.size(1) - s > T:
            _, _, mem = model(idx[:, s:s + T], mem=mem)
            s += T
        seg_start = s
    cur = idx
    logits_list, toks = [], []
    with torch.no_grad():
        for _ in range(steps):
            seg = cur[:, seg_start:] if K else cur[:, -T:]
            logits, _, mem_new = model(seg, mem=mem)
            lg = logits[:, -1, :].float()
            logits_list.append(lg.clone())
            nxt = sample_token(lg.clone(), cur, temp, top_k, None, None, 1.0)
            toks.append(nxt)
            cur = torch.cat((cur, nxt), dim=1)
            if K and cur.size(1) - seg_start >= T:
                mem, seg_start = mem_new, cur.size(1) - 1
    return logits_list, toks, cur


def gen_cache(model, idx, steps, temp=0.8, top_k=40):
    """待测实现：KV cache，每步只算 1 个位置。"""
    T = model.cfg["block_size"]
    K = model.n_mem
    B = idx.size(0)
    mem = model.mem0.expand(B, -1, -1).contiguous() if K else None
    cache = [None] * len(model.blocks)
    pos = seg = 0
    pending = idx
    logits_list, toks = [], []
    with torch.no_grad():
        while pending.size(1) > 0:                        # prefill
            if seg >= T:                                  # prompt 超长：先滚动
                mem, cache, pos = model._roll(mem, pos, cache)
                seg = 0
            take = min(T - seg, pending.size(1))
            chunk, pending = pending[:, :take], pending[:, take:]
            if cache[0] is None:                          # 段首，前面带读槽
                x = torch.cat([mem, model.wte(chunk)], 1) if K else model.wte(chunk)
                h = model._run(x, 0, cache)
                pos, seg = x.size(1), seg + take
            else:
                h = model._run(model.wte(chunk), pos, cache)
                pos, seg = pos + chunk.size(1), seg + take
        cur = idx
        for _ in range(steps):                            # decode
            lg = model.head(model.nf(h[:, -1:, :]))[:, -1, :].float()
            logits_list.append(lg.clone())
            nxt = sample_token(lg.clone(), cur, temp, top_k, None, None, 1.0)
            toks.append(nxt)
            cur = torch.cat((cur, nxt), dim=1)
            if seg >= T:
                mem, cache, pos = model._roll(mem, pos, cache)
                seg = 0
            h = model._run(model.wte(nxt), pos, cache)
            pos, seg = pos + 1, seg + 1
    return logits_list, toks, cur


def cmp_logits(tag, a, b, atol=ATOL):
    worst = max((x - y).abs().max().item() for x, y in zip(a, b))
    ok = worst <= atol
    print(f"   {tag}: 最大 logits 偏差 {worst:.2e}  {'OK' if ok else '不一致 ❌'}")
    return ok


def cmp_tokens(tag, a, b):
    ok = all(torch.equal(x, y) for x, y in zip(a, b))
    print(f"   {tag}: 采样 token 序列 {'一致' if ok else '不一致 ❌'}")
    return ok


def both(m, idx, steps, tag):
    """同 seed 下各跑一遍；seed 必须在 idx 造好之后设，否则 RNG 位置不同。"""
    torch.manual_seed(1234)
    la, ta, _ = gen_recompute(m, idx, steps)
    torch.manual_seed(1234)
    lb, tb, _ = gen_cache(m, idx, steps)
    ok = cmp_logits(f"logits({tag})", la, lb)
    ok = cmp_tokens(f"tokens({tag})", ta, tb) and ok
    return ok


def test_a(ckpt, steps):
    print(f"\n[A] 旧检查点 cache vs 重算  ({ckpt})")
    if not os.path.exists(ckpt):
        print("   跳过：检查点不存在")
        return False
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = GPT(cfg).to(DEV).eval()
    m.load_state_dict(ck["model"])
    assert m.n_mem > 0 and not hasattr(m.blocks[0].attn, "q_norm")
    print(f"   cfg: n_layer={cfg['n_layer']} n_embd={cfg['n_embd']} n_mem={m.n_mem} "
          f"vocab={cfg['vocab_size']} 非嵌入参数={m.num_params():,}")
    idx = torch.randint(0, cfg["vocab_size"], (1, 200), device=DEV)
    ok = both(m, idx, steps, "200 token")
    del m
    torch.cuda.empty_cache()
    return ok


def test_b(steps):
    print("\n[B] 新配置 GQA + QK-norm + z_loss + attn_res")
    cfg = dict(n_layer=4, n_head=8, n_embd=256, block_size=128, n_mem=16,
               vocab_size=2000, dropout=0.0, attn_res=1, qk_norm=1,
               n_kv_head=2, z_loss=1e-4)
    torch.manual_seed(1)
    m = GPT(cfg).to(DEV)
    x = torch.randint(0, 2000, (2, 64), device=DEV)
    y = torch.randint(0, 2000, (2, 64), device=DEV)
    logits, loss, mem_out = m(x, y)
    loss.backward()
    a = m.blocks[0].attn
    g1 = a.qkv.weight.grad
    g2 = a.q_norm.w.grad if a.qk_norm else None
    print(f"   forward/backward: loss {loss.item():.4f} logits {tuple(logits.shape)} "
          f"mem_out {tuple(mem_out.shape)}  z_loss={m.z_loss}")
    print(f"   qkv.grad_norm {g1.norm().item():.4f}  q_norm.grad_norm "
          f"{g2.norm().item() if g2 is not None else 'n/a'}")
    ok = torch.isfinite(loss).item() and g1.abs().sum().item() > 0
    ok &= tuple(mem_out.shape) == (2, 16, 256)
    ok &= tuple(logits.shape) == (2, 64, 2000)
    if g2 is not None:
        ok &= torch.isfinite(g2).all().item() and g2.abs().sum().item() > 0

    m.eval()
    idx = torch.randint(0, 2000, (1, 90), device=DEV)
    ok &= both(m, idx, steps, "90 token")
    idx2 = torch.randint(0, 2000, (1, 300), device=DEV)
    ok &= both(m, idx2, steps, "300 token 滚动")
    return ok


def test_c(steps):
    print("\n[C] generate(use_cache=True/False) 同 seed 同输出")
    cfg = dict(n_layer=3, n_head=4, n_embd=128, block_size=64, n_mem=8,
               vocab_size=800, dropout=0.0, attn_res=0, qk_norm=1, n_kv_head=2)
    torch.manual_seed(4)
    m = GPT(cfg).to(DEV).eval()
    prompt = torch.randint(0, 800, (1, 150), device=DEV)   # 长于 block_size
    outs = {}
    for use in (True, False):
        torch.manual_seed(7)
        toks = []
        m.generate(prompt.clone(), max_new_tokens=steps, temperature=0.8, top_k=40,
                   use_cache=use, on_token=toks.append)
        outs[use] = toks
    ok = outs[True] == outs[False]
    print(f"   cache    : {outs[True]}")
    print(f"   recompute: {outs[False]}")
    print(f"   序列{'一致' if ok else '不一致 ❌'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_v6/ckpt_best.pt")
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    print(f"device={DEV}")
    results = {
        "A 旧检查点等价": test_a(args.ckpt, args.steps),
        "B 新配置训练+等价": test_b(args.steps),
        "C generate 两条路径": test_c(args.steps),
    }
    print("\n==== 汇总 ====")
    for k, v in results.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
