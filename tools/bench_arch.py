import os
# -*- coding: utf-8 -*-
"""架构体检：解码延迟随上下文长度的变化、head 占前向的比例、记忆槽开销。"""
import sys, time, json, math
import torch

sys.path.insert(0, ROOT)
from model import GPT
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda"
ck = torch.load(ROOT + "/out/ckpt_best.pt", map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = GPT(cfg).to(DEV)
model.load_state_dict(ck["model"])
model.eval()
NONEMB = model.num_params(True)
TOTAL = model.num_params(False)
print(f"params total {TOTAL/1e6:.2f}M  non-emb {NONEMB/1e6:.2f}M  cfg {cfg}")

# 每 token 理论 FLOPs（前向）
d, V = cfg["n_embd"], cfg["vocab_size"]
per_block = 3 * d * d + d * d + d * (int(d * 8 / 3 + 63) // 64 * 64) * 2 + (int(d * 8 / 3 + 63) // 64 * 64) * d
body_f = 2 * per_block * cfg["n_layer"]
head_f = 2 * d * V
print(f"FLOPs/token  body {body_f/1e6:.1f}M  head {head_f/1e6:.1f}M  head share {head_f/(body_f+head_f)*100:.1f}%")

# ---------- 1. 解码速度 vs 上下文长度 ----------
print("\n[decode] 每个上下文长度生成 24 token")
rows = []
for ctx in (64, 256, 512, 768, 1024):
    x = torch.randint(0, V, (1, ctx), device=DEV)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.generate(x, 24, temperature=0.8, top_k=40)
    torch.cuda.synchronize(); dt = time.time() - t0
    tps = 24 / dt
    rows.append((ctx, dt, tps))
    print(f"  ctx {ctx:5d}: {dt*1000:7.0f} ms  {tps:6.1f} tok/s  ({tps/ctx*1000:.3f} tok/s per ctx-token)")

# ---------- 2. 前向+反向吞吐：head 占比 ----------
print("\n[fwd+bwd] bs6 x 1024 token, bf16 autocast")
def bench_step(vocab, n_mem, iters=8):
    c = dict(cfg); c["vocab_size"] = vocab; c["n_mem"] = n_mem
    m = GPT(c).to(DEV)
    x = torch.randint(0, vocab, (6, 1024), device=DEV)
    y = torch.randint(0, vocab, (6, 1024), device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = m(x, y); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = m(x, y); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); dt = (time.time() - t0) / iters
    del m, opt; torch.cuda.empty_cache()
    return dt

t32 = bench_step(32000, 32)
t1k = bench_step(1024, 32)
print(f"  vocab 32000: {t32*1000:.0f} ms/step  {6*1024/t32/1000:.1f}k tok/s")
print(f"  vocab  1024: {t1k*1000:.0f} ms/step  {6*1024/t1k/1000:.1f}k tok/s")
print(f"  -> 词表从 32k 缩到 1k 只省 {(t32-t1k)/t32*100:.1f}% 时间（说明 head 的 GEMM 没占 FLOPs 那么多）")

# ---------- 3. 记忆槽开销 ----------
t_mem0 = bench_step(32000, 0)
print(f"  n_mem 0:     {t_mem0*1000:.0f} ms/step  (n_mem=32 时 {t32*1000:.0f} ms -> 开销 {(t32-t_mem0)/t_mem0*100:+.1f}%)")

# ---------- 4. 注意力自身的耗时（单独几何计算） ----------
print("\n[attn] 手写注意力的时间占比估计")
B, H, T_, Dh = 6, 8, 1088, 64
q = torch.randn(B, H, T_, Dh, device=DEV, dtype=torch.bfloat16)
for T_l in (1024,):
    qq = q[:, :, :T_l]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        att = (qq @ qq.transpose(-2, -1)) * (1 / math.sqrt(Dh))
        att = torch.softmax(att, dim=-1)
        y = att @ qq
    torch.cuda.synchronize()
    print(f"  T={T_l}: 单层 attn 核心 {(time.time()-t0)/20*1000:.1f} ms (fwd, bs6)")
