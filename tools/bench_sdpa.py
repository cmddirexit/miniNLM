import os
# -*- coding: utf-8 -*-
"""实验 A：ROCm 上 SDPA 各后端是否可用（对比手写注意力）
实验 B：torch.compile 后的解码速度（对比现状 24 token/229ms）"""
import sys, time, math
import torch
import torch.nn.functional as F
sys.path.insert(0, ROOT)
from model import GPT
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda"
B, H, T, Dh = 6, 8, 1024, 64
q = torch.randn(B, H, T, Dh, device=DEV, dtype=torch.bfloat16)
k = torch.randn(B, H, T, Dh, device=DEV, dtype=torch.bfloat16)
v = torch.randn(B, H, T, Dh, device=DEV, dtype=torch.bfloat16)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=DEV), 1)

def timeit(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); return (time.time() - t0) / iters

def manual(q, k, v, mask):
    att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(Dh))
    att = att.masked_fill(mask, float("-inf"))
    att = torch.softmax(att, dim=-1)
    return att @ v

print("== A. SDPA 后端 ==")
print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled(),
      "| mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled(),
      "| math:", torch.backends.cuda.math_sdp_enabled())
t_manual = timeit(lambda: manual(q, k, v, mask))
print(f"手写 attn (fwd, bs6x8x1024): {t_manual*1000:.2f} ms")
for name, ctx in (
    ("math", dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)),
    ("mem_efficient", dict(enable_flash=False, enable_math=False, enable_mem_efficient=True)),
    ("flash", dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)),
):
    try:
        with torch.backends.cuda.sdp_kernel(**ctx):
            t = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        print(f"SDPA[{name:12s}] {t*1000:7.2f} ms   speedup vs manual {t_manual/t:.2f}x")
    except Exception as e:
        print(f"SDPA[{name:12s}] 失败: {type(e).__name__}: {e}")

# fwd+bwd 对比（含 8 层循环的真实场景：只比 attn 部分）
print("\n== B. torch.compile 解码 ==")
ck = torch.load(ROOT + "/out/ckpt_best.pt", map_location="cpu", weights_only=False)
cfg = ck["cfg"]
m = GPT(cfg).to(DEV); m.load_state_dict(ck["model"]); m.eval()
x = torch.randint(0, cfg["vocab_size"], (1, 1024), device=DEV)

with torch.autocast("cuda", dtype=torch.bfloat16):
    m.generate(x, 4, 2.0, top_k=40)   # warmup
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    m.generate(x, 24, 0.8, top_k=40, top_p=0.95, repetition_penalty=1.15)
torch.cuda.synchronize(); dt = time.time() - t0
print(f"eager decode: {dt*1000:.0f} ms / 24 tok = {dt/24*1000:.1f} ms/tok  ({24/dt:.1f} tok/s)")

mc = torch.compile(m, mode="reduce-overhead")
with torch.autocast("cuda", dtype=torch.bfloat16):
    mc.generate(x, 4, 2.0, top_k=40)   # warmup / compile
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    mc.generate(x, 24, 0.8, top_k=40, top_p=0.95, repetition_penalty=1.15)
torch.cuda.synchronize(); dt = time.time() - t0
print(f"compiled decode: {dt*1000:.0f} ms / 24 tok = {dt/24*1000:.1f} ms/tok  ({24/dt:.1f} tok/s)")
