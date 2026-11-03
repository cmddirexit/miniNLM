import os
# -*- coding: utf-8 -*-
"""解码瓶颈剖析：用 profiler 看单 token 的时间都花在哪。"""
import sys, time, math
import torch
sys.path.insert(0, ROOT)
from model import GPT

DEV = "cuda"
ck = torch.load(ROOT + "/out/ckpt_best.pt", map_location="cpu", weights_only=False)
cfg = ck["cfg"]
m = GPT(cfg).to(DEV); m.load_state_dict(ck["model"]); m.eval()
x = torch.randint(0, cfg["vocab_size"], (1, 1024), device=DEV)

with torch.autocast("cuda", dtype=torch.bfloat16):
    m.generate(x, 4, temperature=0.8, top_k=40)   # warmup

from torch.profiler import profile, ProfilerActivity
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with torch.autocast("cuda", dtype=torch.bfloat16):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        m.generate(x, 24, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1.15)

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18))
print()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
