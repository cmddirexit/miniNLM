import os
# -*- coding: utf-8 -*-
"""从 log.csv 取验证曲线，拟合 L = a + b * step^-c，外推第二个 epoch 的收益"""
import csv, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))

path = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/out_v6/log.csv"
steps, vals = [], []
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("val_loss"):
            steps.append(int(row["step"]))
            vals.append(float(row["val_loss"]))
steps = np.array(steps, dtype=float)
vals = np.array(vals)

print(f"验证点 {len(vals)} 个，step {int(steps[0])} → {int(steps[-1])}")
print("\n最后 12 个验证点：")
for s, v in list(zip(steps, vals))[-12:]:
    print(f"  step {int(s):7d}   val {v:.4f}")

# 只看后半段拟合，避免 warmup 期干扰
mask = steps >= steps[-1] * 0.25
s, v = steps[mask], vals[mask]
best = None
for c in np.linspace(0.05, 1.5, 600):
    X = np.stack([np.ones_like(s), s ** (-c)], axis=1)
    coef, *_ = np.linalg.lstsq(X, v, rcond=None)
    pred = X @ coef
    sse = float(((v - pred) ** 2).sum())
    if best is None or sse < best[0]:
        best = (sse, c, coef)
sse, c, coef = best
a, b = coef
rmse = (sse / len(v)) ** 0.5
print(f"\n拟合 L = {a:.4f} + {b:.4f} * step^-{c:.3f}   (RMSE {rmse:.4f}, 用 {len(v)} 点)")

last = steps[-1]
for mult, label in [(1.0, "当前"), (1.25, "+25% 步"), (2.0, "第二个 epoch")]:
    st = last * mult
    pred = a + b * st ** (-c)
    print(f"  外推 {label:12s} step {int(st):7d}  val ≈ {pred:.4f}  ppl ≈ {np.exp(pred):.1f}")

# 实测最近 5000 步的下降速度，作为对照
recent = steps >= last - 5000
if recent.sum() >= 2:
    d = vals[recent][0] - vals[recent][-1]
    span = steps[recent][-1] - steps[recent][0]
    print(f"\n实测最近 {int(span)} 步：val 下降 {d:+.4f}（约每 1 万步 {d/span*10000:+.4f}）")
    print(f"按此速度再跑一个 epoch（{int(last)} 步）≈ 再降 {d/span*last:+.4f}  → val ≈ {vals[-1]-d/span*last:.4f}")
