# -*- coding: utf-8 -*-
"""miniNLM 验收评估：验证集整体 loss/ppl + 分段（记忆前/后）分解 + 抽样置信区间

与 train.py 的 evaluate() 完全同协议：
  - 同一份 val.bin、同样的 n_seg 段窗口
  - 随机种子派生方式相同（seed*7919+1，第 k 批用 seed+k），
    因此 --batches 200 的结果应能复现训练日志里的 best_val。
用法：
  python eval_val.py --ckpt out_v6/ckpt_best.pt --batches 400
"""
import os, sys, json, math, time, argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT
from train import get_batch
ROOT = os.path.dirname(os.path.abspath(__file__))


def human(n):
    return f"{n/1e6:.1f}M"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=ROOT + "/out/ckpt_best.pt")
    p.add_argument("--data_dir", default=ROOT + "/data_v4")
    p.add_argument("--batches", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_seg", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ref_batches", type=int, default=200,
                   help="训练时每轮验证用的批数，用于复现对比")
    p.add_argument("--json_out", default="")
    args = p.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = GPT(cfg).to(dev)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    model.eval()

    blk = cfg["block_size"]
    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    mem0 = model.mem0 if cfg.get("n_mem", 0) else None

    print(f"ckpt={os.path.basename(args.ckpt)}  step={ck.get('step')}  "
          f"best_val={ck.get('best_val'):.4f}")
    print(f"模型 {cfg['n_layer']}层 d={cfg['n_embd']} {cfg['n_head']}头  "
          f"n_mem={cfg.get('n_mem')} n_seg={args.n_seg}  "
          f"上下文 {blk * args.n_seg} token  验证集 {human(len(val))} token")
    print(f"批数 {args.batches} x bs {args.batch_size} x {blk} x {args.n_seg}段 "
          f"= {human(args.batches * args.batch_size * blk * args.n_seg)} token")
    print("-" * 72, flush=True)

    seg_sum = [0.0] * args.n_seg
    batch_means = []
    t0 = time.time()
    with torch.no_grad():
        for k in range(args.batches):
            gen = torch.Generator()
            gen.manual_seed((args.seed * 7919 + 1 + k) & 0x7FFFFFFFFFFFFFFF)
            x, y = get_batch(val, blk, args.batch_size, dev, args.n_seg, gen)
            mem = mem0.expand(x.size(0), -1, -1) if mem0 is not None else None
            acc = 0.0
            for s in range(args.n_seg):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss, mem = model(x[:, s], y[:, s], mem=mem)
                seg_sum[s] += loss.item()
                acc += loss.item()
            batch_means.append(acc / args.n_seg)
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{args.batches} 批  已用 {time.time()-t0:.0f}s", flush=True)

    n = args.batches
    bm = np.array(batch_means)
    overall = float(bm.mean())
    se = float(bm.std(ddof=1) / math.sqrt(n))
    ref = float(bm[:args.ref_batches].mean()) if n >= args.ref_batches else None

    print("-" * 72)
    print(f"总体      val_loss {overall:.4f}   val_ppl {math.exp(overall):.2f}   "
          f"±{1.96*se:.4f} (95% CI, 单批标准差 {bm.std(ddof=1):.4f})")
    for s in range(args.n_seg):
        v = seg_sum[s] / n
        print(f"  段 {s}    val_loss {v:.4f}   ppl {math.exp(v):.2f}"
              + ("   （冷启动：只用 mem0，无上下文记忆）" if s == 0 else "   （带上段记忆）"))
    if args.n_seg >= 2:
        d = seg_sum[0] / n - seg_sum[1] / n
        print(f"  记忆收益  段0 - 段1 = {d:+.4f}  "
              f"（{'带上段记忆更好' if d > 0 else '更差'}）")
    if ref is not None:
        print(f"  前 {args.ref_batches} 批   val_loss {ref:.4f}   ppl {math.exp(ref):.2f}  "
              f"← 与训练日志 best_val 同协议")
    print(f"用时 {time.time()-t0:.0f}s")

    if args.json_out:
        out = {
            "ckpt": args.ckpt,
            "step": ck.get("step"),
            "batches": n,
            "batch_size": args.batch_size,
            "n_seg": args.n_seg,
            "block_size": blk,
            "val_loss": overall,
            "val_ppl": math.exp(overall),
            "ci95": 1.96 * se,
            "per_seg": [s / n for s in seg_sum],
            "ref_batches": args.ref_batches,
            "ref_val_loss": ref,
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果写入 {args.json_out}")


if __name__ == "__main__":
    main()
