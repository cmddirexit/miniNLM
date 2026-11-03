#!/usr/bin/env python3
"""长程事实召回评测（向量化版）：记忆槽里到底存了什么？

段 1 = 1024 token（历史）→ 段 2 = 1024 token（目标），三种条件：
  A 带段 1 传来的记忆   B 记忆置零   C 记忆换成同批次另一条样本的（错记忆）

「召回类」目标 = 在段 1 出现过，但不在它前面 128 token 的近期窗口里（只能靠记忆恢复）。
"""
import os, sys, time, collections
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from model import GPT
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV = "cuda"
D = ROOT + "/data_v2"
T = 1024
BS = 4
NWIN = 320
RECENT = 128
TAGS = [("mem", "带记忆"), ("zero", "零记忆"), ("shuf", "错记忆")]


def build_recall(x1, x2, y2):
    """返回 (召回掩码, 距离)。

    召回类 = 目标词在段1出现过，且在段2里是**第一次**出现。
    这样就排除了「段2内部注意力就能抄到」的情况：唯一的来源只剩记忆槽。
    距离 = 该词在段1最后一次出现的位置，距当前位置多远。
    """
    B, Tt = len(x1), len(x1[0])
    rec = np.zeros((B, Tt), dtype=bool)
    dist = np.full((B, Tt), -1, dtype=np.int32)
    for b in range(B):
        last = {}
        for i, tok in enumerate(x1[b]):
            last[tok] = i
        seen = set()
        for t in range(Tt):
            seen.add(x2[b][t])          # 预测 y[t] 时，模型已看到 x[0..t]
            tk = y2[b][t]
            if tk in last and tk not in seen:
                rec[b, t] = True
                dist[b, t] = (Tt - last[tk]) + t
    return rec, dist


def main():
    torch.cuda.set_per_process_memory_fraction(0.75)
    ck = torch.load(ROOT + "/out/ckpt_best.pt", map_location=DEV, weights_only=False)
    model = GPT(ck["cfg"]).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    print("检查点 step %d  n_mem %d" % (ck.get("step", -1), ck["cfg"].get("n_mem", 0)), flush=True)
    val = np.memmap(os.path.join(D, "val.bin"), dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(7)

    vals = {t: {"rec": [], "oth": []} for t, _ in TAGS}
    ranks = {t: [] for t, _ in TAGS}
    bucket = {t: {i: [] for i in range(4)} for t in ("mem", "zero", "shuf")}
    t0 = time.time()

    for bi in range(NWIN // BS):
        ix = torch.randint(len(val) - 2 * T - 1, (BS,), generator=g)
        xs, ys = [], []
        for i in ix.tolist():
            chunk = torch.from_numpy(val[i:i + 2 * T + 1].astype(np.int64))
            xs.append(chunk[:-1]); ys.append(chunk[1:])
        x = torch.stack(xs).to(DEV).view(BS, 2, T)
        y = torch.stack(ys).to(DEV).view(BS, 2, T)

        x1 = x[:, 0].tolist(); x2 = x[:, 1].tolist(); y2 = y[:, 1].tolist()
        rec, dist = build_recall(x1, x2, y2)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, mem1 = model(x[:, 0], y[:, 0], mem=model.mem0.expand(BS, -1, -1))
            mem1 = mem1.float()
            mem_shuf = mem1[torch.randperm(BS, device=DEV)]
            mems = {"mem": mem1, "zero": torch.zeros_like(mem1), "shuf": mem_shuf}

            for tag, _ in TAGS:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _ = model(x[:, 1], y[:, 1], mem=mems[tag])
                logits = logits.float()
                tgt = logits.gather(-1, y[:, 1].unsqueeze(-1))
                r = (logits > tgt).sum(-1).to(torch.int32) + 1
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      y[:, 1].reshape(-1), reduction="none").view(BS, T)
                nll_np = nll.cpu().numpy()
                r_np = r.cpu().numpy()
                del logits, nll, tgt, r

                vals[tag]["rec"].append(nll_np[rec])
                vals[tag]["oth"].append(nll_np[~rec])
                ranks[tag].append(r_np[rec])
                if tag in bucket:
                    for i in range(4):
                        m = rec & (dist >= i * 512) & (dist < (i + 1) * 512)
                        bucket[tag][i].append(nll_np[m])
        if (bi + 1) % 10 == 0:
            print("  ...%d/%d  %.0fs" % (bi + 1, NWIN // BS, time.time() - t0), flush=True)

    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    R = {t: cat(vals[t]["rec"]) for t, _ in TAGS}
    O = {t: cat(vals[t]["oth"]) for t, _ in TAGS}
    K = {t: cat(ranks[t]) for t, _ in TAGS}

    print("\n窗口 %d 个，召回类 token %d 个（占 %.1f%%）" %
          (NWIN, len(R["mem"]), 100.0 * len(R["mem"]) / max(1, len(R["mem"]) + len(O["mem"]))))
    print("\n条件        召回类 loss    其他 loss")
    for t, name in TAGS:
        print("%-8s %12.4f %11.4f" % (name, R[t].mean(), O[t].mean()))

    print("\n关键差值（正数 = 带记忆更好）：")
    print("  召回类  零记忆 − 带记忆 = %+.4f nats" % (R["zero"].mean() - R["mem"].mean()))
    print("  召回类  错记忆 − 带记忆 = %+.4f nats" % (R["shuf"].mean() - R["mem"].mean()))
    print("  其他    零记忆 − 带记忆 = %+.4f nats" % (O["zero"].mean() - O["mem"].mean()))

    print("\n真实 token 排名：")
    for t, name in TAGS:
        k = np.sort(K[t])
        print("  %-6s 中位 %6.0f   平均 %8.1f   top-10 %.2f%%   top-100 %.2f%%"
              % (name, np.median(k), k.mean(), 100.0 * (k <= 10).mean(), 100.0 * (k <= 100).mean()))

    print("\n按距离分段（历史词最后一次出现距当前多远）：")
    print("  距离        样本数   带记忆   错记忆   零记忆   错−带   零−带")
    for i, lab in enumerate(["0-512", "512-1024", "1024-1536", "1536-2048"]):
        a, s, z = cat(bucket["mem"][i]), cat(bucket["shuf"][i]), cat(bucket["zero"][i])
        if len(a):
            print("  %-10s %6d  %7.4f  %7.4f  %7.4f  %+6.4f  %+6.4f"
                  % (lab, len(a), a.mean(), s.mean(), z.mean(),
                     s.mean() - a.mean(), z.mean() - a.mean()))
    print("\n总用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
