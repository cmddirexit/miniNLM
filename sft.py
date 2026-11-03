# -*- coding: utf-8 -*-
"""miniNLM 的 SFT（指令微调）：在预训练权重上，只用「回答」部分算 loss

和 train.py 的区别就三点：
  1. 数据多一份 mask.bin，targets 里 mask=0 的位置设成 -1（交叉熵自动忽略）；
  2. 学习率小得多（1e-4，预训练峰值的 1/6），epoch 也只有 2~3；
  3. 每轮结束在固定提示词上采样，方便肉眼对比「训前 / 训后」。
其余（记忆槽、bf16、梯度累积、原子存盘）都沿用预训练那套。

用法：
  python sft.py --sft_dir data_sft --init out_v6/ckpt_best.pt --out_dir out_sft
"""
import os, sys, json, time, math, argparse, csv
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT

from train import fmt_time, progress_bar, atomic_save, lr_at
ROOT = os.path.dirname(os.path.abspath(__file__))

PROBES = [
    "你好，你是谁？",
    "用一句话概括《西游记》讲了什么。",
    "把这句话改简洁：他在心里面反复地想了很久很久。",
    "请接着下面这段往下写，保持同样的风格：\n\n夜色如墨，雨点敲在青石板上。",
]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_dir", default=ROOT + "/data_sft")
    p.add_argument("--init", default=ROOT + "/out/ckpt_best.pt")
    p.add_argument("--out_dir", default=ROOT + "/out_sft")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=5)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0, help="SFT 数据少时可以开 0.05")
    p.add_argument("--n_seg", type=int, default=1)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--probe_every", type=int, default=0, help="每多少步采样一次，0 只在首尾")
    p.add_argument("--gen_tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", dest="compile", action="store_true", default=False)
    p.add_argument("--mem_frac", type=float, default=0.85)
    p.add_argument("--tag", default="sft1")
    return p.parse_args()


def load_sft(path, block_size, batch_size, device, gen):
    """同 train.get_batch，但 targets 按 mask 屏蔽掉提问部分"""
    tok = np.memmap(os.path.join(path, "train.bin"), dtype=np.uint16, mode="r")
    tok_m = np.memmap(os.path.join(path, "train_mask.bin"), dtype=np.uint8, mode="r")
    val = np.memmap(os.path.join(path, "val.bin"), dtype=np.uint16, mode="r")
    val_m = np.memmap(os.path.join(path, "val_mask.bin"), dtype=np.uint8, mode="r")
    return tok, tok_m, val, val_m


def batch_from(tok, msk, T, B, device, gen):
    ix = torch.randint(len(tok) - T - 1, (B,), generator=gen)
    x = torch.empty(B, T, dtype=torch.long)
    y = torch.empty(B, T, dtype=torch.long)
    for j, i in enumerate(ix.tolist()):
        x[j] = torch.from_numpy(tok[i:i + T].astype(np.int64))
        yy = torch.from_numpy(tok[i + 1:i + T + 1].astype(np.int64))
        mm = torch.from_numpy(msk[i + 1:i + T + 1].astype(np.int64))
        y[j] = torch.where(mm > 0, yy, torch.full_like(yy, -1))
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tok, msk, T, B, device, n_batches, seed):
    model.eval()
    tot, n = 0.0, 0
    for k in range(n_batches):
        g = torch.Generator(); g.manual_seed((seed + k) & 0x7FFFFFFFFFFFFFFF)
        x, y = batch_from(tok, msk, T, B, device, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        tot += loss.item(); n += 1
    model.train()
    m = tot / max(n, 1)
    return m, math.exp(min(m, 20))


@torch.no_grad()
def probe(raw_model, sp, out_path, temperature=0.75, top_k=40, rep=1.15, n_tokens=120):
    raw_model.eval()
    lines = [f"# 采样  {time.strftime('%F %T')}\n"]
    for q in PROBES:
        prompt = "\n用户：" + q + "\n助手："
        ids = torch.tensor([sp.encode(prompt)], dtype=torch.long, device="cuda")
        out = raw_model.generate(ids, n_tokens, temperature=temperature, top_k=top_k,
                                 eos_id=4, repetition_penalty=rep)
        txt = sp.decode(out[0, ids.size(1):].tolist())
        lines.append(f"### 用户：{q}\n\n[temperature={temperature}] {txt.strip()}\n")
    raw_model.train()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return lines


def main():
    args = parse()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mem_frac:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    import sentencepiece as spm
    with open(os.path.join(args.sft_dir, "meta.json")) as f:
        smeta = json.load(f)
    sp = spm.SentencePieceProcessor(model_file=ROOT + "/data_v4/spm48k_v4.model")

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["dropout"] = args.dropout
    cfg["block_size"] = args.block_size
    print("=" * 78, flush=True)
    print(f"miniNLM SFT  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  初始化    {args.init}  (预训练 step {ck.get('step')}  best_val {ck.get('best_val'):.4f})", flush=True)
    print(f"  结构      {cfg['n_layer']}层 d={cfg['n_embd']} {cfg['n_head']}头 "
          f"n_mem={cfg.get('n_mem')} block={cfg['block_size']}", flush=True)
    print(f"  数据      {smeta['train_samples']} 条训练 / {smeta['val_samples']} 条验证  "
          f"{smeta['train_tokens']:,} token（回答部分 {smeta['train_answer_tokens']:,}）", flush=True)

    tok, tok_m, val, val_m = load_sft(args.sft_dir, cfg["block_size"], args.batch_size, device, None)
    tokens_per_step = args.batch_size * args.grad_accum * cfg["block_size"] * args.n_seg
    if args.max_steps < 0:
        args.max_steps = max(1, int(len(tok) // tokens_per_step * args.epochs))
    print(f"  批次      bs {args.batch_size} x accum {args.grad_accum} x {cfg['block_size']} "
          f"= {tokens_per_step} token/step", flush=True)
    print(f"  计划      {args.max_steps} 步（{args.epochs} epoch，约 {args.max_steps*tokens_per_step/1e6:.1f}M token）"
          f"  lr {args.lr:g}→{args.min_lr:g}", flush=True)
    print("=" * 78, flush=True)

    raw_model = GPT(cfg).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    missing, unexpected = raw_model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] 加载权重 missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    mem0 = raw_model.mem0 if cfg.get("n_mem", 0) else None
    model = torch.compile(raw_model) if args.compile else raw_model

    decay = [p for n, p in raw_model.named_parameters() if p.dim() >= 2 and not n.endswith("res.q")]
    nodecay = [p for p in raw_model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                               {"params": nodecay, "weight_decay": 0.0}],
                              lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    log_path = os.path.join(args.out_dir, "log.csv")
    log_f = open(log_path, "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["step", "lr", "train_loss", "val_loss", "val_ppl", "elapsed_s"])
    log_f.flush()

    v0, p0 = evaluate(raw_model, val, val_m, cfg["block_size"], args.batch_size, device,
                      args.eval_batches, args.seed * 7919 + 1)
    print(f"[基线] SFT 之前 验证 loss {v0:.4f}  ppl {p0:.2f}（只统计回答部分）", flush=True)
    probe(raw_model, sp, os.path.join(args.out_dir, "before.md"), n_tokens=args.gen_tokens)

    best_val = v0
    model.train()
    t0 = time.time()
    run_loss, n_acc = 0.0, 0

    def save(path, step, with_opt):
        obj = {"model": raw_model.state_dict(), "cfg": cfg, "step": step,
               "best_val": best_val, "args": vars(args)}
        if with_opt:
            obj["optim"] = optim.state_dict()
        atomic_save(obj, path)

    for step in range(args.max_steps):
        lr = lr_at(step, args, args.max_steps)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        gen = torch.Generator(); gen.manual_seed((args.seed * 1000003 + step) & 0x7FFFFFFFFFFFFFFF)
        for _ in range(args.grad_accum):
            x, y = batch_from(tok, tok_m, cfg["block_size"], args.batch_size, device, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y, mem=mem0.expand(x.size(0), -1, -1) if mem0 is not None else None)
            (loss / args.grad_accum).backward()
            run_loss += loss.item(); n_acc += 1
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optim.step()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            avg = run_loss / max(n_acc, 1)
            el = time.time() - t0
            print(f"[{progress_bar((step+1)/args.max_steps)}] {100*(step+1)/args.max_steps:5.1f}%  "
                  f"step {step+1:>5}/{args.max_steps}  loss {avg:6.4f}  lr {lr:.2e}  已用 {fmt_time(el)}",
                  flush=True)
            writer.writerow([step, f"{lr:.3e}", f"{avg:.4f}", "", "", f"{el:.1f}"]); log_f.flush()
            run_loss, n_acc = 0.0, 0

        if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
            vl, vp = evaluate(raw_model, val, val_m, cfg["block_size"], args.batch_size, device,
                              args.eval_batches, args.seed * 7919 + 1)
            flag = ""
            if vl < best_val:
                best_val = vl
                save(os.path.join(args.out_dir, "ckpt_sft_best.pt"), step, with_opt=False)
                flag = "  ★最优"
            print(f"    ├─ 验证 @ step {step+1}: loss {vl:.4f}  ppl {vp:.2f}  (最优 {best_val:.4f}){flag}",
                  flush=True)
            writer.writerow([step, "", "", f"{vl:.4f}", f"{vp:.2f}", f"{time.time()-t0:.1f}"]); log_f.flush()

        if args.probe_every and (step + 1) % args.probe_every == 0:
            probe(raw_model, sp, os.path.join(args.out_dir, f"probe_{step+1}.md"), n_tokens=args.gen_tokens)

        if (step + 1) % args.save_interval == 0:
            save(os.path.join(args.out_dir, "ckpt_sft.pt"), step, with_opt=True)

    save(os.path.join(args.out_dir, "ckpt_sft_last.pt"), args.max_steps - 1, with_opt=True)
    lines = probe(raw_model, sp, os.path.join(args.out_dir, "after.md"), n_tokens=args.gen_tokens)
    print(f"[done] {(time.time()-t0)/60:.1f} 分钟  best_val {best_val:.4f}  采样写到 {args.out_dir}/after.md", flush=True)


if __name__ == "__main__":
    main()
