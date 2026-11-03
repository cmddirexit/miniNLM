# -*- coding: utf-8 -*-
"""miniNLM 预训练：单卡 bf16 + AdamW + 余弦退火 + 梯度累积 + 记忆槽 + torch.compile"""
import os, sys, json, time, math, argparse, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT
ROOT = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=1024),
    "base":  dict(n_layer=8, n_head=8, n_embd=512, block_size=1024),
    "wide":  dict(n_layer=6, n_head=8, n_embd=640, block_size=1024),
    "deep":  dict(n_layer=16, n_head=8, n_embd=512, block_size=1024),
    # 深而窄（2026-10）：同参数下更深更好（MobileLLM / SmolLM2 的宽深比 19~37）。
    # d=384: 宽/深 21，体部 27.9M + 嵌入 18.4M ≈ 46M
    "thin":    dict(n_layer=18, n_head=8, n_embd=384, block_size=1024),
    # d=320: 宽/深 11，追 MobileLLM「sub-1B 最优深度 ~30 层」；注意力 FLOPs 会多 2.3 倍
    "thin30":  dict(n_layer=30, n_head=8, n_embd=320, block_size=1024),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=ROOT + "/data_v4")
    p.add_argument("--out_dir", default=ROOT + "/out")
    p.add_argument("--preset", default="base", choices=list(PRESETS))
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--n_mem", type=int, default=32, help="记忆槽数量，0 关闭")
    p.add_argument("--attn_res", type=int, default=1, help="注意力残差，1 开 / 0 关（默认开）")
    p.add_argument("--n_kv_head", type=int, default=0,
                   help="KV 头数，0 = 等于 n_head（MHA）；小于 n_head 即 GQA，如 2")
    p.add_argument("--qk_norm", type=int, default=0, help="QK-Norm，1 开 / 0 关")
    p.add_argument("--z_loss", type=float, default=0.0, help="logits 的 z-loss 系数，0 关闭")
    p.add_argument("--n_seg", type=int, default=2, help="每步串几段（段间传记忆）")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=-1, help="-1 = 12 // n_seg 自动")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=40)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=500, help="每多少步存一次 ckpt.pt")
    p.add_argument("--resume", default="", help="ckpt 路径，或 'auto' 用 out_dir/ckpt.pt")
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--mem_frac", type=float, default=0.85, help="显存上限占比，给桌面留余量")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--tag", default="base")
    return p.parse_args()


def fmt_time(sec):
    """把秒数格式化成 1h23m / 4m07s"""
    sec = max(0, int(sec))
    if sec >= 3600:
        return f"{sec//3600}h{sec%3600//60:02d}m"
    if sec >= 60:
        return f"{sec//60}m{sec%60:02d}s"
    return f"{sec}s"


def progress_bar(frac, width=24):
    frac = min(1.0, max(0.0, frac))
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


def atomic_save(obj, path):
    """先写 .tmp 再原子替换：训练中随时可以安全读取上一个完整检查点"""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def get_batch(data, block_size, batch_size, device, n_seg=1, gen=None):
    """返回 (B, n_seg, T) 的 x/y；同一条样本内 n_seg 段在原文里是连续的

    gen 给定时用这个生成器抽窗口，使窗口只取决于 (seed, 绝对 step)，
    断点续训不会从头重放同一批样本。
    """
    span = block_size * n_seg
    ix = torch.randint(len(data) - span - 1, (batch_size,), generator=gen)
    xs = torch.empty(batch_size, n_seg, block_size, dtype=torch.long)
    ys = torch.empty(batch_size, n_seg, block_size, dtype=torch.long)
    for j, i in enumerate(ix.tolist()):
        chunk = torch.from_numpy(data[i:i + span + 1].astype(np.int64))
        xs[j] = chunk[:-1].view(n_seg, block_size)
        ys[j] = chunk[1:].view(n_seg, block_size)
    return xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)


def forward_segments(model, x, y, n_seg, mem0=None):
    """段 0 用初始记忆，段 s 用段 s-1 的写槽输出；返回各段平均 loss"""
    B = x.size(0)
    mem = mem0.expand(B, -1, -1) if mem0 is not None else None
    total = None
    for s in range(n_seg):
        _, loss, mem = model(x[:, s], y[:, s], mem=mem)
        total = loss if total is None else total + loss
    return total / n_seg


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, n_batches, n_seg=1, mem0=None, seed=None):
    model.eval()
    losses = torch.zeros(n_batches)
    for k in range(n_batches):
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed((seed + k) & 0x7FFFFFFFFFFFFFFF)
        x, y = get_batch(data, block_size, batch_size, device, n_seg, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = forward_segments(model, x, y, n_seg, mem0)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item(), torch.exp(losses.mean()).item()


def lr_at(step, args, max_steps):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step > max_steps:
        return args.min_lr
    ratio = (step - args.warmup) / max(1, max_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * ratio))


def main():
    args = parse()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mem_frac and 0 < args.mem_frac < 1:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = dict(PRESETS[args.preset])
    for k in ("n_layer", "n_head", "n_embd", "block_size"):
        v = getattr(args, k)
        if v:
            cfg[k] = v
    cfg["n_mem"] = args.n_mem
    cfg["attn_res"] = args.attn_res
    cfg["n_kv_head"] = args.n_kv_head or cfg["n_head"]
    cfg["qk_norm"] = args.qk_norm
    cfg["z_loss"] = args.z_loss
    if args.batch_size <= 0:
        args.batch_size = max(1, 12 // max(1, args.n_seg))

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    cfg["vocab_size"] = meta["vocab"]
    cfg["dropout"] = args.dropout

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    tokens_per_step = args.batch_size * args.grad_accum * cfg["block_size"] * args.n_seg
    if args.max_steps < 0:
        args.max_steps = max(1, int(len(train_data) // tokens_per_step * args.epochs))
    total_tokens = args.max_steps * tokens_per_step

    raw_model = GPT(cfg).to(device)
    mem0 = raw_model.mem0 if cfg.get("n_mem", 0) else None
    print("=" * 78, flush=True)
    print(f"miniNLM 预训练  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  模型      {args.preset}: {cfg['n_layer']}层 d={cfg['n_embd']} {cfg['n_head']}头  "
          f"参数 {raw_model.num_params(False)/1e6:.2f}M (非嵌入 {raw_model.num_params()/1e6:.2f}M)", flush=True)
    print(f"  记忆槽    n_mem={cfg['n_mem']}  n_seg={args.n_seg}  "
          f"有效上下文 {cfg['block_size']*args.n_seg} token", flush=True)
    print(f"  残差      {'注意力残差 AttnRes（深度方向 softmax 聚合）' if cfg.get('attn_res') else '标准残差'}",
          flush=True)
    print(f"  注意力    {'MHA' if cfg['n_kv_head'] == cfg['n_head'] else 'GQA'}"
          f"（KV 头 {cfg['n_kv_head']}/{cfg['n_head']}）  "
          f"QK-Norm {'开' if cfg.get('qk_norm') else '关'}  "
          f"z-loss {cfg.get('z_loss', 0.0):g}", flush=True)
    print(f"  数据      训练 {len(train_data)/1e6:.1f}M token  验证 {len(val_data)/1e6:.1f}M token", flush=True)
    print(f"  批次      bs {args.batch_size} x accum {args.grad_accum} x {cfg['block_size']} x {args.n_seg}段 "
          f"= {tokens_per_step} token/step", flush=True)
    print(f"  计划      {args.max_steps} 步  {total_tokens/1e9:.2f}B token  "
          f"({total_tokens/max(1, raw_model.num_params()):.1f} token/参数)", flush=True)
    print(f"  保存      每 {args.save_interval} 步写 ckpt.pt（原子替换），验证改善时写 ckpt_best.pt", flush=True)
    print("=" * 78, flush=True)

    decay = [p for n, p in raw_model.named_parameters()
             if p.dim() >= 2 and not n.endswith("res.q")]
    nodecay = [p for p in raw_model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8,
    )

    start_step, best_val = 0, 1e9
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    stop_path = os.path.join(args.out_dir, "STOP")
    resume = args.resume
    if resume == "auto":
        resume = ckpt_path if os.path.exists(ckpt_path) else ""
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        raw_model.load_state_dict(sd)
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", 1e9)
        print(f"[resume] {resume} @ step {start_step}  best_val {best_val:.4f}", flush=True)

    model = torch.compile(raw_model) if args.compile else raw_model
    if args.compile:
        print("[compile] torch.compile 首次约 30s ...", flush=True)

    log_path = os.path.join(args.out_dir, "log.csv")
    new_log = start_step == 0 or not os.path.exists(log_path)
    log_f = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["step", "lr", "train_loss", "train_ppl", "val_loss", "val_ppl",
                         "tok_per_s", "elapsed_s"])
        log_f.flush()

    model.train()
    t0 = time.time()
    run_loss, n_acc = 0.0, 0
    last_val = float("nan")

    def save(path, step, with_opt):
        obj = {"model": raw_model.state_dict(), "cfg": cfg, "step": step,
               "best_val": best_val, "args": vars(args)}
        if with_opt:
            obj["optim"] = optim.state_dict()
        atomic_save(obj, path)

    try:
        for step in range(start_step, args.max_steps):
            if os.path.exists(stop_path):
                save(ckpt_path, step, with_opt=True)
                print(f"    ├─ 看到停止标记，已保存 {ckpt_path} @ step {step}", flush=True)
                print("[停止] 删掉 %s 后可用 --resume auto 继续" % stop_path, flush=True)
                sys.exit(0)
            lr = lr_at(step, args, args.max_steps)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.zero_grad(set_to_none=True)
            step_gen = torch.Generator()
            step_gen.manual_seed((args.seed * 1000003 + step) & 0x7FFFFFFFFFFFFFFF)
            for _ in range(args.grad_accum):
                x, y = get_batch(train_data, cfg["block_size"], args.batch_size, device,
                                 args.n_seg, step_gen)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = forward_segments(model, x, y, args.n_seg, mem0)
                (loss / args.grad_accum).backward()
                run_loss += loss.item()
                n_acc += 1
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optim.step()

            if step % args.log_interval == 0 or step == args.max_steps - 1:
                el = time.time() - t0
                done_steps = step - start_step + 1
                tps = done_steps * tokens_per_step / max(el, 1e-6)
                frac = (step + 1) / args.max_steps
                eta = (args.max_steps - step - 1) * el / max(done_steps, 1)
                avg = run_loss / max(n_acc, 1)
                print(f"[{progress_bar(frac)}] {frac*100:5.1f}%  "
                      f"step {step+1:>6}/{args.max_steps}  "
                      f"loss {avg:6.4f}  ppl {math.exp(min(avg, 20)):8.1f}  "
                      f"lr {lr:.2e}  {tps/1000:5.1f}k tok/s  "
                      f"已用 {fmt_time(el)}  剩余 {fmt_time(eta)}", flush=True)
                writer.writerow([step, f"{lr:.3e}", f"{avg:.4f}", f"{math.exp(min(avg, 20)):.1f}",
                                 "", "", f"{tps:.0f}", f"{el:.1f}"])
                log_f.flush()
                run_loss, n_acc = 0.0, 0

            if step > 0 and step % args.eval_interval == 0:
                ev0 = time.time()
                vl, vp = evaluate(model, val_data, cfg["block_size"], args.batch_size,
                                  device, args.eval_batches, args.n_seg, mem0,
                                  seed=args.seed * 7919 + 1)
                last_val = vl
                flag = ""
                if vl < best_val:
                    best_val = vl
                    save(os.path.join(args.out_dir, "ckpt_best.pt"), step, with_opt=False)
                    flag = "  ★最优，已保存 ckpt_best.pt"
                print(f"    ├─ 验证 @ step {step}: val_loss {vl:.4f}  val_ppl {vp:.2f}  "
                      f"(最优 {best_val:.4f})  用时 {fmt_time(time.time()-ev0)}{flag}", flush=True)
                writer.writerow([step, f"{lr:.3e}", "", "", f"{vl:.4f}", f"{vp:.2f}", "",
                                 f"{time.time()-t0:.1f}"])
                log_f.flush()

            if step > 0 and step % args.save_interval == 0:
                save(ckpt_path, step, with_opt=True)
                print(f"    ├─ 已保存 {ckpt_path} @ step {step}", flush=True)

        save(os.path.join(args.out_dir, "ckpt_last.pt"), args.max_steps - 1, with_opt=True)
        print(f"[done] {(time.time()-t0)/60:.1f} 分钟  best_val {best_val:.4f}  last_val {last_val:.4f}",
              flush=True)
        # 完成标记：run_train.sh 看到 DONE 就不再重启（否则每轮都会从 ckpt.pt 重放最后一段）。
        # 想继续加训（提高 --max_steps 或换 --epochs）时先删掉这个文件。
        try:
            with open(os.path.join(args.out_dir, "DONE"), "w") as f:
                f.write(f"step {args.max_steps}  best_val {best_val:.4f}  last_val {last_val:.4f}  "
                        f"{time.strftime('%F %T')}\n")
        except Exception as e:
            print(f"[warn] 写 DONE 失败: {e}", flush=True)
    except KeyboardInterrupt:
        save(ckpt_path, step, with_opt=True)
        print(f"\n[中断] 已保存 {ckpt_path} @ step {step}，可用 --resume auto 继续", flush=True)


if __name__ == "__main__":
    main()
