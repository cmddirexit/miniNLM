# -*- coding: utf-8 -*-
"""用 SFT 后的权重按「用户/助手」格式对话。

  python ask.py --q "你好，你是谁？"
  python ask.py            # 交互模式，输入空行退出
"""
import argparse, os, sys, json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT
ROOT = os.path.dirname(os.path.abspath(__file__))

SPM = ROOT + "/data_v4/spm48k_v4.model"
EOS_ID = 4


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=ROOT + "/out_sft/ckpt_sft_best.pt")
    p.add_argument("--q", default="")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--rep", type=float, default=1.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=SPM)
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["dropout"] = 0.0
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
    print(f"[load] {args.ckpt}  step {ck.get('step')}  val {ck.get('best_val', float('nan')):.4f}")

    torch.manual_seed(args.seed)
    history = "\n"
    if args.q:
        history += "用户：" + args.q + "\n助手："
    while True:
        if not args.q:
            try:
                q = input("\n用户：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            history += "用户：" + q + "\n助手："
        ids = torch.tensor([sp.encode(history)], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(ids, args.n, temperature=args.temperature, top_k=args.top_k,
                                 eos_id=EOS_ID, repetition_penalty=args.rep)
        new = out[0, ids.size(1):].tolist()
        if EOS_ID in new:
            new = new[:new.index(EOS_ID)]
        ans = sp.decode(new).strip()
        print("助手：" + ans)
        history += ans + "<|endoftext|>\n"
        if args.q:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
