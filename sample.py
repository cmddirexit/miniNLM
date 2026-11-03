# -*- coding: utf-8 -*-
"""miniNLM 采样：给一段开头，续写小说"""
import os, sys, argparse
import torch
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=ROOT + "/out/ckpt_best.pt")
    p.add_argument("--spm", default=ROOT + "/data_v4/spm48k_v4.model")
    p.add_argument("--prompt", default="")
    p.add_argument("--n", type=int, default=300, help="生成 token 数")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--rep_penalty", type=float, default=1.15)
    p.add_argument("--num", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    eos = sp.piece_to_id("<|endoftext|>")

    prompt = args.prompt
    ids = sp.encode(prompt, out_type=int) if prompt else [eos]
    x = torch.tensor([ids], dtype=torch.long, device=args.device)

    for i in range(args.num):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(x, args.n, temperature=args.temperature,
                                 top_k=args.top_k, eos_id=eos,
                                 repetition_penalty=args.rep_penalty)
        text = sp.decode(out[0].tolist())
        print("=" * 60)
        print(prompt + text if i == 0 else text)


if __name__ == "__main__":
    main()
