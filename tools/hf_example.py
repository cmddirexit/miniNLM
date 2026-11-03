# -*- coding: utf-8 -*-
"""miniNLM 独立推理示例

只依赖 torch + safetensors + sentencepiece，用同目录下的
model.safetensors / config.json / spm48k_v4.model / model.py。

    pip install torch safetensors sentencepiece
    python example.py --prompt "他推开客栈的门，"
    python example.py --prompt "他推开客栈的门，" --n 200 --temperature 0.75 --top_k 40
"""
import argparse
import json
import math
import os

import torch
import sentencepiece as spm
from safetensors.torch import load_model

from model import GPT

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model_and_tokenizer(path=HERE, device="cpu"):
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        conf = json.load(f)
    model = GPT(conf["arch"])
    load_model(model, os.path.join(path, "model.safetensors"))
    model.to(device).eval()
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(path, conf.get("tokenizer_file", "spm48k_v4.model")))
    return model, sp, conf


@torch.no_grad()
def generate(model, sp, prompt, n=200, temperature=0.75, top_k=40,
             rep_penalty=1.15, eos_id=None, device="cpu"):
    ids = sp.encode(prompt, out_type=int) if prompt else [eos_id]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, n, temperature=temperature, top_k=top_k,
                         eos_id=eos_id, repetition_penalty=rep_penalty)
    return sp.decode(out[0][len(ids):].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=HERE, help="模型目录（默认脚本所在目录）")
    p.add_argument("--prompt", default="")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--rep_penalty", type=float, default=1.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu / cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, sp, conf = load_model_and_tokenizer(args.path, args.device)
    prompt = args.prompt
    if not prompt:
        prompt = "他推开客栈的门，"
    print("=" * 60)
    print(prompt + generate(model, sp, prompt, args.n, args.temperature,
                            args.top_k, args.rep_penalty,
                            conf.get("eos_token_id"), args.device))


if __name__ == "__main__":
    main()
