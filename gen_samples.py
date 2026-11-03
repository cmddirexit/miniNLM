# -*- coding: utf-8 -*-
"""miniNLM 验收：多提示词批量采样，输出 markdown（一次加载模型，省得反复读 185MB）"""
import os, sys, time, argparse

import torch
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT
ROOT = os.path.dirname(os.path.abspath(__file__))

CASES = [
    ("夜色渐深，城市的霓虹一盏盏亮起，", 0.9, 50, 260, 2),
    ("他推开门的瞬间，房间里弥漫着淡淡的血腥味。", 0.8, 50, 260, 2),
    ("“你确定要这么做？”她低声问，指尖停在半空。", 0.9, 50, 220, 2),
    ("", 1.0, 50, 240, 1),
    ("东海市，清晨六点。", 0.7, 40, 200, 1),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=ROOT + "/out/ckpt_best.pt")
    p.add_argument("--spm", default=ROOT + "/data_v4/spm48k_v4.model")
    p.add_argument("--out", default=ROOT + "/out/samples_v6.md")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = GPT(cfg).to(args.device)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    eos = sp.piece_to_id("<|endoftext|>")

    lines = []
    lines.append(f"# miniNLM 采样结果\n")
    lines.append(f"- ckpt: `{os.path.basename(args.ckpt)}` (step {ck.get('step')}, best_val {ck.get('best_val'):.4f})")
    lines.append(f"- 模型: {cfg['n_layer']} 层 d={cfg['n_embd']} {cfg['n_head']} 头, n_mem={cfg.get('n_mem')}, 上下文 {cfg['block_size']}")
    lines.append(f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- {len(CASES)} 组提示词，每组按设定采样次数生成；空提示=从 <|endoftext|> 直接起头\n")

    with torch.no_grad():
        for ci, (prompt, temp, top_k, n, num) in enumerate(CASES, 1):
            lines.append(f"## {ci}. 提示词（温度 {temp}, top_k {top_k}, 生成 {n} token）\n")
            lines.append("```text")
            lines.append(prompt if prompt else "<|endoftext|>（空提示）")
            lines.append("```\n")
            ids = sp.encode(prompt, out_type=int) if prompt else [eos]
            x = torch.tensor([ids], dtype=torch.long, device=args.device)
            for j in range(num):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.generate(x, n, temperature=temp, top_k=top_k,
                                         eos_id=eos, repetition_penalty=1.15)
                text = sp.decode(out[0].tolist())[len(prompt):] if prompt else sp.decode(out[0].tolist())
                lines.append(f"**样本 {j+1}**\n")
                lines.append("```text")
                lines.append((prompt + text) if prompt else text)
                lines.append("```\n")
            print(f"[{ci}/{len(CASES)}] 完成", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
