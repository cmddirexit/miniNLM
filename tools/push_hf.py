# -*- coding: utf-8 -*-
"""把训练好的 checkpoint 打包成 HuggingFace 模型仓库，并可选上传

用法：
    # 1) 只打包到本地目录，先检查产物
    python tools/push_hf.py --staging hf_export

    # 2) 在 https://huggingface.co/settings/tokens 建一个 write token，然后
    export HF_TOKEN=hf_xxxxxxxx
    python tools/push_hf.py --repo 你的用户名/miniNLM-thin-46m

    # 3) 只上传已经打好的目录
    python tools/push_hf.py --repo 你的用户名/miniNLM-thin-46m --upload-only

产物（一个标准的 HF 模型仓库）：
    model.safetensors   weights（tied head 已去重，185 MB）
    config.json         结构超参 + 训练元信息 + tokenizer/eos 说明
    spm48k_v4.model     SentencePiece 词表（vocab 48000）
    model.py            GPT 结构（权重自带架构，不依赖本仓库）
    example.py          独立推理脚本（只依赖 torch/safetensors/sentencepiece）
    README.md           model card
    .gitattributes      LFS 规则（用 API 上传时也保留，方便以后 git clone）
"""
import argparse
import json
import os
import shutil
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def build(args):
    import torch
    from safetensors.torch import save_model
    from model import GPT

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}

    model = GPT(cfg)
    model.load_state_dict(sd)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    non_embed = total - model.wte.weight.numel()
    n_tok = int(ck.get("args", {}).get("trained_tokens", 0) or 0)

    out = args.staging
    if os.path.isdir(out) and not args.upload_only:
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    st_path = os.path.join(out, "model.safetensors")
    save_model(model, st_path, metadata={
        "format": "pt",
        "architecture": "GPT",
        "params": str(total),
    })

    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "mininlm",
            "architectures": ["GPT"],
            "arch": cfg,
            "vocab_size": cfg["vocab_size"],
            "block_size": cfg["block_size"],
            "n_mem": cfg.get("n_mem", 0),
            "tokenizer_file": "spm48k_v4.model",
            "eos_token_id": 4,
            "n_params": total,
            "n_params_non_embed": non_embed,
            "training": {
                "step": int(ck.get("step", 0)),
                "best_val": float(ck.get("best_val", 0.0)),
                "trained_tokens": n_tok,
                "dtype": "fp32",
            },
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")

    shutil.copyfile(args.spm, os.path.join(out, "spm48k_v4.model"))
    shutil.copyfile(os.path.join(ROOT, "model.py"), os.path.join(out, "model.py"))
    shutil.copyfile(os.path.join(ROOT, "tools", "hf_example.py"),
                    os.path.join(out, "example.py"))

    with open(os.path.join(out, ".gitattributes"), "w", encoding="utf-8") as f:
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
                "*.model filter=lfs diff=lfs merge=lfs -text\n")

    card = open(os.path.join(ROOT, "tools", "hf_model_card.md"),
                encoding="utf-8").read()
    rep = {
        "@REPO@": args.repo or "<你的用户名>/miniNLM-thin-46m",
        "@PARAMS@": "%.2fM" % (total / 1e6),
        "@PARAMS_NE@": "%.2fM" % (non_embed / 1e6),
        "@SIZE@": human(os.path.getsize(st_path)),
        "@STEP@": str(int(ck.get("step", 0))),
        "@VAL@": "%.4f" % float(ck.get("best_val", 0.0)),
        "@PPL@": "%.2f" % (pow(2.718281828459045, float(ck.get("best_val", 0.0)))),
        "@CTX@": str(cfg["block_size"]),
        "@NTS@": str(cfg.get("n_mem", 0)),
    }
    for k, v in rep.items():
        card = card.replace(k, v)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    print("staging ->", out)
    for name in sorted(os.listdir(out)):
        p = os.path.join(out, name)
        print("  %-22s %10s" % (name, human(os.path.getsize(p))))
    print("params: %.2fM (non-embed %.2fM)  step %d  best_val %.4f"
          % (total / 1e6, non_embed / 1e6, int(ck.get("step", 0)),
             float(ck.get("best_val", 0.0))))


def upload(args):
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print("logged in as:", who.get("name"))
    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=args.staging,
                      commit_message=args.message,
                      ignore_patterns=["__pycache__/*", "*.pyc"])
    print("done: https://huggingface.co/%s" % args.repo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(ROOT, "out", "ckpt_best.pt"))
    p.add_argument("--spm", default=os.path.join(ROOT, "data_v4", "spm48k_v4.model"))
    p.add_argument("--staging", default=os.path.join(ROOT, "hf_export"))
    p.add_argument("--repo", default=None, help="比如 someone/miniNLM-thin-46m")
    p.add_argument("--private", action="store_true")
    p.add_argument("--message", default="Upload miniNLM thin 46M")
    p.add_argument("--upload-only", action="store_true")
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()

    if not args.upload_only:
        build(args)
    if args.upload_only or not args.no_upload:
        if not args.repo:
            sys.exit("要上传必须给 --repo user/name（或者加 --no-upload 只打包）")
        upload(args)


if __name__ == "__main__":
    main()
