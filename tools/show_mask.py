import os
# -*- coding: utf-8 -*-
"""看一眼 mask 长什么样：哪些 token 算 loss，哪些被忽略"""
import sys
sys.path.insert(0, ROOT)
import sentencepiece as spm
from sft_data import render, encode_masked
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sp = spm.SentencePieceProcessor(model_file=ROOT + "/data_v4/spm48k_v4.model")
msgs = [{"role": "user", "content": "你好，你是谁？"},
        {"role": "assistant", "content": "我是 miniNLM，一个中文小模型。"}]
ids, mask = encode_masked(sp, render(msgs))
print("原文:", repr(sp.decode(ids)))
print("token 数:", len(ids), " 算 loss 的:", sum(mask))
print()
print(f"{'token':<16}{'id':>7}  loss")
for i, m in zip(ids, mask):
    print(f"{sp.id_to_piece(i)!r:<16}{i:>7}  {'YES' if m else '-'}")
