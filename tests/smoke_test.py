# -*- coding: utf-8 -*-
"""CPU 冒烟测试：不需要数据文件，也不需要预训练权重。

跑通这些说明仓库的「骨架」是好的：
  1. 前向形状对                  (B,T,V) / mem_out (B,K,d)
  2. loss 是有限数，反向梯度有限
  3. targets=-1 的位置真的不算 loss（mask 语义）
  4. 记忆槽接力：第二段带着第一段的 mem_out 也能跑
  5. generate 的 cache 路径与「每步重算」路径结果一致
在 CI 里跑：python tests/smoke_test.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                    # noqa: E402
import torch.nn.functional as F                 # noqa: E402
from model import GPT                           # noqa: E402

V, T, K, D = 128, 32, 4, 32
CFG = dict(vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=D,
           n_mem=K, attn_res=1, qk_norm=1, n_kv_head=1, z_loss=0.0, dropout=0.0)


def main():
    torch.manual_seed(0)
    m = GPT(CFG)
    m.eval()
    B = 2
    x = torch.randint(0, V, (B, T))
    y = x.clone()

    # 1) 前向形状
    logits, loss, mem_out = m(x, y)
    assert logits.shape == (B, T, V), logits.shape
    assert mem_out is not None and mem_out.shape == (B, K, D), mem_out.shape

    # 2) loss 有限 + 反向梯度有限
    assert torch.isfinite(loss), loss
    loss.backward()
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None))
    assert torch.isfinite(gnorm) and gnorm > 0, gnorm
    m.zero_grad(set_to_none=True)

    # 3) targets=-1 的位置不算 loss：手工只在前半段算 CE，必须和模型给的 loss 一致
    y_mask = y.clone()
    y_mask[:, T // 2:] = -1
    _, loss_a, _ = m(x, y_mask)
    manual = F.cross_entropy(logits.reshape(-1, V), y_mask.reshape(-1), ignore_index=-1)
    assert torch.allclose(loss_a, manual, atol=1e-6), (loss_a.item(), manual.item())

    # 4) 记忆槽接力：第二段吃第一段的 mem_out
    _, _, mem1 = m(x, y)
    logits2, loss2, mem2 = m(x, y, mem=mem1)
    assert logits2.shape == (B, T, V) and mem2.shape == (B, K, D)
    assert torch.isfinite(loss2)
    assert not torch.allclose(mem1, mem2), "两个段落的写槽不该完全一样"

    # 5) cache 路径 == 重算路径（贪心解码下逐 token 相同）
    prompt = x[:, :8]
    torch.manual_seed(0)
    out_fast = m.generate(prompt, 6, temperature=0.0, top_k=1)
    torch.manual_seed(0)
    out_slow = m.generate(prompt, 6, temperature=0.0, top_k=1, use_cache=False)
    assert out_fast.shape == (B, 14) and out_slow.shape == (B, 14)
    assert torch.equal(out_fast, out_slow), (out_fast.tolist(), out_slow.tolist())

    # 6) 无记忆槽的老配置也得能跑（向后兼容）
    m0 = GPT(dict(CFG, n_mem=0, attn_res=0))
    lg0, ls0, mo0 = m0(x, y)
    assert lg0.shape == (B, T, V) and mo0 is None and torch.isfinite(ls0)

    n_par = sum(p.numel() for p in m.parameters())
    print(f"ok  forward/backward/mask/memory/cache/legacy 全过  (params {n_par/1e3:.1f}K)")


if __name__ == "__main__":
    main()
