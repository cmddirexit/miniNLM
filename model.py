# -*- coding: utf-8 -*-
"""miniNLM 模型：decoder-only Transformer (RoPE + RMSNorm + SwiGLU + 权重绑定)

可选特性（都写在 cfg 里，旧检查点缺键时自动退化成原行为）：
  · n_mem > 0     记忆槽：段输入 = [读槽 K][本段 T][写槽 K]，段尾写槽做摘要接力
  · attn_res > 0  注意力残差：沿深度方向 softmax 聚合（Attention Residuals, 2026）
  · qk_norm > 0   QK-Norm：q/k 在 RoPE 之前各做一次 RMSNorm（ViT-22B / Qwen3 / Gemma-3 默认做法）
  · n_kv_head     GQA：k/v 头数，默认 = n_head（等价于 MHA）；8 头配 2 组 KV 更省 cache
  · z_loss        PaLM 式 z-loss 系数（建议 1e-4），抑制 logits 漂移

推理：
  generate(..., use_cache=True) 走 KV cache —— prefill 一次，之后每 token 只算 1 个位置；
  段满（T 个 token）时跑一次写槽（K 个位置）拿到新 mem，重置 cache 继续。
  use_cache=False 退回"每 token 重算整段"的老路径（慢 5~15 倍，用来对数）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class AttnRes(nn.Module):
    """注意力残差：沿"深度"方向的注意力聚合。

        w  = stack_i( RMSNorm(s_i) · q_l )        (l, B, T)
        α  = softmax_i(w)                         沿深度轴
        h  = Σ_i α_i · s_i                        (B, T, d)

    s_0 = 词嵌入，s_i = 第 i 层块输出的增量（attn + mlp）。
    每层一个 q_l，共 n_layer + 1 个（多出的一行给最后的读出聚合用）。
    """

    def __init__(self, d, n_layer, eps=1e-6):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(n_layer + 1, d))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, states, q):
        w = torch.stack([(self._norm(s) * q).sum(-1) for s in states], 0)   # (l, B, T)
        a = w.float().softmax(0).to(states[0].dtype)                        # fp32 里做 softmax 更稳
        out = a[0].unsqueeze(-1) * states[0]
        for i in range(1, len(states)):
            out = out + a[i].unsqueeze(-1) * states[i]
        return out


def build_rope(T, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(T, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, S, Dh)  仅在偶数维上旋转；cos/sin: (S, Dh/2)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.n_kv_head = int(cfg.get("n_kv_head", cfg["n_head"]))
        assert self.n_head % self.n_kv_head == 0, "n_head 必须能被 n_kv_head 整除"
        self.n_rep = self.n_head // self.n_kv_head
        self.d_head = d // self.n_head
        self.qk_norm = bool(int(cfg.get("qk_norm", 0)))
        self.qkv = nn.Linear(d, (self.n_head + 2 * self.n_kv_head) * self.d_head, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.get("dropout", 0.0))
        if self.qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x, cos, sin, mask, cache=None, layer=0):
        B, S, C = x.shape
        q, k, v = torch.split(self.qkv(x), [self.n_head * self.d_head,
                                             self.n_kv_head * self.d_head,
                                             self.n_kv_head * self.d_head], dim=2)
        q = q.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_kv_head, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_kv_head, self.d_head).transpose(1, 2)
        if self.qk_norm:
            # QK-Norm：把 q/k 的尺度钉死，attention logits 不再随训练漂移
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            kv = cache[layer]
            if kv is None:
                cache[layer] = [k, v]
            else:
                k = torch.cat((kv[0], k), dim=2)      # 沿时间轴拼历史
                v = torch.cat((kv[1], v), dim=2)
                kv[0], kv[1] = k, v
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 手写注意力：ROCm(gfx1200) 上 SDPA 只有 MATH 后端，实测比手算慢 4 倍
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill_(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, S, C)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["n_embd"]
        h = int(d * cfg.get("mlp_ratio", 8 / 3))
        h = (h + 63) // 64 * 64                      # 对齐到 64
        self.up = nn.Linear(d, 2 * h, bias=False)
        self.down = nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(cfg.get("dropout", 0.0))

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.drop(self.down(F.silu(a) * b))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg["n_embd"])
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg["n_embd"])
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask, cache=None, layer=0):
        x = x + self.attn(self.n1(x), cos, sin, mask, cache=cache, layer=layer)
        x = x + self.mlp(self.n2(x))
        return x


def sample_token(logits, idx, temperature=1.0, top_k=None, top_p=None, min_p=None,
                 repetition_penalty=1.0):
    """logits: (B,V) fp32；idx: (B,n) 已生成的 token（重复惩罚用）。返回 (B,1)。"""
    if repetition_penalty != 1.0:
        for b in range(logits.size(0)):
            uniq = torch.unique(idx[b])
            v = logits[b, uniq]
            # CTRL 式：负 logit 除以 >1 会变大（等于奖励重复），所以按符号分开
            logits[b, uniq] = torch.where(v > 0, v / repetition_penalty,
                                          v * repetition_penalty)
    logits = logits / max(temperature, 1e-5)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        # 核采样：按概率从大到小累加，越过 top_p 的尾巴全砍掉（至少留 1 个）
        sp, si = torch.sort(logits, descending=True, dim=-1)
        pr = F.softmax(sp, dim=-1)
        cut = (torch.cumsum(pr, dim=-1) - pr) > top_p
        logits = torch.full_like(logits, -float("inf")).scatter(
            1, si, sp.masked_fill(cut, -float("inf")))
    if min_p is not None and 0.0 < min_p < 1.0:
        # min_p：低于 最高概率 × min_p 的 token 砍掉，比 top_p 更自适应
        thresh = logits.max(dim=-1, keepdim=True).values + math.log(min_p)
        logits = logits.masked_fill(logits < thresh, -float("inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        T = cfg["block_size"]
        d = cfg["n_embd"]
        self.n_mem = int(cfg.get("n_mem", 0))        # 记忆槽数量 K（读槽、写槽各 K 个）
        # 老检查点里没有这个键 → 默认关，保证旧权重还能加载
        self.use_ar = bool(int(cfg.get("attn_res", 0)))
        self.z_loss = float(cfg.get("z_loss", 0.0))
        L = T + 2 * self.n_mem

        self.wte = nn.Embedding(cfg["vocab_size"], d)
        self.drop = nn.Dropout(cfg.get("dropout", 0.0))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.head.weight = self.wte.weight          # 权重绑定
        if self.use_ar:
            self.res = AttnRes(d, cfg["n_layer"])

        cos, sin = build_rope(L, d // cfg["n_head"], torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))
        if self.n_mem:
            self.mem0 = nn.Parameter(torch.empty(1, self.n_mem, d))
            nn.init.normal_(self.mem0, mean=0.0, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
        return n

    # ------------------------------------------------------------------ 前向
    def _run(self, x, pos0, cache=None):
        """x: (B,S,d) 已嵌入的输入，绝对位置从 pos0 开始；返回 (B,S,d) 隐状态。
        cache 为 None 时是训练路径；给了 list（长度 n_layer，元素 None/[k,v]）就地累积 KV。"""
        S = x.size(1)
        end = pos0 + S
        cos, sin = self.cos[pos0:end], self.sin[pos0:end]
        mask = self.mask[pos0:end, :end]
        x = self.drop(x)
        if self.use_ar:
            # states[0] = 词嵌入，之后每个是"块输出 − 块输入"的增量
            states = [x]
            for i, b in enumerate(self.blocks):
                h = self.res(states, self.res.q[i])
                y = b(h, cos, sin, mask, cache=cache, layer=i)
                states.append(y - h)
            return self.res(states, self.res.q[len(self.blocks)])   # 读出时再聚合一次
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, mask, cache=cache, layer=i)
        return x

    def forward(self, idx, targets=None, mem=None):
        """idx: (B,T)   targets: (B,T)   mem: (B,K,d) 上一段的记忆，None 用初始记忆
        返回 (logits (B,T,V), loss, mem_out (B,K,d) 或 None)"""
        B, T = idx.shape
        K = self.n_mem
        if K:
            mem = self.mem0.expand(B, -1, -1) if mem is None else mem
            x = torch.cat([mem, self.wte(idx), mem], dim=1)   # 读槽 + 本段 + 写槽
        else:
            x = self.wte(idx)
        L = x.size(1)
        x = self._run(x, 0, None)
        x = self.nf(x)
        if K:
            mem_out = x[:, L - K:]                            # 写槽输出 → 下一段读槽
            x = x[:, K:K + T]                                 # 只在真实 token 上出 logits
        else:
            mem_out = None
        logits = self.head(x)
        loss = None
        if targets is not None:
            flat = logits.reshape(-1, logits.size(-1))
            loss = F.cross_entropy(flat, targets.reshape(-1), ignore_index=-1)
            if self.z_loss:
                lse = torch.logsumexp(flat.float(), dim=-1)
                loss = loss + self.z_loss * (lse * lse).mean()
        return logits, loss, mem_out

    # ------------------------------------------------------------------ 生成
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 min_p=None, eos_id=None, repetition_penalty=1.0, on_token=None,
                 use_cache=True):
        """逐 token 采样并返回完整序列。

        use_cache=True：KV cache 路径，每步只算 1 个位置（快 5~15 倍）。
        use_cache=False：每步重算整段（旧行为，用来和 cache 路径对数）。
        on_token(tok_id)：每采样出一个 token 就回调一次，供上层做流式输出。
        """
        if not use_cache:
            return self._generate_nocache(idx, max_new_tokens, temperature, top_k, top_p,
                                          min_p, eos_id, repetition_penalty, on_token)
        if idx.size(1) == 0:
            raise ValueError("generate 需要非空 prompt")

        K, T = self.n_mem, self.cfg["block_size"]
        dtype = self.wte.weight.dtype
        mem = self.mem0.expand(idx.size(0), -1, -1).contiguous().to(dtype) if K else None
        cache = [None] * len(self.blocks)

        # ---- 1. prefill：把 prompt 分块灌进 cache（超出 T 就滚动分段）----
        pos, seg = 0, 0                       # pos = 下一个输入的绝对位置；seg = 本段已放 token 数
        pending = idx
        while pending.size(1) > 0:
            if seg >= T:                      # prompt 比一段还长：先滚动再继续灌
                mem, cache, pos = self._roll(mem, pos, cache)
                seg = 0
            take = min(T - seg, pending.size(1))
            chunk, pending = pending[:, :take], pending[:, take:]
            if cache[0] is None:              # 段首：前面要带读槽
                x = torch.cat([mem, self.wte(chunk)], 1) if K else self.wte(chunk)
                h = self._run(x, 0, cache)
                pos = x.size(1)
            else:
                h = self._run(self.wte(chunk), pos, cache)
                pos += chunk.size(1)
            seg += take

        out = idx
        for _ in range(max_new_tokens):
            logits = self.head(self.nf(h[:, -1:, :]))[:, -1, :].float()
            nxt = sample_token(logits, out, temperature, top_k, top_p, min_p, repetition_penalty)
            out = torch.cat((out, nxt), dim=1)
            if on_token is not None:
                on_token(int(nxt[0, 0]))
            if eos_id is not None and (nxt == eos_id).all():
                break
            if seg >= T:                       # 段满：先把 logits 用掉，再滚动
                mem, cache, pos = self._roll(mem, pos, cache)
                seg = 0
            h = self._run(self.wte(nxt), pos, cache)
            pos += 1
            seg += 1
        return out

    def _roll(self, mem, pos, cache):
        """段满：跑写槽拿新 mem → 清 cache → 把新读槽灌回 cache。返回 (mem, cache, pos)。"""
        K = self.n_mem
        if K:
            mem = self._write_slots(mem, pos, cache)
        cache = [None] * len(self.blocks)
        if K:
            self._run(mem, 0, cache)          # 新读槽占据位置 0..K-1
            return mem, cache, K
        return None, cache, 0

    def _write_slots(self, mem, pos0, cache):
        """段满时把写槽跑一遍，返回新的记忆（B,K,d），与训练的 mem_out 同构。"""
        h = self._run(mem, pos0, cache)
        return self.nf(h)

    def _generate_nocache(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                          min_p=None, eos_id=None, repetition_penalty=1.0, on_token=None):
        """老路径：每步重算整段。保留它做正确性对照和 use_ar 之外的兜底。"""
        K = self.n_mem
        T = self.cfg["block_size"]
        B = idx.size(0)
        mem = self.mem0.expand(B, -1, -1).contiguous() if K else None
        seg_start = 0
        if K and idx.size(1) > T:
            # prompt 比一段还长：按 T 分段灌满并滚动 mem，只留最后不足一段的部分当解码段
            s = 0
            while idx.size(1) - s > T:
                _, _, mem = self(idx[:, s:s + T], mem=mem)
                s += T
            seg_start = s
        for _ in range(max_new_tokens):
            seg = idx[:, seg_start:] if K else idx[:, -T:]
            logits, _, mem_new = self(seg, mem=mem)
            logits = logits[:, -1, :].float()
            nxt = sample_token(logits, idx, temperature, top_k, top_p, min_p, repetition_penalty)
            idx = torch.cat((idx, nxt), dim=1)
            if on_token is not None:
                on_token(int(nxt[0, 0]))
            if K and idx.size(1) - seg_start >= T:
                mem = mem_new
                seg_start = idx.size(1) - 1
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
