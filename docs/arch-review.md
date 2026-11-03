# miniNLM 架构体检 + 论文改进候选（2026-10-27）

> 目的：把 `model.py` 现有架构逐层拆开，量出参数量 / FLOPs / 实测瓶颈，
> 再对照 2024–2026 的论文，列出"值得加进架构"的改动，按 收益÷成本 排序。
> 本文只做分析，不改代码。

---

## 0. 三句话结论

1. **架构本身没毛病**：标准的 LLaMA 式 decoder（RoPE + RMSNorm + SwiGLU + 权重绑定 + pre-norm），
   该有的都有，没有任何"写错了"的地方。缺陷都在"论文里人人都有、这里没有"的那一层。
2. **最大的坑不在架构，在推理**：`generate()` 没有 KV cache，每生成 1 个 token 要把 1024+ 长度的整段
   重算一遍 → 24 个 token 触发 **10,087 次 kernel 启动（≈420 次/token）**，CPU 时间 331.9 ms、
   GPU 时间只有 70.8 ms。**85 tok/s 里 95% 是启动开销和重复计算，不是算力不够。**
   加 KV cache + 静态 shape 是 5–15× 的速度，比任何架构改动都值。
3. **训练侧最值得抄的两件事**：QK-Norm（20 行，稳定 + 允许更高 LR）和 Muon（60 行，社区普遍报告
   1.3–2× token 效率）。这两个是"零风险、立刻能 A/B"的。真正的结构改动（更深更窄、value residual、
   MTP、looped depth）要排到后面，而且必须小规模筛。

---

## 1. 现状盘点

### 1.1 结构清单

| 组件 | 实现 | 参数量 | 说明 |
|---|---|---|---|
| 词嵌入 | `nn.Embedding(32000, 512)` | 16.38 M | 与 head 权重绑定 |
| 注意力 | `qkv` 一个 Linear(512→1536) + `proj` | 786,432 + 262,144 / 层 | 8 头 × 64 维，手写 (q@kᵀ)·mask·softmax·(att@v) |
| RoPE | `build_rope` 预建 cos/sin buffer，偶数维旋转 | 0 | base=10000，长度 L = T + 2K = 1088 |
| FFN | SwiGLU，h = 1408（8/3 d 对齐 64） | 1,441,792 + 720,896 / 层 | `up` 输出 2h 后 chunk |
| 归一化 | RMSNorm（仅 weight，eps 1e-5） | 512×2/层 | pre-norm |
| 残差 | 标准 `x = x + f(x)`；`attn_res` 选项实现了但**旧 ckpt 没开** | `res.q`(9×512) 若有 | 见 §3.9 |
| 记忆槽 | `mem0`(1×32×512) + 段尾写槽 | 16,384 | 读槽 32 + 本段 1024 + 写槽 32 |
| 输出 | `head` 与 `wte` 绑定 | 0（共享） | 无 bias |
| 初始化 | 全部 N(0, 0.02)，`proj`/`down` 再缩 1/√(2L) | — | 没有 muP，没有 QK-norm |
| 精度 | bf16 autocast 训练，无 master weights | — | AdamW(0.9,0.95)，wd 0.1，clip 1.0 |

**每层 3,211,264 参数 × 8 = 25.69 M（非嵌入）+ 嵌入 16.38 M = 42.10 M**
（换 v4 的 48k 词表 → 50.30 M，非嵌入不变）

### 1.2 FLOPs 分解（前向，每 token）

| 项 | 32k 词表 | 48k 词表 | 占比 |
|---|---|---|---|
| 注意力投影 qkv+proj | 16.8 MFLOP | 16.8 MFLOP | 17% |
| FFN up+down | 34.6 MFLOP | 34.6 MFLOP | 34% |
| 注意力核心（1024 上下文）| 17.2 MFLOP | 17.2 MFLOP | 17% |
| LM head | 32.8 MFLOP | 49.2 MFLOP | **32% / 42%** |
| 合计 | **101.4 MFLOP** | **117.8 MFLOP** | |

- head 是单项最大开销（词表 48k 时接近一半）。这不是 bug，是"小 d + 大词表"的必然结果。
- 注意力核心只占 17%：**1024 上下文下，"换更省算力的注意力"收益很小**，
  这也是 §3 里 MLA / NSA / 线性注意力排在最后的原因。
- 实测印证：词表 32000→1024，单步 338 ms → 271 ms（**省 19.8%**），比 FLOPs 占比略低（GEMM 效率差异）。

### 1.3 实测性能（本次体检，RX 9060 XT / ROCm 6.4 / torch 2.6）

| 场景 | 数值 | 备注 |
|---|---|---|
| 训练 | 41.4k tok/s（compile，bs6×accum4×1024×2 段） | ≈12.5 TFLOP/s 有效，约 25–30% MFU |
| 单步（bs6×1024，无 compile，含 AdamW） | 338 ms | 18.2k tok/s |
| 记忆槽开销 | n_mem=0 → 307 ms，n_mem=32 → 338 ms（+10%） | 段长 1024→1088（+6%）+ mask |
| 解码 | **85.8 tok/s**（eager）/ 100.3 tok/s（`reduce-overhead`） | 1×1024 上下文，24 token |
| 解码 kernel 启动 | **10,087 次 / 24 token ≈ 420 次/token** | `hipLaunchKernel` 105 ms，占 CPU 32% |
| 解码 CPU vs GPU | CPU 331.9 ms，GPU 70.8 ms | CPU 是 GPU 的 4.7× → 完全启动瓶颈 |
| 验证 loss | best 4.3465 @85.5k，最后 4.3618 @87.5k | ppl 77–78，val 噪声 ±0.05–0.08 |
| 训练量 | 87,708 步 × 49,152 = **4.31B token** | 102 token/参数，Chinchilla 的 5 倍 |

**SDPA 实测（决定了"手写注意力"是对的选择）**

| 后端 | 时间（bs6×8头×1024） | 结论 |
|---|---|---|
| 手写 (q@kᵀ → softmax → @v) | **4.21 ms** | 当前实现 |
| SDPA math | 17.81 ms | 慢 4.2×（多算了一遍 mask 上三角） |
| SDPA mem_efficient / flash | 直接抛错 | **gfx1200 没编译 flash/mem-efficient kernel** |

> 结论：在 gfx1200 上 `F.scaled_dot_product_attention` 只有 math 后端可用，而且更慢。
> 想要更快的注意力只能自己写（分块/tiling），或者升级 torch 找带 flash 的 ROCm 轮子。

### 1.4 收敛状态

- 训练 loss 7.23（step 500）→ 4.10（step 87,700）；val 4.35–4.52 区间震荡，best 4.3465。
- **val 噪声 ±0.08**：85500 步 4.3465、85000 步 4.5219，单次 eval（40 个 batch）的抖动就有 0.17。
  → 任何"A 比 B 好 0.02"的结论，用现在的 eval 配置都测不出来（§4 讲怎么测）。
- 4.31B token / 42M 参数 = 102 token/参数，已经严重"过训练"，所以：
  **加容量（更宽/更大词表）的边际收益 < 提升样本效率（优化器、数据、MTP）的收益。**

---

## 2. 缺什么（对照论文的 gap 清单）

### 2.1 训练稳定性 / 效率
| 缺的东西 | 论文出处 | 现状风险 |
|---|---|---|
| QK-Norm（q、k 各做一次 RMSNorm） | ViT-22B (2302.05442)、Qwen3、Gemma-3、Chameleon 都默认开 | bf16 下 attention logits 无上界，只能靠 clip 1.0 兜着 |
| logit soft-cap / z-loss | Gemma-2 (2408.00118) softcap 30·tanh；PaLM (2204.02311) z-loss 1e-4 | logits 漂移没有约束，长训后期容易抖 |
| Muon 优化器 | Keller Jordan 2024；Moonlight (2502.16982) | 还在用 AdamW，白扔 1.3–2× token 效率 |
| muP / 分组 LR | Tensor Programs V (2203.03466) | 换宽度就要重调 LR（做 A/B 时很痛） |
| 权重 EMA | 常见做法（DiT/EDM 系） | 白扔 0.5–1% ppl |
| FP32 master weights | — | bf16 autocast + AdamW 的 eps 精度问题（小模型影响小） |

### 2.2 表达能力（结构）
| 缺的东西 | 说明 |
|---|---|
| 深度 | 只有 8 层。MobileLLM (2402.14905) 的核心结论：**同参数量下，深比宽好** |
| value residual | ResFormer (2410.17897)：每层 value 混入第 0 层 value，缓解注意力集中 |
| attn_res | 代码写了（沿深度 softmax 聚合），但训练用的 ckpt 里 `res.q` 不存在 → **没开过** |
| MTP | 多 token 预测 (2404.19737)，数据效率 + 可做投机解码 |
| looped depth | Ouro (2510.25741)：同一组层循环 k 次，参数量不变、有效深度翻倍 |
| 局部/滑窗注意力 | 1024 上下文下没必要 |

### 2.3 推理（最大短板）
| 缺的东西 | 影响 |
|---|---|
| KV cache | 每 token 重算整段 → 85 tok/s，理论应有 500+ |
| 静态 shape / CUDA graph | 420 次 kernel 启动/token，`reduce-overhead` 只从 85.8 → 100.3 tok/s（因为 seq 长度每步变，图捕获不了） |
| 批处理 / 连续 batching | 单请求 GPU 利用率 <2% |
| 投机解码 | 有 MTP 之后才有意义 |
| 量化（int8/int4） | 42M 模型在 16GB 卡上不需要 |
| 采样开销 | 每步 `torch.unique(idx)`（重复惩罚）+ 32k softmax/sort，属于"每次都付"的固定税 |

### 2.4 位置 / 长度
- RoPE base 固定 10000，长度写死 1088；没有 NTK/YaRN 外推。
- 记忆槽是自创机制，训练时只做了 2 段 BPTT（有效 2048），**没有和 compressive memory /
  Infini-attention (2404.07143) 这类做法对比过**，属于"未验证的设计"。
  实测长生成会滑进"目录/ISBN"式文本，但那是数据边界问题（v4 已修），不是记忆槽的锅。

### 2.5 已经做对、不用再抄的
pre-norm + RMSNorm、SwiGLU、RoPE、无 bias、权重绑定、bf16 autocast、梯度累积、
cosine + warmup、grad clip、dropout=0（数据 100 token/参数，不需要正则）、
文档边界 EOS（v4）、按书留出验证集。这些在 2026 年的小模型里就是标准答案。

---

## 3. 候选改进（按 收益÷成本 排序）

> 预期收益一栏：论文报告 + 我的判断，标了 ⚠ 的是"我拿不准，必须实测"。

| # | 改动 | 类型 | 预期收益 | 成本 | 风险 | 建议 |
|---|---|---|---|---|---|---|
| 1 | **KV cache + 静态 shape 解码** | 推理 | **5–15× 生成速度**（85 → 500+ tok/s） | 半天 | 低 | **立刻做** |
| 2 | **QK-Norm** | 稳定/质量 | 训练更稳，可把 LR 抬到 1e-3；ppl 小幅改善 | 20 行 | 极低 | **立刻做** |
| 3 | **Muon** | 优化器 | 同 token 数下 loss 更低（社区 1.3–2× 效率）⚠ | 60 行 + 调 LR | 中 | 先 150M token 小规模验证 |
| 4 | **z-loss + logit softcap** | 稳定 | 长训后期不抖；允许更高 LR | 10 行 | 低 | 顺手做（softcap 有 5–10% 步时开销） |
| 5 | **12 层 × 448（同参数更深）** | 结构 | 小模型上稳定的 ppl 改善 ⚠ | 改 preset | 低 | 做 A/B |
| 6 | **Value residual（ResFormer）** | 结构 | 1–3% ppl（论文在 1B–13B 上稳定） ⚠ | 40 行 | 低中 | 二期 |
| 7 | **MTP（2 头，共享 head）** | 结构/训练 | 数据效率 +，可做投机解码 | 100 行，+20% 步时 | 中 | 二期 |
| 8 | **Looped depth（Ouro 式）** | 结构 | 同参数下有效深度翻倍 ⚠（2× 计算） | 80 行 | 中 | 三期 |
| 9 | **attn_res（已实现）** | 结构 | 深层（≥12 层）收益明显，浅层接近持平 | 0 | 低 | 与 #5 一起开 |
| 10 | **GQA（8 Q → 2 KV）** | 效率 | KV cache 4× 小；解码略快；质量基本不变 | 20 行 | 低 | 与 #1 一起 |
| 11 | **48k 词表（v4 数据）** | 数据/结构 | 同字数 token 数 −5%，有效上下文 1706 字 | 数据已就绪 | 低 | 下次训练直接用 |
| 12 | 因子化 head（d→256→V） | 效率 | 省 ~20% 步时 | 20 行 | 中（质量略降） | 只在训练太慢时考虑 |

**明确不建议现在做**（理由都是"规模/平台不匹配"，不是"不好"）：
- **MLA**（DeepSeek-V2 2405.04434）：省 KV cache，但这里 KV cache 只有 8 层×2×1024×512×2B ≈ 16 MB，省它没意义。
- **NSA / DSA**（2502.11089）：为 64k+ 上下文设计；1024 上下文下注意力核心只占 17% FLOPs。
- **线性注意力混合**（Kimi Linear 2510.26692、Gated DeltaNet 2412.06464）：解码 O(1) 很香，但需要 Triton/FLA kernel，
  **gfx1200 连 flash attention 都没编译**，移植风险远大于收益。
- **MoE / 稀疏化**：42M 参数没有专家可分。
- **128k 词表**：嵌入会变 65 M 参数（超过非嵌入部分），小模型撑不住。
- **muP**：只有做宽度 A/B 时才必要。

### 3.1 KV cache（#1）——为什么能快 5–15×

现在 `generate()` 的每一步都跑 `self(seg, mem=mem)`，`seg` 长度从 1 长到 1024：
每 token 的 FLOPs 是 O(L)，总 FLOPs 是 O(L²)。但实测 GPU 只花 70.8 ms / 24 token = 2.95 ms/token，
而 1024 上下文单 token 前向只有 ~100 MFLOP（≈5 µs）→ **GPU 利用率 0.2%**。
真正花掉的是：每层 ~50 次 kernel 启动 × 8 层 × 4.7（CPU 慢于 GPU 的倍数）。

改造要点：
1. 每层缓存 `k, v`（`(B, H, L, Dh)`），只对最后一个 token 算 q；
2. `idx` 用固定长度 buffer（预分配 `block_size + 2*n_mem`），避免每步重新分配/改 shape；
3. 段填满（1088）时，用写槽的 hidden state 建新段 cache：把 `mem_out` 作为读槽重新 prefill 一次（一次性 ~20 ms）；
4. 用 `torch.compile(mode="reduce-overhead")` 包住 decode step 再图捕获，才能把 420 次启动压到几十次。

预期：单条 1024 上下文 300–800 tok/s（不图捕获 200–300，图捕获 600–1000）。
**这是唯一能让 chat.py 从"能看"变成"好用"的改动。**

### 3.2 QK-Norm（#2）——20 行、几乎白给

```python
# Attention.__init__
self.q_norm = RMSNorm(self.d_head)
self.k_norm = RMSNorm(self.d_head)
# forward，RoPE 之前
q = self.q_norm(q)   # (B,H,T,Dh) 上按最后一维归一化
k = self.k_norm(k)
```
ViT-22B 的结论是"加了之后可以不用 warmup、LR 可以更大"；Qwen3 / Gemma-3 直接默认开启。
代价：每层多 128 个参数（8 头 × 64 × 2），FLOPs 可忽略。
对 bf16 训练尤其值：现在 attention logits 无上界，只能靠 `grad_clip 1.0` 兜。

### 3.3 Muon（#3）——最可能"白捡 loss"

对每个 2D 隐层权重（qkv/proj/up/down）用 Newton-Schulz 正交化的动量更新；
嵌入、head、norm 仍用 AdamW。Moonlight 用 Muon 训 3B/16B MoE，达到同效果只需 ~52% 的 token。

```python
# 5 步 Newton-Schulz（bf16 即可）
def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16(); X /= X.norm() + 1e-7
    transposed = X.size(0) > X.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X
# 更新：p -= lr * max(1, rows/cols)**0.5 * ortho(momentum_nesterov)
```
风险：lr 要按参数形状缩放（Moonlight 用 lr≈0.02 + wd 0.1），第一次跑要扫 2–3 个 lr。
建议：150M token（≈1 小时/arm）跑 AdamW vs Muon，只看 val loss。

### 3.4 z-loss / softcap（#4）

```python
if targets is not None:
    lse = torch.logsumexp(logits.float(), dim=-1)
    loss = F.cross_entropy(...) + 1e-4 * (lse ** 2).mean()
```
softcap（Gemma-2）在 CE 之前 `logits = 30 * torch.tanh(logits / 30)`，
代价是对 (B,T,V) 做一次 tanh（32k×6144 = 2 亿元素），实测会吃掉 5–10% 步时；z-loss 便宜得多。
**建议只上 z-loss**，softcap 留给推理端（防离谱 logit 导致采样崩）。

### 3.5 更深 × 更窄（#5）

| 方案 | 层 | d | 头 | 非嵌入参数 | 相对计算量 | 备注 |
|---|---|---|---|---|---|---|
| 现状 base | 8 | 512 | 8 | 25.7 M | 1.00× | |
| 深窄 | 12 | 448 | 7 (64 维) | 25.9 M* | ~1.05× | 同参数、略多计算，MobileLLM 结论支持 |
| 更深 | 16 | 384 | 6 | 25.4 M* | ~1.10× | 再深收益递减，且 attn_res 才刚开始有用 |
| 加宽 | 6 | 640 | 8 | 30.7 M | 1.25× | 不推荐（宽不如深） |

\* 需实测确认；d=448 时 h = int(448·8/3)=1194→1216，单层 2.86 M。

配 attn_res（#9）：Attention Residuals (2603.15031) 的报告是**层数越深收益越明显**，
8 层时几乎持平（这也解释了为什么当年那个 ckpt 没开它）。所以 #5 和 #9 应该一起做。

### 3.6 MTP（#7）与 looped depth（#8）—— 三期

- MTP：多预测 2 个未来 token。共享 `wte` 做 head 时，每个额外位置都要再乘一次 512×V 的矩阵，
  **head 已经是 32–42% 的 FLOPs，MTP-2 会让步时 +20%**。收益是数据效率（2404.19737 在代码上报告最多 3×，
  小模型上小得多）。所以它更像"以后想上投机解码"才值得。
- looped depth：8 层循环 2 次 = 有效 16 层、参数量不变、计算 ×2（4.3B token 要 10 小时）。
  Ouro 的卖点其实是"推理时按难度决定循环几次"。我们 42M 的规模，先做 #5 更划算。

### 3.7 GQA（#10）

8 个 query 头共享 2 组 KV：KV cache 从 16 MB → 4 MB，解码时 k/v 的读取少 4×。
因为解码是启动瓶颈，实际加速很小；质量上 GQA ≈ MHA（2305.13245），MobileLLM 也给小模型用 GQA。
**跟着 #1 一起改，几乎零成本。**

### 3.8 48k 词表（#11）

data_v4 已经用 identity 归一化的 48k BPE 重打包好（1.666 字/token，比 32k 省 5.09% token）。
代价：嵌入 16.4 M → 24.6 M，head FLOPs 32.8 → 49.2 MFLOP/token（前向 +16%），步时约 +8–10%。
净收益取决于"省 5% token"和"+9% 步时"哪个大 —— 基本打平，但**换来全角标点/换行不被破坏**，
这个对中文小说续写是实打实的质量提升。建议直接换。

### 3.9 attn_res（#9）

已经写好（`AttnRes`：每层一个伪查询向量，对前序所有状态做深度方向 softmax）。
论文（2603.15031）主张在深层网络里比标准残差更好，开销可忽略。
注意：**旧 ckpt 的 `res.q` 不存在，说明这个特性从未真正训练过**，没有本地数据支撑，属于"待验证"。

---

## 4. 怎么验证（别被 val 噪声骗）

**问题**：现在 `--eval_batches 40` 的 val loss 抖动 ±0.08；而 QK-Norm/Muon 这类改动的真实收益可能就是 0.02–0.05。
用 40 个 batch 做 A/B，结论全是噪声。

**做法**：
1. **固定验证子集**：从 `data_v4/val.bin` 取固定的前 200 个 block（或 5 本整书），所有 arm 用同一份、
   同一 seed、同一 offset；`--eval_batches 200`（约 200×6×1024 = 1.2M token，几十秒）。
2. **三档预算**（按 41.4k tok/s 折算）：

| 档 | token 数 | 时长/arm | 用途 |
|---|---|---|---|
| 筛选 | 150 M | ≈1.0 h | 一次筛 4–6 个 arm（QK-norm / Muon / z-loss / 深窄 / 组合） |
| 决赛 | 500 M | ≈3.4 h | 前 2 名，确认收益不是噪声 |
| 出货 | 7.2 B（v4 1 epoch） | **≈48 h** | 定稿配置（注意：不是 README 里的 3.6 h，那是旧 0.54B 数据） |

3. **纪律**：同 token 数比 val loss（不比 step）；LR schedule 形状一致（按 token 数算，不按 step）；
   每个 arm 至少 2 个 seed（小模型 seed 方差同样有 0.02 量级）。
4. 新目录约定：`out/ab/<arm>/`，`--tag` 写进 ckpt，`log.csv` 已带 val 行，直接聚合即可。

---

## 5. 建议的落地顺序

**第一批（1 天内，风险低，先做）**
1. KV cache + 静态 shape 解码（#1）+ GQA（#10）→ chat.py 从 85 → 500+ tok/s
2. QK-Norm（#2）+ z-loss（#4）→ 训练更稳
3. 用 v4 数据（48k，EOS 密度 1/1695）重训一版 baseline，把 val 噪声基线打出来

**第二批（A/B 筛选，各 1 小时）**
4. Muon vs AdamW
5. 12×448 + attn_res vs 8×512
6. （可选）value residual

**第三批（看第二批结果再定）**
7. MTP-2 / looped depth / 投机解码

---

## 6. 参考文献

- Attention Residuals, arXiv [2603.15031](https://browse-export.arxiv.org/abs/2603.15031)（`model.py` 注释里的 Moonshot 2026）
- Manifold-Constrained Hyper-Connections (mHC/finetune), arXiv [2607.18130](https://export.arxiv.org/abs/2607.18130)；原始 Hyper-Connections (ICLR 2025)
- Muon 可扩展性 / Moonlight, arXiv [2502.16982](https://browse-export.arxiv.org/abs/2502.16982)；Keller Jordan 的 Muon 博客
- Value Residual Learning (ResFormer), arXiv [2410.17897](https://browse-export.arxiv.org/abs/2410.17897)
- Scaling Latent Reasoning via Looped Language Models (Ouro), arXiv [2510.25741](https://browse-export.arxiv.org/abs/2510.25741)
- MobileLLM（深 > 宽、GQA、嵌入共享）, arXiv [2402.14905](https://export.arxiv.org/pdf/2402.14905)
- Better & Faster LLMs via Multi-token Prediction, arXiv [2404.19737](https://scholar.google.com.mx/citations?view_op=view_citation&hl=en&user=wN9rBkcAAAAJ&sortby=pubdate)
- Kimi Linear（KDA 混合）, arXiv [2510.26692](https://browse-export.arxiv.org/abs/2510.26692)
- Native Sparse Attention, arXiv [2502.11089](https://export.arxiv.org/abs/2502.11089)
- DeepSeek-V2 MLA, arXiv 2405.04434；GQA, arXiv 2305.13245；Gemma-2 softcap, arXiv 2408.00118；PaLM z-loss, arXiv 2204.02311；ViT-22B QK-norm, arXiv 2302.05442；muP, arXiv 2203.03466；Infini-attention, arXiv 2404.07143

> 注：本环境的 DNS 把 arxiv.org 解析到了非公网 IP，`web_fetch` 无法打开正文，
> 上述数字来自论文摘要/二手引用与我的既有知识；**标 ⚠ 的预期收益必须实测**。

---

## 7. 已落地（2026-10-27，代码已改并验证）

第一批只做了"低风险、立刻有用"的三项（#1 KV cache + #10 GQA、#2 QK-Norm、#4 z-loss），
Muon / 深窄 / value residual 仍未动。

| 项 | 代码位置 | 开关（默认） | 验证 |
|---|---|---|---|
| KV cache 解码 | `model.py` `generate(use_cache=True)`、`_roll()`、`_write_slots()`；旧路径保留为 `_generate_nocache()` | 默认开 | `tools/test_cache.py` 三段全过：旧 ckpt 上 cache 与重算的逐步 logits 偏差 ≤3e-5、采样序列一致；prompt 超过 block_size 的滚动分段也一致 |
| GQA | `Attention.qkv` 输出改成 `(n_head + 2*n_kv_head)*d_head`，`n_rep` 用 `repeat_interleave` 展开 | `--n_kv_head N`（0 = MHA） | 新配置前向/反向正常，cache 与重算一致 |
| QK-Norm | q/k 在 RoPE 之前各过一遍 `RMSNorm(d_head)`（**只在开启时才建参数**，旧 ckpt 仍能 strict 加载） | `--qk_norm 1`（默认 0） | 同上；梯度有限非零 |
| z-loss | `forward()` 里 `loss += z_loss * mean(logsumexp(logits)²)` | `--z_loss 1e-4`（默认 0） | 同上 |
| 采样重构 | 重复惩罚/top_k/top_p/min_p 抽成模块级 `sample_token()`，train/推理共用 | — | 与改造前行为一致 |

`train.py` 冒烟（`--n_kv_head 2 --qk_norm 1 --z_loss 1e-4`，3 步）：参数 47.15M（非嵌入 22.58M，48k 词表），
loss 10.85 → 10.73，无报错。旧 ckpt 用新代码推理无需任何转换（cfg 缺的键一律按"关闭"处理）。

### 解码实测（RX 9060 XT，96 token，top_k=50）

| prompt | fp32 重算 | fp32 cache | 加速 | bf16 重算 | bf16 cache |
|---|---|---|---|---|---|
| 64 | 78.7 tok/s | **123.2** | 1.6× | 112.1 | 110.6 |
| 256 | 45.0 | **121.8** | 2.7× | 109.0 | 112.0 |
| 1024 | 93.9 | **111.0** | 1.2× | 112.4 | 109.1 |

**结论要说清楚**：cache 在 fp32 下 1.6–2.7×，但在 **bf16 autocast 下几乎没有收益**——因为 bf16 的
matmul 已经很快，每 token 的 ~420 次 kernel 启动成了唯一瓶颈（8.3–9.2 ms/token ≈ CPU 侧极限），
而 cache 只减少 GPU 计算、不减启动次数。chat.py 用 bf16，所以体感是 76 → ~110 tok/s（约 1.4×，
一半来自 cache、一半来自没再重算 1024 长段）。要真的到 500 tok/s，得靠 CUDA graph（`torch.compile
(mode="reduce-overhead")` 实测只到 100 tok/s，因为每步 shape 在变）或把逐层算子融合成单 kernel，
属于下一步的活。

GQA 的收益同样要等**重新训练**才体现（旧 ckpt 是 8/8 MHA，不能事后改）：KV cache 从 16 MB 降到 4 MB，
解码时 k/v 读取少 4×。

## 附：体检脚本

- `_bench_arch.py`：参数/FLOPs 分解、解码速度随上下文长度、vocab 32k vs 1k 步时、n_mem 开销
- `_prof_dec.py`：解码的 kernel 级 profile（发现 420 次启动/token）
- `_exp_sdpa.py`：SDPA 三种后端可用性与耗时对比、compile 前后解码速度
