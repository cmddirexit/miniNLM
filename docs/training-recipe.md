# 训练配方调研：怎么把 v6 这一段 147 小时花得更值

> 目的：不改数据、不改架构，只改「怎么训」——学习率调度、稳定性、权重平均、批次与数据顺序。
> 结论先行：**现在只跑了 0.3%（421/146473 步），是换配方最便宜的时刻**；如果等跑完 147 小时再发现问题，代价是整段算力。

---

## 1. 现状（v6，2026-10-27 15:34 快照）

| 项 | 值 |
|---|---|
| 模型 | `thin` 18 层 × d384 × 8 头，head_dim 48，GQA 2/8，qk_norm 开，z_loss 1e-4，attn_res 开，n_mem 32 |
| 参数 | 46.34M（非嵌入 27.90M） |
| 数据 | `data_v4`：7,199,477,760 token / 7,030,740 块 / 6580 本；val 406,569,984 token / 371 本 |
| 批次 | bs 4 × accum 6 × 1024 × 2 段 = **49,152 token/步** |
| 计划 | 146,473 步 = 7.2B token = **1 epoch**；token/参数 = **155** |
| 调度 | **预注册的 cosine**，6e-4 → 6e-5，warmup 500 |
| 实测 | 2.88–3.6 s/步（13.6k–17.1k tok/s），**ETA 约 147 小时**（11/2 凌晨） |
| 已验证 | `tools/test_cache.py` 三段 PASS；`tools/check_presets.py` 参数与 FLOPs |
| 进度 | step 421 / 146,473（**0.29%**），loss 10.83 → 7.19（warmup 内） |

---

## 2. 文献证据（都读过全文/摘要，附链接）

### 2.1 学习率调度：cosine 是「必须提前赌总步数」的方案

| 论文 | 关键结论 | 对我们的含义 |
|---|---|---|
| [Hägele et al. 2405.18392](https://arxiv.org/abs/2405.18392) *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* | 常数 LR + 尾部短 cooldown（梯形/WSD）**与调好的 cosine 打平**；cooldown 占 **10–20%** 就够，长训练时 **5%（1 万步）** 也够；cooldown 形状 **(1-√x) 稳定优于 linear**；最终 LR 衰减到接近 **0** 比留 10% 好；SWA 沿轨迹免费提升 | 我们现在的 `min_lr=6e-5`（峰值 10%）是次优；`--min_lr 0` + (1-√x) 是明确改进。更重要的是**不用先把 146,473 步赌死** |
| [Wen et al. 2410.05192](https://arxiv.org/abs/2410.05192) *Understanding WSD: A River Valley Loss Landscape Perspective* | 大 LR 在「河谷」里沿河前进快、但在谷宽方向震荡；**stable 相位的 loss 会明显高于 cosine**，decay 一压就掉到更低；提出 WSD-S（一次运行产出多个可比较 ckpt，0.1B–1.2B 验证优于 WSD 与 cyclic-cosine） | 看板要提前说明「中间悬着不是坏了」；一次 run 可以留多个「分支点」 |
| [2508.01483](https://arxiv.org/abs/2508.01483) *Training Dynamics of the Cooldown Stage in WSD* | cooldown 阶段是 bias–variance 权衡：不同形状对应不同的偏差/方差取舍 | 形状别用 linear，用 (1-√x) 或余弦尾 |
| [MiniCPM 2404.06395](https://arxiv.org/abs/2404.06395) | WSD 的 decay 阶段用 **10% 总 token 就能到最优**，**2.5% 明显不够**；decay 阶段**混入高质量数据**比只用预训练数据好；用 Tensor Program（muP）让最优 LR 跨 0.04B→0.5B 稳定；用 0.009B 的「风洞」试配方 | 我们 147h 里应留 **~15k 步（10%）** 做 decay，不是 5% |

### 2.2 稳定性：我们已开的正是 OLMo 2 那一套，但缺两块

| 论文 | 关键结论 | 对我们的含义 |
|---|---|---|
| [OLMo 2 2501.00656](https://arxiv.org/abs/2501.00656) | 用 **QK-Norm + z-loss + 重排 norm + embedding 不做 weight decay** 消除 loss/grad spike；**重复 n-gram 的文档是 spike 的重要来源**；提议 **micro-annealing** 独立评估各数据源；7B 上**对同一数据 anneal 三次（各 50B，不同顺序）再把三个模型平均** | 我们已有 qk_norm/z_loss；**缺「embedding 不进 weight decay」**（`train.py:189-196` 目前 `wte.weight` 在 decay 组），以及**spike 自愈**；末期 anneal + 多模型平均可直接照搬 |
| [2309.14322](https://arxiv.org/abs/2309.14322) *Small-scale proxies for large-scale Transformer training instabilities* | 高 LR 的**小模型也能复现** logit 增长与 logit/log-prob 发散；z-loss / QK-norm / 降 LR 都能压 | 说明我们在 46M 上看到的现象可信，不必担心「规模不到所以测不出」 |
| [2410.05052](https://arxiv.org/abs/2410.05052) | 参数**范数不均匀**是 spike 的一个成因；重参数化初始化可缓解 | 我们的 init 是统一 std 0.02，深度 18 层时可按层缩放（低优先） |

### 2.3 权重平均：最便宜的免费提升

| 论文 | 关键结论 | 对我们的含义 |
|---|---|---|
| [Sanyal et al. 2306.03241](https://arxiv.org/abs/2306.03241) *Early Weight Averaging meets High LR* | LAWA（滑动窗口检查点平均）在 **高 LR** 训练下收益最大，且**越早开始越大**；GPT2-small 2.963→**2.917**、medium 2.855→**2.819**、large 2.977→**2.908**（比 EMA/SWA 都好）；但**早期 SWA 会发散**（必须在最后阶段做） | 我们的 val 抖动 ±0.08，比这些收益还大 → **averaging 既是提分也是降噪**。做法：从 60% 进度起维护影子权重 |
| [Hägele 2405.18392](https://arxiv.org/abs/2405.18392) | SWA 可**替代 cooldown**（若可接受一点差距） | 与 WSD 组合是「免费」的第二保险 |

### 2.4 数据重复与批次

| 论文 | 关键结论 | 对我们的含义 |
|---|---|---|
| [Muennighoff et al. 2305.16264](https://arxiv.org/abs/2305.16264) *Scaling Data-Constrained LMs* | 数据受限时，**重复 ≤4 个 epoch 与全新数据几乎无差**，再往上收益衰减到 0 | 我们 1 epoch = 155 token/参数（Chinchilla 最优 ~20 的 7.7 倍），**跑第 2/3 个 epoch 完全合理**，比换模型划算 |
| [McCandlish et al. 1812.06162](https://arxiv.org/abs/1812.06162) *An Empirical Model of Large-Batch Training* | 梯度噪声尺度 `B_noise` 预测「加大 batch 还有没有用」：`B ≪ B_noise` 时线性加速 | 我们现在 49k token/步。测一次 `B_noise`（两次同分布 batch 的梯度即可估）就知道 accum 6 是否浪费 |
| [2107.02027](https://arxiv.org/abs/2107.02027) *Efficient Sequence Packing without Cross-contamination* | 变长序列直接 padding 可浪费 50% token；packing 要保证不串样本 | 我们已经跨书连续打包 + EOS 边界，没有 padding 浪费；**但 attention 不按书 reset**（见 §4.6） |

---

## 3. 一次运行能改的旋钮（按 ROI 排序）

| # | 改动 | 改法 | 证据 | 预期 | 成本 / 风险 |
|---|---|---|---|---|---|
| 1 | **cosine → WSD（stable + (1-√x) decay）** | `--lr_schedule wsd --decay_frac 0.10 --cooldown 1-sqrt --min_lr 0`；`--max_steps` 变成「stable 时长」，decay 前随时可改主意 | 2405.18392 / 2410.05192 / MiniCPM | 与 cosine 同分或略好；**换来「不必现在赌 147h」的自由度** + 可续跑 2/3 epoch | 中：`lr_at()` 重写 ~20 行；看板要能接受中间 loss 偏高 |
| 2 | **embedding 不做 weight decay** | `train.py:189` 的 `decay` 列表排除 `wte.weight`（head 与 wte 绑定，需按名字处理） | OLMo 2 §3.4.2（wd on embeddings → norm 下降、grad norm 上升、spike 分数 0.16 vs 0.092） | 稳定性小改进，几乎零风险 | 极小：3 行 |
| 3 | **SWA/LAWA 影子权重** | 从 ~60% 进度起，每次 `save_interval` 把 fp32 影子权重按计数平均；eval 同时报 `val(swa)`；最终 `ckpt_last` 用平均 | 2306.03241 / 2405.18392 | val 降 0.02–0.07 + **降噪**；选模型更稳 | 小：~15 行；显存 +190MB（16GB 够） |
| 4 | **spike 自愈 + grad_norm 观测** | 记录 `grad_norm`（`clip_grad_norm_` 的返回值，现在被丢掉）；`loss > 均值+4σ` 或 `grad_norm > 10×中位数` 则**跳过本步不更新**并计数；连续 3 次从 `ckpt.pt` 重载并跳过坏 batch | OLMo 2 / PaLM 实践 | 无人值守 6 天的保险；也让 log.csv 可诊断 | 小：~25 行 |
| 5 | **末期数据退火（anneal 子集）** | 先给 6951 本书打「干净度分」（用 `sanitize_v4` 的 tags/ents/removed + 短行/重复行 + CJK 比），取 top 10% 单独 pack 成 `anneal.bin`（~0.7B token），decay 阶段换成它 | Llama 3 / OLMo 2 / MiniCPM | 文献里这是最稳的一档收益；但**我们的质量信号是代理指标**，需先小规模 A/B | 中：打分脚本 + 再 pack 一次（~10 分钟）+ 训练侧切数据源 |
| 6 | **attention 按书 reset** | 打包时记录每块内「书边界」位置，训练时用 block-diagonal mask（或把 book id 存成 sidecar） | 2306.03241 的对立面：跨书注意力会串味；packing 论文也强调不串样本 | 可能小改善（长篇内仍连续，只有跨书被切断） | 中：要改 packer 产出 sidecar + mask 构造；`n_seg=2` 时跨段更复杂 |
| 7 | **n_mem 32 → 16** | 架构开关，需重训 | 我们实测 n_mem 只占 10% 时间（338→307 ms） | 省约 12 小时 | 低但收益只是速度；记忆槽是有用的（长程一致） |
| 8 | **测 B_noise 再定 accum** | 两步法：同参数下用 accum=1 与 accum=6 的梯度估 `tr(Σ)/‖G‖²` | 1812.06162 | 若 `B_noise ≫ 49k` → 加大 accum 可缩短墙钟 | 小：一个探针脚本；要在训练间隙跑 |
| 9 | **μP / 风洞** | 以后换 preset 时用 0.009B 代理扫 LR，再按 muP 缩放 | MiniCPM / muP 2203.03466 | 避免每换一次架构就重扫 LR | 中；**这次不适用**（preset 已定） |
| 10 | **多 anneal 平均** | 若时间允许：同一 stable ckpt 用 3 种数据顺序各 anneal 50B→ 三个模型平均 | OLMo 2 §4 | 论文里是明确的额外收益 | 大（3× anneal 成本） |

**不要做**（证据不足或成本不匹配）：MoE / 稀疏注意力 / 线性注意力（单卡 16GB，收益靠长上下文）；蒸馏（没有教师）；课程学习（chunk 随机采样已经够）；换 tokenizer（刚重训完）。

---

## 4. 建议：v7 配方（如果现在重启，只损失 26 分钟）

### 4.1 命令行

```bash
setsid nohup python -u train.py \
  --data_dir data_v4 --out_dir out_v7 --preset thin \
  --n_kv_head 2 --qk_norm 1 --z_loss 1e-4 --warmup 500 \
  --batch_size 4 --grad_accum 6 \
  --lr_schedule wsd --lr 6e-4 --min_lr 0 --decay_frac 0.10 --cooldown 1-sqrt \
  --no_wd_embed 1 --swa_start 0.60 --spike_guard 1 \
  --eval_interval 1000 --eval_batches 200 --save_interval 1000 --tag v7 \
  > train_v7.log 2>&1 &
```

- `--max_steps` 不给（= 1 epoch 的 stable 相位 131,825 步），decay 的 14,648 步在最后自动接上；**中途想延长/缩短只改 decay 起点**。
- 期望曲线：stable 相位 loss 停在 ~3.9–4.1 并**高于** v6 的 cosine 同点；进 decay 后 3–5k 步内掉到 3.6 以下——这是 WSD 的正常形态（2410.05192 的 river valley）。

### 4.2 代码改动清单（约 90 行）

1. `lr_at()` 换成 `lr_at(step, args, max_steps)` 的分支实现：`cosine` / `wsd`；WSD 的 decay 段用 `1 - sqrt(x)`。
2. `decay` 组排除 `wte.weight`（并注意 `head.weight` 是同一张量，`named_parameters` 去重后只出现一次）。
3. `swa_shadow`：`dict[name] = clone(p.detach().float())`；每次 `save_interval` 做 `p_new = (n*p_old + p)/ (n+1)`；`--swa_start 0.6` 前不动；eval 时临时 `load_state_dict(swa)` → 评 → 还原。
4. `grad_norm = clip_grad_norm_(...)`；维护 `gn_hist`（最近 200 个）；spike 判定与 `skip`/`reload` 逻辑；log.csv 加 `grad_norm` 列。
5. `--swa_start`、`--spike_guard`、`--lr_schedule`、`--decay_frac`、`--cooldown`、`--no_wd_embed` 六个 argparse 项。

### 4.3 判断“值不值得重启”的三个数字

- 已投入：**26 分钟 / 147 小时 = 0.3%**。
- 可拿回：调度自由度（2–3 个 epoch 的续跑能力）+ 免费 averaging + 稳定性保险 + 更低的最终 LR。
- 代价：编译重来一次（6 分钟）+ 26 分钟训练 + 我改代码约 30 分钟（可在训练跑着的同时改，改完再切）。

---

## 5. 如果选择不重启

那至少做这两件事（都不需要停训练）：

1. **准备 `tools/probe_noise_scale.py`**：训练一结束就跑，确定 49k token/步是否已在「浪费区」，为下一次定 accum。
2. **把 anneal 子集的打分脚本写出来**（`tools/score_books.py`），下一轮直接用；顺便复算「高质量 0.7B token 子集」够不够 10% decay。

---

## 6. 参考文献

1. Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*, arXiv:2405.18392 — https://arxiv.org/abs/2405.18392
2. Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective*, arXiv:2410.05192 — https://arxiv.org/abs/2410.05192
3. *Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay LR Scheduler*, arXiv:2508.01483 — https://arxiv.org/abs/2508.01483
4. Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, arXiv:2404.06395 — https://arxiv.org/abs/2404.06395
5. OLMo Team, *2 OLMo 2 Furious*, arXiv:2501.00656 — https://arxiv.org/abs/2501.00656
6. Grattafiori et al., *The Llama 3 Herd of Models*, arXiv:2407.21783 — https://arxiv.org/abs/2407.21783
7. Sanyal et al., *Early Weight Averaging meets High Learning Rates for LLM Pre-training*, arXiv:2306.03241 — https://arxiv.org/abs/2306.03241
8. Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv:2305.16264 — https://arxiv.org/abs/2305.16264
9. McCandlish et al., *An Empirical Model of Large-Batch Training*, arXiv:1812.06162 — https://arxiv.org/abs/1812.06162
10. Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, arXiv:2309.14322 — https://arxiv.org/abs/2309.14322
11. Karpukhin et al., *Efficient Sequence Packing without Cross-contamination*, arXiv:2107.02027 — https://arxiv.org/abs/2107.02027
12. Lee et al., *How to Train Data-Efficient LLMs*, arXiv:2402.09668 — https://arxiv.org/abs/2402.09668
