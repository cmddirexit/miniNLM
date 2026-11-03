# miniNLM SFT 指南（从 0 到跑通，以及三轮实验的教训）

> 基础模型：`out/ckpt_best.pt`（thin 18 层 d=384，64.77M 参数，7.2B token 中文小说预训练，val_loss 4.0726 / ppl 58.6）
> SFT 目标：把「只会续写」的模型，调成「按指令回答」。
> 结论先行：**管线已经通了，模型也没被弄坏；但 25 条指令 + 99% 自动续写数据学不出可用对话能力 —— 瓶颈在数据，不在代码。**

---

## 1. SFT 是什么（和预训练的区别）

| | 预训练（train.py） | SFT（sft.py） |
|---|---|---|
| 数据 | 7.2B token 原始小说，拼成连续流 | 几万条「指令 → 回答」样本 |
| loss 范围 | 每个 token 都算 | **只算「助手：」后面的回答**，问题部分不算 |
| 学习率 | 6e-4 起步 cosine 到 6e-5 | 1e-4 ~ 3e-5，几百到几千步 |
| 目标 | 学会语言的分布 | 学会「听到问题要回答」这个格式和行为 |

一句话：预训练教它「会说话」，SFT 教它「听人话」。

---

## 2. 四步流程

```
demo.jsonl ──sft_data.py──> data_sft/{train,val}.bin + {train,val}_mask.bin + meta.json
                                      │
out/ckpt_best.pt ──sft.py──> out_sft/ckpt_sft.pt  (+ before.md / after.md 采样对比)
                                      │
                                  ask.py  ← 按「用户/助手」格式对话
```

### (1) 数据格式：JSONL，一行一条

```json
{"messages": [{"role": "user", "content": "你好，你是谁？"},
              {"role": "assistant", "content": "我是 miniNLM，一个中文小模型。"}]}
```

`demo_sft.jsonl` 里手写了 25 条（寒暄、概括、成语、改病句、起名、写对话、续写、拒答、列表……），格式就是上面这样。

### (2) 模板 + mask（最关键的一步）

`sft_data.py` 把一条样本渲染成：

```
\n用户：你好，你是谁？\n助手：我是 miniNLM，一个中文小模型。<|endoftext|>
```

- `<|endoftext|>`（id 4）当 EOS，让模型学会「答完就停」；
- **只有「助手：」之后 + EOS 这些 token 参与 loss**，问题部分 mask 掉。

实测一条样本的 token 与 mask（`tools/show_mask.py`）：

```
原文: '\n用户：你好，你是谁？\n助手：我是 miniNLM，一个中文小模型。<|endoftext|>'
token 数: 25   算 loss 的: 15

'<0x0A>'  15  -        '我是'   573  YES
'用户'   23392  -       '▁'   40561  YES
'：'     39441  -       'm'   41979  YES
'你好'    2968  -       'in'   7098  YES
'，'     39401  -       'i'   41132  YES
'你是谁'   7111  -       'N'   41437  YES
'？'     39433  -       'L'   41758  YES
'<0x0A>'   15  -       'M'   41446  YES
'助手'   11857  -       '，'   39401  YES
'：'     39441  -       '一个'   264  YES
                        '中文' 16881  YES ... '<|endoftext|>' 4  YES
```

**为什么 mask**：如果问题也算 loss，模型会花容量去学「复述问题」，而且 90% 以上 token 是问题会淹没回答。mask 之后 loss 直接反映「答得好不好」。

实现上不需要改模型：`model.py` 的 `F.cross_entropy(..., ignore_index=-1)` 天然支持 —— **把 y 里不该算的 token 设成 -1** 即可：

```python
y = torch.where(mask, y, torch.full_like(y, -1))
```

### (3) 训练

```bash
python sft_data.py --in demo_sft.jsonl --books data_v4/books --books_n 3000 \
                   --out_dir data_sft --val_frac 0.1
python sft.py --sft_dir data_sft --init out/ckpt_best.pt --out_dir out_sft \
              --steps 120 --lr 3e-5 --batch_size 4 --grad_accum 5 --block 1024
```

- 每 15 步会打印进度、每 30 步在验证集上算 loss，并采样写 `before.md` / `after.md` / `probe_N.md`；
- `--init` 只加载 `model` 权重（不加载优化器），所以是从预训练模型干净地开始；
- GPU（RX 9060 XT 15.9G）上约 **3 秒/步**（20480 token/step）。

### (4) 对话

```bash
python ask.py --ckpt out_sft/ckpt_sft_best.pt --q "你好，你是谁？"
python ask.py --ckpt out_sft/ckpt_sft_best.pt          # 交互模式
```

---

## 3. 三轮实验（真跑出来的数字）

| 实验 | 步数 / lr / dropout | train loss | 验证 loss（只算回答） | 小说验证集 loss | 结果 |
|---|---|---|---|---|---|
| base | — | — | 4.1702 | 4.0718 | 只会续写 |
| A `out_sft` | 226 (2 epoch) / 1e-4 / 0 | 4.33 → 3.56 | 4.1702 → **4.3011 ↑** | **4.2439 ↑** | 过拟合，模型被带偏 |
| B `out_sft_b` | 60 / 3e-5 / **0.1** | **8.0 起步** | 4.31 ↑ | — | 直接崩，loss 差一倍 |
| C `out_sft_c` | 120 (1 epoch) / 3e-5 / 0 | 4.33 → 4.10 | 4.1702 → **4.1527 ↓** | 4.0798 | 模型没坏，但没学会答 |

### 三条教训

1. **别在预训练之后加 dropout**。B 用了 0.1，同一份数据、同一个权重，第一步 loss 就从 4.33 变成 8.0 —— 一个按 dropout=0 训出来的模型，突然在每个残差/注意力里丢掉 10% 的激活，等于把它推离训练分布，很难再回来。要正则化就用更小 lr / 更少步数 / 权重衰减。
2. **lr 1e-4 对 46M 小模型偏大**。A 跑 2 epoch 后，连「写小说」的能力都掉了（小说 val 4.07 → 4.24），这叫**灾难性遗忘**。3e-5 跑 1 epoch 时几乎无损（4.0798）。
3. **验证 loss 不代表对话能力**。C 的验证 loss 降了，但采样出来还是「我是……我是……」。因为验证集 99% 是自动续写样本，它衡量的还是「续写」而不是「答题」。

### 为什么采样出来是乱码 / 绕圈

诊断脚本 `tools/check_damage.py` 的结论：**不是权重坏了**。

- C 在原始小说验证集上 loss 4.0798 ≈ base 4.0718，说明语言能力没丢；
- 乱码是因为：① 贪心/低温解码下小模型容易掉进重复循环（base 自己也会「我是……我是……」）；② 训练数据里「助手：」后面 99% 接的是小说正文（带引号对话），模型学到的是「助手：之后继续写小说」，所以会出现 `用户：` 行、`JYA CIN-METI` 这种抽风 token；
- 25 条指令 vs 3000 条续写，比例 1:120，指令信号被淹没了。

---

## 4. 想要真的「会聊天」，需要什么

1. **数据量**：至少 5k–50k 条**真实、多样**的指令数据。25 条手写只够演示格式。
   - 最省事：用大模型（DeepSeek / Qwen）把你手上的小说语料蒸馏成指令对（摘要、改写、起名、问答、人物关系……），几万条几小时就能生成；
   - 或者直接用开源中文 SFT 集（Belle、COIG、Firefly 等）抽样。
2. **配比**：指令数据 ≥ 70%，续写样本 ≤ 30%（保留一点续写能力，别让它忘了怎么写小说）。
3. **超参**：lr 3e-5 ~ 5e-5，1–2 epoch，`--steps` 按 `总token / 20480` 估算；验证 loss 只统计回答部分，同时**每 30 步看采样**，采样崩了就立刻停。
4. **模型容量**：46M 参数装不下太多「知识」，但它可以学会「格式 + 简单任务」。要真答得好，18 层 d=512（≈74M）或更大更划算。

## 5. 文件清单

| 文件 | 作用 |
|---|---|
| `sft_data.py` | JSONL → bin + mask，支持从 `data_v4/books` 自动造续写样本 |
| `sft.py` | 加载预训练权重，mask 后训练，带验证与采样探针 |
| `demo_sft.jsonl` | 25 条手写指令样本，可直接扩写 |
| `ask.py` | 单轮/交互对话 |
| `tools/show_mask.py` | 打印一条样本的 token 与 mask |
| `tools/check_damage.py` | 对比 base / SFT 在小说验证集上的 loss 与多种采样 |
| `out_sft/` `out_sft_b/` `out_sft_c/` | 三轮实验的 ckpt、日志、before/after 采样 |
