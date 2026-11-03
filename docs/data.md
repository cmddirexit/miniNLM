# 数据管线（从 HF 原始分片到 `train.bin`）

全流程在本机（WSL2 + 32 线程 CPU）约 **1.5 小时**；下载 39 GB 另算。

## 0. 结果概览

| 项 | 值 |
|---|---|
| 来源 | HF `wdndev/webnovel-chinese`，10 个 jsonl 分片，39 GB，**一行一章** |
| 清洗后 | **6951 本 / 127.94 亿字 / 4,483,413 章**（丢短书 775、低质章 100,589、书内重复章 1,291） |
| 分词 | `spm48k_v4.model`，SentencePiece BPE 48000，**identity 归一化**（全角标点、换行都保留） |
| 压缩率 | **1.666 字 / token**（32k 词表 1.585，48k 省 5.09% token） |
| 训练集 | `train.bin` 14.4 GB = **7,199,477,760 token** / 7,030,740 块 · 6580 本 |
| 验证集 | `val.bin` 813 MB = **4.07 亿 token** / 397,041 块 · 371 本（按书 id 哈希 5% 整本留出） |
| EOS 密度 | 1/1695 token，**57.5% 的 1024 块里至少一个 EOS** |
| byte-fallback | 0.001%（48k 词表 + identity 后几乎不触发） |

## 1. 下载

```bash
python tools/dl_hf.py --out data/hf_novels_v4      # 39 GB，断点续传
```

## 2. 抽取与清洗

```bash
# jsonl → 每本书一个 txt（丢短书、丢低质章、去掉书内重复章）
python tools/prep_v4.py --shards 0 1 2 3 4 5 6 7 8 9 --workers 6 --out data_v4/books

# 行内 HTML / 实体二次净化（约 315 s）
python tools/sanitize_v4.py --books data_v4/books --workers 8
```

`prep_v4.py` 的关键约定：

- 一行一章；章与章之间、书末各插一个 `<|endoftext|>`（`eos_id=4`），保证 EOS 不会被当垃圾删掉；
- 书名/作者从 jsonl 元数据里带出来，写进文件名与 `books/` 索引；
- 「低质章」= 少于 N 字、纯目录、纯广告、重复率过高，判定阈值写在脚本顶部。

## 3. 训练分词器

```bash
python tools/train_spm_v4.py --books data_v4/books --out data_v4 --sample-mb 300 --vocab 32000 48000
```

- 同时产出 32k / 48k 两个词表，方便对比压缩率；
- `identity` 归一化：不做全角→半角、不折叠换行（旧 32k 词表用默认 `nmt_nfkc`，会把标点和换行吃掉，已废弃）；
- 词表内只有 3 个控制符，`<|endoftext|>` = **id 4**，其余全部是内容 token。

## 4. 打包

```bash
python tools/pack_v4.py --spm data_v4/spm48k_v4.model --out data_v4
python tools/check_v4.py --data data_v4          # 验收：抽样解码 + EOS 密度 + byte-fallback
```

`pack_v4.py` 产出：

- `train.bin` / `val.bin`：`uint16` 平铺 token 流（48k 词表刚好放得下 `uint16`，比 `uint32` 省一半 IO）；
- `meta.json`：`vocab_size`、`block_size`、`train_tokens`、`eos_id`、分词器文件名；
- 验证集按书 id 哈希整本留出，保证同一本书不会同时出现在训练/验证里。

## 5. 语料体检（可选）

```bash
python tools/report_corpus.py --dir data_v4/books --out data_v4/_report.jsonl --limit 400
```

## 6. 训练时的约定

- `train.py` 只读 `meta.json` 里的字段，**不信任文件名**：`--data_dir` 给错目录会直接报错。
- 块边界不跨书：一个训练样本的 1024 token 永远来自同一本（打包时按书对齐，不足则丢弃尾部）。
- 数据流是不可变的：想换词表/换清洗规则必须重新打包，旧检查点会因为词表不匹配而失效（`cfg["vocab_size"]` 会挡住）。

## 7. 旧语料 `data/`

仓库里可能还留着更早的 `data/`（529 本 / 9.76 亿字 / 32k 词表 / 5.4 亿 token），
`tools/legacy/build_data.py` 是那套流程。**两套词表互不兼容**：旧 `out/*.pt` 只能用旧 `data/` 接着训。
