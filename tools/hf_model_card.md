---
license: mit
language:
  - zh
tags:
  - chinese
  - text-generation
  - novel
  - mininlm
  - from-scratch
library_name: pytorch
pipeline_tag: text-generation
---

# miniNLM thin 46M

一个**从零训练**的中文小模型（46.34M 参数，非嵌入 27.90M），语料是中文网络小说，训练了 7.2B token。
适合用来学习语言模型结构、跑通训练/推理流程、或者当一个小体积的中文续写玩具模型。

配套代码：**[github.com/cmddirexit/miniNLM](https://github.com/cmddirexit/miniNLM)**

权重文件 `model.safetensors`（@SIZE@，fp32），结构代码随仓库附带，不依赖其它仓库即可加载。

## 指标

| 项目 | 数值 |
|---|---|
| 参数 | @PARAMS@ 总 / @PARAMS_NE@ 非嵌入 |
| 训练 token | 7.20B |
| 上下文 | @CTX@（＋@NTS@×2 记忆槽 → 有效 2048） |
| 验证集 loss | @VAL@（ppl @PPL@） |
| 训练步数 | @STEP@ |
| 词表 | SentencePiece BPE，48000 |

验证集是按同样协议（`n_seg=2`、`bs=4`、block 1024、固定随机种子）在留出的小说分片上算的
next-token loss。测试时 600 批跑出的 loss 是 4.0576（ppl 57.83，95% CI ±0.014）。

## 用法

```bash
pip install torch safetensors sentencepiece

# 方式一：git
git lfs install
git clone https://huggingface.co/@REPO@
cd miniNLM-thin-46m
python example.py --prompt "他推开客栈的门，"

# 方式二：只下载文件
python -c "
from huggingface_hub import snapshot_download
snapshot_download('@REPO@', local_dir='miniNLM-thin-46m')"
cd miniNLM-thin-46m && python example.py --prompt "他推开客栈的门，"
```

输出示例（prompt = `他把剑收回鞘中，转身看向身后的少年。`，temperature 0.75 / top_k 40 / rep_penalty 1.15）：

```
他把剑收回鞘中，转身看向身后的少年。
“你是何人，为何对我天君出手？”白凌雪指着杨叶问道。
杨叶道：“天君，我是杨某人。”
白凌雪闻言脸色顿时一沉，然后看向对面不远处的那个少年：“是你？”
杨叶道：“不错，是我，白仙子，想不到你居然认得我。”
```

排版、引号、对话分行、人称都像样；但名字会串、人物关系会乱、越往后越飘。

18 行代码加载模型：

```python
import json, torch, sentencepiece as spm
from safetensors.torch import load_model
from model import GPT

conf = json.load(open("config.json"))
model = GPT(conf["arch"]); load_model(model, "model.safetensors"); model.eval()
sp = spm.SentencePieceProcessor(model_file="spm48k_v4.model")
ids = sp.encode("他把剑收回鞘中，转身看向身后的少年。")
x = torch.tensor([ids])
out = model.generate(x, 100, temperature=0.75, top_k=40, eos_id=4,
                     repetition_penalty=1.15)
print(sp.decode(out[0][len(ids):].tolist()))   # 只解码新生成的部分
```

## 架构

- 18 层 Transformer，d=384，8 头，GQA（2 个 KV 头），RoPE 位置编码 + QK-norm
- **注意力残差**（AttnRes）：每层对前面所有层的输出做一次学习到的加权，代替单纯的残差累加
- **记忆槽**：每段前面 32 个「读槽」token、后面 32 个「写槽」token；写槽的输出接给下一段当读槽，把上一段的语义摘要带过来（有效上下文 1024 → 2048）
- SwiGLU FFN、RMSNorm、权重绑定（嵌入 = 输出层）
- 训练用 bf16 混合精度 + AdamW + z-loss 1e-4

单卡 AMD RX 9060 XT（16 GB）上约 2.9 s/step（batch 4 × grad-accum 6 × 1024 token）。

## 怎么用比较好

- `temperature 0.7–0.8`、`top_k 40`、`repetition_penalty 1.15`：稳、不绕圈
- 温度 0.9 以上开始重复，贪心解码容易掉进 `我是……我是……` 的循环
- 适合：给一句话让它续写网文片段；不适合：问答、代码、事实性知识

## 局限

- 46M 参数 + 单一小说语料：知识面很窄，2–3 句之后容易跑题、人名会串
- 没有做过指令微调（SFT 只到「学会对话格式」的程度），不要期望它听懂复杂指令
- 中文标点和分段学得不错，但长文本逻辑一致性弱

## 训练与许可

- 代码（训练 / 数据管线 / 评测 / SFT）：https://github.com/cmddirexit/miniNLM
- 本仓库里也带了一份 `model.py`，权重自带架构，不依赖上面那个仓库也能加载
- 语料为公开网络小说，仅用于研究；**模型输出可能复现训练语料片段**，商用前请自行评估版权风险
- 权重以 MIT 许可发布
