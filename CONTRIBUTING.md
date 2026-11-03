# 贡献指南

欢迎 issue / PR。这个仓库的目标是「**小到能读懂**」：一个 400 行的模型 + 一个 300 行的训练循环，
任何改动都应该保持这个规模。

## 环境

```bash
pip install -r requirements.txt
python tools/check_presets.py            # 只依赖 torch/numpy，不需要数据
```

## 改动前

- 模型结构 / 训练逻辑的改动，请带上**可复现的实验数字**（loss 曲线、tok/s、显存峰值），
  并写进 `docs/experiments.md`。
- 数据格式改动（`train.bin` / `meta.json` 字段）会破坏旧检查点，请在 PR 里明确说明。
- 不要提交 `data*/`、`out*/`、`*.pt`、`*.bin`、`*.log` —— `.gitignore` 已经挡住了，但别用 `-f` 硬塞。

## 代码风格

- Python 3.10+，单文件可读优先，不过度抽象；
- 面向用户的脚本都用 `ROOT = os.path.dirname(os.path.abspath(__file__))` 推导路径，
  **不要出现绝对路径**（`--data_dir` / `--out_dir` 默认值都基于 `ROOT`）；
- 命令行参数用 `argparse`，默认值要能在仓库根目录直接跑通；
- 注释写「为什么」，不写「是什么」。

## 提交信息

```
train: 修 resume 时 step 计数多算 1
sft:   mask 边界把 EOS 漏掉的情况
docs: 补 data.md 的清洗阈值
```

## 自检

```bash
python -m py_compile *.py tools/*.py
python tools/check_presets.py --preset thin
python chat.py --selftest        # 只测命令解析，不加载模型
```
