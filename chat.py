# -*- coding: utf-8 -*-
"""miniNLM 交互续写 v2（防手滑 + 连续写作）

用法:
    python chat.py                            # 自动选 out/ 里最新的检查点
    python chat.py --ckpt out_v6/ckpt_best.pt    # 指定检查点
    python chat.py --once '第1章 边陲' --n 300  # 一次性续写，不进交互
    python chat.py --once - < 开头.txt          # 从 stdin 读开头
    python chat.py --selftest                 # 只测命令解析，不加载模型

交互:
    输入文字   = 拿它当开头（覆盖当前上下文）
    直接回车   = 接着刚才写的内容继续
    : 开头     = 命令（全角 ：也行），:help 看全部

和 v1 的区别（都是"写小说"实际会用到的东西）:
    · 有上下文了：写完的内容会留着，回车就能接着写，不用每次粘一遍
    · top_p / min_p / 固定种子 / 停止词：控制"写多稳、写多长、写到哪停"
    · 多行粘贴（:paste）、读文件（:file）、存档（:save）
    · 撤销（:undo）、重写（:retry）、看上一轮（:last）
    · 没有 CUDA 自动退到 CPU，不再死在 autocast 上
"""
import os
import sys
import math
import time
import argparse
import difflib
import threading
import unicodedata
import contextlib

from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT
ROOT = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = ("ckpt_best.pt", "ckpt.pt", "ckpt_last.pt")

AUTO_CAP = 2000          # :n 0（一直写到结束符）时的硬上限，防止停不下来
MAX_TEXT_CHARS = 20000   # 只留上下文末尾这么多字，控制每轮重新分词的开销
HISTORY_PATH = "~/.miniNLM_history"

COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")
_RL = [False]            # readline 是否可用：提示符里的颜色要用 \001..\002 包起来


# ---------------------------------------------------------------- 颜色 / 提示符
def C(code, s):
    """普通颜色（输出用）"""
    return "\033[%sm%s\033[0m" % (code, s) if COLOR else s


def _rl(s):
    return "\001" + s + "\002" if _RL[0] else s


def CP(code, s):
    """提示符里的颜色：不可见字符必须包起来，否则 readline 会把光标算错"""
    if not COLOR:
        return s
    return _rl("\033[%sm" % code) + s + _rl("\033[0m")


# ---------------------------------------------------------------- 输入清洗
def normalize(s):
    """全角→半角、去零宽字符、压缩空白、去首尾空格（只用于命令解析）"""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    return " ".join(s.split()).strip()


def parse_command(line):
    """返回 (命令名原始, 参数字符串)；不是命令返回 (None, None)"""
    line = normalize(line)
    if not line.startswith((":", "：")):
        return None, None
    body = line[1:].strip()
    if not body:
        return "", ""
    parts = body.replace("=", " ").split(None, 1)
    return parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")


def resolve(name):
    """命令名/别名 → 规范名；认不出来就猜一个最像的"""
    if not name:
        return None, "输入 :help 查看全部命令。"
    if name in ALIAS:
        return ALIAS[name], None
    cand = difflib.get_close_matches(name, list(ALIAS), n=1, cutoff=0.55)
    if cand:
        return None, "没有 :%s 这个命令，你是不是想用 :%s ？" % (name, cand[0])
    return None, "没有 :%s 这个命令。输入 :help 看全部命令。" % name


# ---------------------------------------------------------------- 命令表
# 名字 -> (说明, 用法/范围)
SPEC = {
    "reload": ("重新加载检查点（训练写了新的之后）", ""),
    "load":   ("加载指定检查点；不带参数就列出 out_v6 里的", ":load out_v6/ckpt_best.pt"),
    "auto":   ("自动加载新检查点 开/关", ":auto on"),
    "temp":   ("温度，越大越随机", "0.05-3.0"),
    "k":      ("top_k，0 = 关闭", "0-32000"),
    "p":      ("top_p 核采样，1 = 关闭", "0-1"),
    "minp":   ("min_p 最低概率，0 = 关闭", "0-1"),
    "rep":    ("重复惩罚（>1 压重复）", "0.5-3.0"),
    "n":      ("生成长度，0 = 写到结束符 / Ctrl+C", "0-2000"),
    "seed":   ("随机种子，同种子同上下文可复现", ":seed 42 / :seed random"),
    "stop":   ("停止词，生成到它就停；:stop - 清空", ":stop 第"),
    "info":   ("当前设置 + 检查点信息", ""),
    "eval":   ("在验证集上算 loss / ppl", ":eval 10"),
    "new":    ("清空上下文，重新开始", ""),
    "cont":   ("接着写（等同直接回车）", ""),
    "add":    ("把文字接到上下文末尾（不生成）", ":add 第二章"),
    "last":   ("重看最近一轮的续写", ""),
    "hist":   ("最近几轮概览", ":hist 5"),
    "undo":   ("撤销上一轮", ""),
    "retry":  ("按当前设置重写上一轮", ""),
    "try":    ("从当前上下文生成几个候选，挑一个", ":try 3"),
    "pick":   ("采用 :try 的第几个候选", ":pick 2"),
    "paste":  ("多行粘贴：输入到单独一行 :end", ""),
    "file":   ("读一个 txt 当开头（自动认编码）", ":file ~/a.txt"),
    "save":   ("把这次会话存成 txt", ":save out/第1章.txt"),
    "tok":    ("看一段文字是怎么分词的（排错用）", ":tok 第1章"),
    "help":   ("这份帮助；:help 命令 看单条", ""),
    "q":      ("退出", ""),
}

# 帮助里分组的顺序（"" 表示"直接回车"这一行）
GROUPS = [
    ("生成参数", ["temp", "k", "p", "minp", "rep", "n", "seed", "stop"]),
    ("上下文",   ["", "cont", "new", "add", "last", "hist", "undo", "retry",
                  "try", "pick"]),
    ("检查点",   ["reload", "load", "auto", "info", "eval"]),
    ("输入 / 存档", ["paste", "file", "save", "tok"]),
    ("其他",     ["help", "q"]),
]

ALIAS = {
    "r": "reload", "reload": "reload", "重新加载": "reload",
    "load": "load", "open": "load", "加载": "load",
    "a": "auto", "auto": "auto", "自动": "auto",
    "t": "temp", "temp": "temp", "temperature": "temp", "温度": "temp",
    "k": "k", "topk": "k", "top_k": "k",
    "p": "p", "top_p": "p", "topp": "p", "nucleus": "p", "核": "p",
    "minp": "minp", "min_p": "minp", "mp": "minp",
    "rep": "rep", "r_p": "rep", "penalty": "rep", "重复": "rep",
    "n": "n", "len": "n", "length": "n", "num": "n", "长度": "n",
    "seed": "seed", "种子": "seed",
    "stop": "stop", "停": "stop", "停止": "stop",
    "i": "info", "info": "info", "信息": "info", "status": "info",
    "ev": "eval", "eval": "eval", "ppl": "eval", "评估": "eval",
    "new": "new", "clear": "new", "reset": "new", "清空": "new",
    "c": "cont", "cont": "cont", "continue": "cont", "go": "cont",
    "more": "cont", "继续": "cont", "接着写": "cont",
    "add": "add", "append": "add", "接上": "add",
    "last": "last", "show": "last", "上次": "last",
    "hist": "hist", "history": "hist", "历史": "hist",
    "u": "undo", "undo": "undo", "撤销": "undo",
    "retry": "retry", "again": "retry", "重写": "retry",
    "try": "try", "cand": "try", "cands": "try", "sample": "try", "候选": "try",
    "pick": "pick", "use": "pick", "采用": "pick", "选": "pick",
    "paste": "paste", "pl": "paste", "粘贴": "paste",
    "file": "file", "read": "file", "读": "file",
    "w": "save", "save": "save", "write": "save", "保存": "save",
    "tok": "tok", "tokens": "tok", "分词": "tok",
    "h": "help", "help": "help", "?": "help", "？": "help", "帮助": "help",
    "q": "q", "quit": "q", "exit": "q", "退出": "q",
}

# 数值命令：规范名 -> 设置项、类型、范围、显示格式、中文名
NUM = {
    "temp": dict(attr="temperature", kind=float, lo=0.05, hi=3.0, fmt="%.2f", label="温度"),
    "k":    dict(attr="top_k", kind=int, lo=0, hi=32000, fmt="%d", label="top_k"),
    "p":    dict(attr="top_p", kind=float, lo=0.0, hi=1.0, fmt="%.2f", label="top_p"),
    "minp": dict(attr="min_p", kind=float, lo=0.0, hi=1.0, fmt="%.3f", label="min_p"),
    "n":    dict(attr="n", kind=int, lo=0, hi=2000, fmt="%d", label="生成长度"),
    "rep":  dict(attr="rep", kind=float, lo=0.5, hi=3.0, fmt="%.2f", label="重复惩罚"),
}

# 规范名 -> Repl 里的处理方法（cmd_ 后面的部分）
HANDLER_NAMES = {
    "reload": "reload", "load": "load", "auto": "auto", "info": "info", "eval": "eval",
    "temp": "num", "k": "num", "p": "num", "minp": "num", "rep": "num", "n": "num",
    "seed": "seed", "stop": "stop",
    "new": "new", "cont": "cont", "add": "add",
    "last": "last", "hist": "hist", "undo": "undo", "retry": "retry",
    "try": "try", "pick": "pick",
    "paste": "paste", "file": "file", "save": "save", "tok": "tok",
    "help": "help", "q": "quit",
}

ARG_FORBIDDEN = {"reload", "info", "new", "cont", "undo", "retry", "paste", "q"}


def settings_line(S):
    s = ("温度 %.2f  top_p %.2f  top_k %d  min_p %.3f  长度 %s  重复惩罚 %.2f  种子 %s"
         % (S.temperature, S.top_p, S.top_k, S.min_p,
            "自动" if S.n == 0 else S.n, S.rep,
            "随机" if S.seed is None else S.seed))
    if S.stop:
        s += "  停止词 %s" % "/".join(repr(x) for x in S.stop)
    return s


def help_text(S, ctx_len, auto_on, block):
    out = ["", "  " + C("1", "── 怎么用")]
    out.append("    输入文字  = 用它当开头（覆盖当前上下文）")
    out.append("    直接回车  = 接着刚才写的内容继续")
    out.append("    : 开头    = 命令（全角 ：也行，大小写都行）")
    out.append("")
    for title, names in GROUPS:
        out.append("  " + C("1", "── " + title))
        for canon in names:
            if canon == "":
                out.append("    %-9s 接着写（也可以写 :c 或 :cont）" % "（回车）")
                continue
            desc, usage = SPEC[canon]
            alias = sorted(a for a, c in ALIAS.items() if c == canon and a != canon)
            al = ("（" + "/".join(alias) + "）") if alias else ""
            out.append("    :%-8s %s%s%s" % (canon, desc, al, ("   " + usage) if usage else ""))
        out.append("")
    out.append("  " + C("1", "── 例子"))
    out.append("    第1章 边陲              ← 用这段开头写")
    out.append("    （直接回车）             ← 接着上一段继续")
    out.append("    :temp 0.85   :p 0.95    ← 改完立刻生效")
    out.append("    :save out/第1章.txt     ← 把这次写的存下来")
    out.append("")
    out.append("  " + C("1", "── 当前"))
    out.append("    " + settings_line(S))
    out.append("    上下文 %d / %d token    自动加载 %s"
               % (min(ctx_len, block), block, "开" if auto_on else "关"))
    return "\n".join(out)


# ---------------------------------------------------------------- 文本 / 文件
def read_text_any(path, max_chars=200000):
    """读文本，自动认 utf-8 / gbk / utf-16（中文 txt 一半是 gbk）"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "utf-16"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(max_chars)
        except UnicodeDecodeError:
            continue
        except OSError:
            raise
    with open(path, "rb") as f:
        return f.read(max_chars).decode("utf-8", "replace")


def ckpt_line(path, ck):
    bv = ck.get("best_val", None)
    bv = "%.4f(ppl %.1f)" % (bv, math.exp(min(bv, 30))) if isinstance(bv, (int, float)) else "?"
    return "已加载 %s  step=%s  best_val=%s  n_mem=%s  attn_res=%s" % (
        path, ck.get("step", "?"), bv,
        ck["cfg"].get("n_mem", 0), ck["cfg"].get("attn_res", 0))


def newest(out_dir):
    paths = [os.path.join(out_dir, f) for f in CANDIDATES]
    paths = [p for p in paths if os.path.exists(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def pick_device(want):
    if want and want != "auto":
        if want.startswith("cuda") and not torch.cuda.is_available():
            print("（没有可用的 CUDA，改用 CPU）")
            return "cpu"
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_spm(path):
    """加载分词器；路径不对就试同目录下其它 *.model"""
    import sentencepiece as spm
    cands = [path]
    d = os.path.dirname(path) or "."
    if os.path.isdir(d):
        cands += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".model")]
    last = None
    for c in cands:
        if not os.path.exists(c):
            continue
        try:
            return spm.SentencePieceProcessor(model_file=c)
        except Exception as e:                    # noqa: BLE001
            last = e
    sys.exit("分词器加载失败（%s）%s" % (path, (": %s" % last) if last else ""))


def load_model(path, device):
    last = None
    for i in range(3):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            ck.pop("optim", None)        # 优化器状态几百 MB，聊天用不上
            ck.pop("optimizer", None)
            model = GPT(ck["cfg"]).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            return model, ck
        except Exception as e:                    # noqa: BLE001
            last = e
            if i < 2:
                time.sleep(1.0)
    raise last


def load_safe(path, device):
    try:
        return load_model(path, device)
    except FileNotFoundError:
        print("找不到检查点：%s" % path)
    except Exception as e:                        # noqa: BLE001
        print("加载失败：%s: %s" % (type(e).__name__, e))
    return None, None


def amp_kind(model, device):
    """bf16 autocast 能不能用（CPU / 老卡 / 驱动不支持时退回 fp32）"""
    if not str(device).startswith("cuda"):
        return "off"
    try:
        x = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        return "bf16"
    except Exception:                             # noqa: BLE001
        return "off"


# ---------------------------------------------------------------- 检查点监视
UI = {"busy": False}          # 生成中置 True，Watcher 就不往屏幕上插话


class Watcher(threading.Thread):
    """后台线程：只做 os.stat，发现新检查点就挂个待加载标记。

    为什么不在线程里直接加载模型：
      · 加载要动 GPU（建模型、拷权重），主线程可能正在生成，会抢显存
      · 所以在安全点（两次生成之间）由主线程自己加载
    为什么直接信任 mtime 变化：
      train.py 的 atomic_save 是 "写 .tmp 再 os.replace"，读到的永远是完整文件
    """

    def __init__(self, out_dir, interval=3.0, notify=None):
        super().__init__(daemon=True)
        self.out_dir = out_dir
        self.interval = interval
        self.notify = notify
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = None
        self._last = None

    def mark(self, path):
        """以"当前已加载的文件"为基准，避免刚启动就重载一次"""
        path = os.path.abspath(path)   # 相对路径要和 newest() 返回的绝对路径对得上
        try:
            st = os.stat(path)
        except OSError:
            return
        with self._lock:
            self._last = (path, st.st_mtime_ns, st.st_size)
            self._pending = None

    def take(self):
        with self._lock:
            p, self._pending = self._pending, None
        return p

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                p = newest(self.out_dir)
                if not p:
                    continue
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size <= 0:
                continue
            sig = (p, st.st_mtime_ns, st.st_size)
            with self._lock:
                if sig == self._last:
                    continue
                self._last = sig
                self._pending = p
            if self.notify is not None and not UI["busy"]:
                try:
                    self.notify(p)
                except Exception:                 # noqa: BLE001
                    pass


class _Stop(Exception):
    """命中停止词，用来中断生成"""


# ---------------------------------------------------------------- 交互主对象
class Repl:
    def __init__(self, args, quiet=False):
        self.args = args
        self.quiet = quiet
        self.S = SimpleNamespace(
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            min_p=args.min_p, n=args.n, rep=args.rep_penalty,
            seed=(args.seed if args.seed is not None and args.seed >= 0 else None),
            stop=[s for s in (args.stop or []) if s],
        )
        self.H = {c: getattr(self, "cmd_" + m) for c, m in HANDLER_NAMES.items()}

        self.device = pick_device(args.device)
        self.sp = load_spm(args.spm)
        self.eos = self.sp.piece_to_id("<|endoftext|>")
        if self.eos < 0:                          # 词表里没有就退回 eos_id
            self.eos = self.sp.eos_id()
        self.vocab = self.sp.get_piece_size()

        path = args.ckpt or newest(args.out_dir)
        if args.ckpt and os.path.isdir(args.ckpt):
            path = newest(args.ckpt) or path
        if not path:
            sys.exit("%s 里没有检查点（ckpt_best / ckpt / ckpt_last 都没有）" % args.out_dir)
        self.path = os.path.abspath(path)
        model, ck = load_safe(self.path, self.device)
        if model is None:
            sys.exit(1)
        self.model, self.ck = model, ck
        self.block = ck["cfg"]["block_size"]
        self.amp = amp_kind(self.model, self.device)

        self.text = ""            # 当前文档（上下文）原文
        self.ctx_len = 0          # 它有多少 token
        self.rounds = []          # 这次会话写过的每一轮
        self.pending = None       # :try 生成的候选（before, [候选文本]）
        self.ctrl_c = 0
        self.n_input = 0
        self._trim_warned = False

        self.watcher = None
        if not args.no_auto_reload:
            self.watcher = self._make_watcher()
            self.watcher.start()

        if not quiet:
            self.banner()

    # ------------------------------------------------------------------ 基础
    def _make_watcher(self):
        w = Watcher(self.args.out_dir, self.args.watch_interval, notify=self._notify_new)
        w.mark(self.path)
        return w

    def _notify_new(self, p):
        try:
            import readline
            buf = readline.get_line_buffer()
        except Exception:                         # noqa: BLE001
            buf = ""
        sys.stdout.write("\r\033[K[发现新检查点 %s]\n%s%s"
                         % (os.path.basename(p), self.prompt(), buf))
        sys.stdout.flush()

    def amp_ctx(self):
        if self.amp == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def set_text(self, txt):
        """换上下文：末尾裁到 MAX_TEXT_CHARS 字，顺便记 token 数"""
        self.text = txt[-MAX_TEXT_CHARS:] if len(txt) > MAX_TEXT_CHARS else txt
        self.ctx_len = len(self.sp.encode(self.text)) if self.text else 0

    def prompt(self):
        S = self.S
        body = "T%.2f P%.2f K%d N%s C%d" % (
            S.temperature, S.top_p, S.top_k,
            "auto" if S.n == 0 else S.n, min(self.ctx_len, self.block))
        return CP("36", "你") + CP("90", "[" + body + "]") + "> "

    def banner(self):
        cfg = self.ck["cfg"]
        n_par = self.model.num_params(False) / 1e6
        n_ne = self.model.num_params(True) / 1e6
        print(ckpt_line(self.path, self.ck))
        print("  设备 %s（%s）  %d 层 d=%d %d 头  参数 %.1fM（非嵌入 %.1fM）"
              % (self.device, "bf16" if self.amp == "bf16" else "fp32",
                 cfg["n_layer"], cfg["n_embd"], cfg["n_head"], n_par, n_ne))
        print("  上下文 %d token  n_mem %d  分词器 %d 词"
              % (self.block, cfg.get("n_mem", 0), self.vocab))
        print(C("2", "  输入文字 = 开头；直接回车 = 接着写；:help 看命令，:q 退出"))
        if self.watcher is not None:
            print(C("90", "  （自动加载已开：训练存了新检查点，会在下一次生成前换上）"))
        print()

    # ------------------------------------------------------------------ 切检查点
    def _swap(self, model, ck, path):
        self.model, self.ck, self.path = model, ck, os.path.abspath(path)
        self.block = ck["cfg"]["block_size"]
        self.amp = amp_kind(self.model, self.device)
        if self.watcher is not None:
            self.watcher.mark(self.path)
        print("↻ " + ckpt_line(self.path, self.ck))

    def try_reload(self):
        """安全点加载：只在两次生成之间调用，由主线程动手，不跟后台线程抢 GPU"""
        if self.watcher is None:
            return False
        p = self.watcher.take()
        if not p:
            return False
        old = self.ck.get("step", 0)
        m, c = load_safe(p, self.device)
        if m is None:
            return False
        self._swap(m, c, p)
        new = c.get("step", 0)
        if isinstance(new, int) and isinstance(old, int) and new > old:
            print("   (+%d 步)" % (new - old))
        return True

    # ------------------------------------------------------------------ 生成
    def generate(self, use_new_text=None, stats=True, prefix=None, stream=True):
        """use_new_text 不为 None → 用它当开头；None → 接着当前上下文写"""
        S = self.S
        if use_new_text is not None:
            before = use_new_text
        else:
            if not self.text:
                print("（还没有上下文：先输入一段开头，或 :file 读个文件）")
                return None
            before = self.text
        self.set_text(before)

        ids = self.sp.encode(before) if before else []
        if len(ids) > self.block:
            if not self._trim_warned:
                print(C("90", "（开头超过上下文，只保留最后 %d token）" % self.block))
                self._trim_warned = True
            ids = ids[-self.block:]
        if not ids:
            ids = [self.eos]

        limit = S.n if S.n > 0 else AUTO_CAP
        stops = [s for s in S.stop if s]
        hold = (max(len(s) for s in stops) + 16) if stops else 0
        out_ids = []
        printed = 0
        stop_at = [None]
        stop_hit = [None]
        reason = ""

        def visible(final):
            txt = self.sp.decode(out_ids)
            if not final:
                txt = txt.rstrip("\ufffd")
                if hold:
                    txt = txt[:max(0, len(txt) - hold)]
            elif stop_at[0] is not None:
                txt = txt[:stop_at[0]]
            return txt

        def draw(final):
            nonlocal printed
            txt = visible(final)
            if len(txt) > printed:
                if stream:
                    sys.stdout.write(txt[printed:])
                    sys.stdout.flush()
                printed = len(txt)

        def on_token(tok):
            if tok == self.eos:                   # 结束符不显示
                return
            out_ids.append(tok)
            if stops:
                txt = self.sp.decode(out_ids)
                for s in stops:
                    p = txt.find(s)
                    if p >= 0:
                        stop_at[0] = p
                        stop_hit[0] = s
                        raise _Stop()
            draw(False)

        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        if S.seed is not None:
            torch.manual_seed(S.seed)

        pfx = "续> " if prefix is None else prefix
        if pfx:
            sys.stdout.write(C("36", pfx))
            sys.stdout.flush()
        UI["busy"] = True
        t0 = time.time()
        try:
            with torch.no_grad(), self.amp_ctx():
                self.model.generate(x, limit, temperature=S.temperature,
                                    top_k=S.top_k or None, top_p=S.top_p, min_p=S.min_p,
                                    eos_id=self.eos, repetition_penalty=S.rep,
                                    on_token=on_token)
        except _Stop:
            reason = "命中停止词"
        except KeyboardInterrupt:
            reason = "手动打断"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n显存不够了：把 :n 调小（比如 :n %d），或关掉别的占显存的程序"
                      % max(1, limit // 2))
            else:
                print("\n生成出错：%s: %s" % (type(e).__name__, e))
            return None
        except Exception as e:                    # noqa: BLE001
            print("\n生成出错：%s: %s" % (type(e).__name__, e))
            return None
        finally:
            UI["busy"] = False

        draw(True)
        if stream:
            print()
        dt = max(1e-6, time.time() - t0)
        full = self.sp.decode(out_ids)
        gen_text = full[:stop_at[0]] if stop_at[0] is not None else full
        ntok = len(out_ids)
        if not reason:
            reason = "遇到结束符" if ntok < limit else "到达上限 %d" % limit
        elif stop_hit[0] is not None:
            reason = "命中停止词「%s」" % stop_hit[0]

        if ntok and full.count("\ufffd") / len(full) > 0.05:
            print("    ⚠ 字节碎片偏多（开头太短或太生僻），换个更完整的开头会更稳")
        if not ntok:
            print("    （模型立刻吐了结束符：换个开头，或把 :temp 调高一点）")
        elif not gen_text:
            print("    （写出来的全被停止词吃掉了：:stop - 清空，或换个开头）")

        base = self.sp.decode(ids) if before else ""
        self.set_text(base + gen_text)
        rec = {
            "prompt": use_new_text, "before": before, "out": gen_text,
            "n": ntok, "secs": dt, "reason": reason,
            "time": time.strftime("%H:%M:%S"),
            "settings": settings_line(S),
        }
        self.rounds.append(rec)
        if stats:
            print(C("90", "   [%d token / %.1fs = %.1f tok/s · %s · 上下文 %d token]"
                    % (ntok, dt, ntok / dt, reason, min(self.ctx_len, self.block))))
            print()
        return rec

    # ------------------------------------------------------------------ 命令
    def cmd_num(self, canon, arg):
        spec = NUM[canon]
        cur = getattr(self.S, spec["attr"])
        lo, hi = spec["lo"], spec["hi"]
        if not arg:
            print(":%s 现在 = %s，用法 :%s <数字>（范围 %s ~ %s）"
                  % (canon, spec["fmt"] % cur, canon, lo, hi))
            if canon == "p" and cur >= 0.999:
                print("  （1 = 关闭核采样）")
            if canon == "minp" and cur <= 0:
                print("  （0 = 关闭）")
            if canon == "n" and cur == 0:
                print("  （0 = 一直写到结束符，硬上限 %d）" % AUTO_CAP)
            return True
        try:
            val = spec["kind"](float(arg))
        except (TypeError, ValueError):
            print(":%s 需要一个数字，你输入的是 %r" % (canon, arg))
            return True
        if not (lo <= val <= hi):
            print(":%s 的 %s 超出范围 %s ~ %s，没有改" % (canon, val, lo, hi))
            return True
        setattr(self.S, spec["attr"], val)
        msg = "→ %s = %s" % (spec["label"], spec["fmt"] % val)
        if canon == "p" and val >= 0.999:
            msg += "（关闭核采样）"
        if canon == "minp" and val <= 0:
            msg += "（关闭）"
        if canon == "n":
            msg += "（0 = 一直写到结束符或 Ctrl+C 停，最多 %d）" % AUTO_CAP if val == 0 \
                else " token"
        print(msg)
        return True

    def cmd_seed(self, canon, arg):
        self._bump = 0
        a = (arg or "").strip().lower()
        if not a:
            print(":seed 现在 = %s" % ("随机" if self.S.seed is None else self.S.seed))
            print("  用法 :seed 42（同种子同上下文可复现） / :seed random（每次随机）")
            return True
        if a in ("random", "rand", "-1", "-", "none", "随机"):
            self.S.seed = None
            print("→ 种子 = 随机")
            return True
        try:
            v = int(float(a))
        except ValueError:
            print(":seed 需要一个整数，你输入的是 %r" % arg)
            return True
        if v < 0:
            self.S.seed = None
            print("→ 种子 = 随机")
        else:
            self.S.seed = v
            print("→ 种子 = %d" % v)
        return True

    def cmd_stop(self, canon, arg):
        a = (arg or "").strip()
        if not a:
            if not self.S.stop:
                print(":stop 现在 = 无，用法 :stop 第（生成到「第」就停）")
            else:
                print(":stop 现在 = %s" % "  ".join(repr(s) for s in self.S.stop))
                print("  :stop -（清空）  :stop 新词（再加一个，最多 5 个）")
            return True
        if a in ("-", "clear", "none", "清空"):
            self.S.stop = []
            print("→ 停止词已清空")
            return True
        if len(a) > 32:
            print("停止词太长了（最多 32 字）")
            return True
        if a in self.S.stop:
            print("→ 已经有这个停止词了")
            return True
        if len(self.S.stop) >= 5:
            print("最多 5 个停止词，先 :stop - 清一下")
            return True
        self.S.stop.append(a)
        print("→ 停止词 %d 个：%s" % (len(self.S.stop), "  ".join(repr(s) for s in self.S.stop)))
        return True

    def cmd_auto(self, canon, arg):
        a = (arg or "").strip().lower()
        if a in ("", "toggle", "切换"):
            want = self.watcher is None
        elif a in ("on", "1", "开", "y", "yes", "true"):
            want = True
        elif a in ("off", "0", "关", "n", "no", "false"):
            want = False
        else:
            print(":auto 只认 on / off（也可以 :auto 开 / :auto 关，不带参数就是切换）")
            return True
        if want and self.watcher is None:
            self.watcher = self._make_watcher()
            self.watcher.start()
            print("→ 自动加载：开")
        elif not want and self.watcher is not None:
            self.watcher.stop()
            self.watcher = None
            print("→ 自动加载：关（还能用 :reload 手动加载）")
        else:
            print("→ 自动加载本来就是%s" % ("开" if want else "关"))
        return True

    def cmd_info(self, canon, arg):
        cfg = self.ck["cfg"]
        print(ckpt_line(self.path, self.ck))
        print("  模型 %d 层 d=%d %d 头  block %d  n_mem %d  attn_res %s"
              % (cfg["n_layer"], cfg["n_embd"], cfg["n_head"], cfg["block_size"],
                 cfg.get("n_mem", 0), cfg.get("attn_res", 0)))
        print("  " + settings_line(self.S))
        print("  上下文 %d / %d token（%d 字）  自动加载 %s"
              % (min(self.ctx_len, self.block), self.block, len(self.text),
                 "开" if self.watcher is not None else "关"))
        print("  设备 %s（%s）  分词器 %d 词  这次写了 %d 轮"
              % (self.device, "bf16" if self.amp == "bf16" else "fp32",
                 self.vocab, len(self.rounds)))
        return True

    def cmd_reload(self, canon, arg):
        newp = newest(self.args.out_dir) or self.path
        m, c = load_safe(newp, self.device)
        if m is not None:
            self._swap(m, c, newp)
        return True

    def cmd_load(self, canon, arg):
        a = (arg or "").strip().strip("\"'")
        if not a:
            print("  用法 :load <检查点>；%s 里有：" % self.args.out_dir)
            for f in CANDIDATES:
                p = os.path.join(self.args.out_dir, f)
                if os.path.exists(p):
                    st = os.stat(p)
                    print("    %-40s %7.1f MB  %s"
                          % (p, st.st_size / 1e6,
                             time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime))))
            print("  当前：%s" % self.path)
            return True
        p = os.path.expanduser(a)
        if os.path.isdir(p):
            p = newest(p) or p
        if not os.path.isfile(p):
            print("找不到文件：%s" % p)
            return True
        m, c = load_safe(p, self.device)
        if m is not None:
            self._swap(m, c, p)
        return True

    def eval_ppl(self, n_batches=10):
        import numpy as np
        val = np.memmap(os.path.join(self.args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
        T = self.block
        g = torch.Generator().manual_seed(0)
        total = 0.0
        self.model.eval()
        with torch.no_grad(), self.amp_ctx():
            for _ in range(n_batches):
                i = int(torch.randint(len(val) - T - 1, (1,), generator=g).item())
                chunk = torch.from_numpy(val[i:i + T + 1].astype(np.int64))
                x = chunk[:-1].unsqueeze(0).to(self.device)
                y = chunk[1:].unsqueeze(0).to(self.device)
                _, loss, _ = self.model(x, y)
                total += float(loss)
        mean = total / max(1, n_batches)
        return mean, math.exp(min(mean, 30))

    def cmd_eval(self, canon, arg):
        n = 10
        if (arg or "").strip():
            try:
                n = max(1, min(100, int(float(arg))))
            except ValueError:
                print(":eval 需要一个批数（默认 10）")
                return True
        t0 = time.time()
        print("  在验证集上算 %d 批（每批 %d token）…" % (n, self.block))
        try:
            loss, ppl = self.eval_ppl(n)
        except FileNotFoundError:
            print("找不到 %s/val.bin" % self.args.data_dir)
            return True
        except Exception as e:                    # noqa: BLE001
            print("评估失败：%s: %s" % (type(e).__name__, e))
            return True
        best = self.ck.get("best_val")
        extra = ""
        if isinstance(best, (int, float)) and best > 0:
            extra = "（检查点记录 best_val %.4f，差 %.4f）" % (best, loss - best)
        print("→ loss %.4f  ppl %.2f%s  用时 %.1fs" % (loss, ppl, extra, time.time() - t0))
        return True

    def cmd_new(self, canon, arg):
        self.set_text("")
        self.rounds = []
        print("→ 上下文已清空，重新开始")
        return True

    def cmd_cont(self, canon, arg):
        if not self.text:
            print("（还没有上下文：先输入一段开头，或 :file 读个文件）")
            return True
        self.generate(None)
        return True

    def cmd_add(self, canon, arg):
        a = arg or ""
        if not a.strip():
            print("用法 :add 要接上的文字（只改上下文，不生成）")
            return True
        self.set_text(self.text + a)
        print("→ 已接到上下文末尾：%d 字 / %d token"
              % (len(self.text), min(self.ctx_len, self.block)))
        return True

    def cmd_last(self, canon, arg):
        if not self.rounds:
            print("（还没写过）")
            return True
        r = self.rounds[-1]
        print(C("90", "  [第 %d 轮  %s  %d token  %s]"
                % (len(self.rounds), r["time"], r["n"], r["reason"])))
        print(r["out"])
        return True

    def cmd_hist(self, canon, arg):
        if not self.rounds:
            print("（还没写过）")
            return True
        n = 5
        if (arg or "").strip():
            try:
                n = max(1, int(float(arg)))
            except ValueError:
                print("用法 :hist [轮数]")
                return True
        recent = self.rounds[-n:]
        start = len(self.rounds) - len(recent)
        for i, r in enumerate(recent, start + 1):
            head = (r["prompt"] if r["prompt"] is not None else "（接着上一轮）")
            head = " ".join(head.split())[:30]
            out = " ".join(r["out"].split())[:50]
            print("  %2d) %s  %s  %d token\n      %s → %s"
                  % (i, r["time"], r["reason"], r["n"], head, out))
        return True

    def cmd_undo(self, canon, arg):
        if not self.rounds:
            print("（没有可撤销的轮次）")
            return True
        r = self.rounds.pop()
        self.set_text(r["before"])
        print("→ 已撤销第 %d 轮（%d token），上下文回到 %d token"
              % (len(self.rounds) + 1, r["n"], min(self.ctx_len, self.block)))
        return True

    def cmd_retry(self, canon, arg):
        if not self.rounds:
            print("（还没有可重写的轮次）")
            return True
        r = self.rounds.pop()
        self.set_text(r["before"])
        print("↻ 重写第 %d 轮" % (len(self.rounds) + 1))
        old = self.S.seed
        if old is not None:
            # 种子固定时，重写必须换个种子，否则逐字复现上一轮
            self._bump = getattr(self, "_bump", 0) + 1
            self.S.seed = old + self._bump
            print("  （种子 %d → %d；:seed %d 可还原）" % (old, self.S.seed, old))
        try:
            self.generate(r["prompt"])
        finally:
            self.S.seed = old
        return True

    def cmd_try(self, canon, arg):
        if not self.text:
            print("（还没有上下文：先输入一段开头，或 :file 读个文件）")
            return True
        k = 3
        if (arg or "").strip():
            try:
                k = max(1, min(8, int(float(arg))))
            except ValueError:
                print("用法 :try [候选数]（默认 3，最多 8）")
                return True
        base_seed = self.S.seed
        before = self.text
        n_rounds = len(self.rounds)
        outs = []
        for i in range(k):
            if base_seed is not None:      # 固定种子时每个候选换一个，否则全一样
                self.S.seed = base_seed + i
            self.set_text(before)
            rec = self.generate(None, stats=False, prefix="", stream=False)
            if rec is None:
                break
            outs.append(rec["out"])
        self.S.seed = base_seed
        del self.rounds[n_rounds:]         # 候选先不进历史，:pick 了才算
        self.set_text(before)              # 上下文回到生成前
        if not outs:
            print("（没生成出候选）")
            return True
        self.pending = (before, outs)
        print()
        for i, t in enumerate(outs, 1):
            print("  [%d] %s" % (i, t.strip() if t.strip() else "（空）"))
        print(C("90", "  :pick 2 采用第 2 个；:try %d 再来一批" % k))
        return True

    def cmd_pick(self, canon, arg):
        if not self.pending:
            print("（先用 :try 生成几个候选）")
            return True
        before, outs = self.pending
        k = 1
        if (arg or "").strip():
            try:
                k = int(float(arg))
            except ValueError:
                print("用法 :pick <序号>")
                return True
        if not (1 <= k <= len(outs)):
            print("序号要在 1 ~ %d 之间" % len(outs))
            return True
        txt = outs[k - 1]
        self.set_text(before + txt)
        self.rounds.append({
            "prompt": None, "before": before, "out": txt,
            "n": len(self.sp.encode(txt)), "secs": 0.0,
            "reason": "采用候选 %d" % k,
            "time": time.strftime("%H:%M:%S"), "settings": settings_line(self.S),
        })
        self.pending = None
        print("→ 已采用候选 %d（%d 字），上下文 %d token"
              % (k, len(txt), min(self.ctx_len, self.block)))
        return True

    def cmd_paste(self, canon, arg):
        print(C("90", "  把内容粘进来，最后单独一行输入 :end（Ctrl-D 也行）"))
        buf = []
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                print("\n（已取消粘贴）")
                return True
            if line == "":
                break
            s = line.rstrip("\r\n")
            if s.strip().lower() in (":end", "：end", ":done", ":e"):
                break
            buf.append(s)
        txt = "\n".join(buf).strip()
        if not txt:
            print("（没有内容）")
            return True
        self.set_text(txt)
        print("→ 上下文已换成粘贴的内容：%d 字 / %d token"
              % (len(self.text), min(self.ctx_len, self.block)))
        return True

    def cmd_file(self, canon, arg):
        a = (arg or "").strip().strip("\"'")
        if not a:
            print("用法 :file <txt 路径>（读进来当开头）")
            return True
        p = os.path.expanduser(a)
        if not os.path.isfile(p):
            print("找不到文件：%s" % p)
            return True
        try:
            txt = read_text_any(p)
        except OSError as e:
            print("读取失败：%s" % e)
            return True
        if not txt.strip():
            print("文件是空的：%s" % p)
            return True
        self.set_text(txt.strip())
        print("→ 已读入 %s：%d 字 / %d token（只取末尾 %d 字）"
              % (p, len(self.text), min(self.ctx_len, self.block), MAX_TEXT_CHARS))
        return True

    def cmd_save(self, canon, arg):
        if not self.rounds and not self.text:
            print("（这次会话还没写东西）")
            return True
        a = (arg or "").strip().strip("\"'")
        if not a:
            a = os.path.join(self.args.out_dir,
                             "chat_%s.txt" % time.strftime("%Y%m%d_%H%M%S"))
        p = os.path.expanduser(a)
        if not os.path.splitext(p)[1]:
            p += ".txt"
        d = os.path.dirname(os.path.abspath(p))
        try:
            os.makedirs(d, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write("miniNLM 会话  %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
                f.write("检查点: %s\n" % self.path)
                f.write("设置: %s\n" % settings_line(self.S))
                f.write("轮次: %d\n\n" % len(self.rounds))
                f.write("=" * 60 + "\n当前全文（最多末尾 %d 字）\n" % MAX_TEXT_CHARS + "=" * 60 + "\n")
                f.write(self.text + "\n\n")
                f.write("=" * 60 + "\n逐轮记录\n" + "=" * 60 + "\n")
                for i, r in enumerate(self.rounds, 1):
                    f.write("\n### 第 %d 轮  %s  %s\n" % (i, r["time"], r["settings"]))
                    if r["prompt"] is not None:
                        f.write("【开头】\n%s\n" % r["prompt"])
                    else:
                        f.write("【开头】（接着上一轮）\n")
                    f.write("【续写】%d token / %.1fs / %s\n%s\n"
                            % (r["n"], r["secs"], r["reason"], r["out"]))
        except OSError as e:
            print("保存失败：%s" % e)
            return True
        chars = sum(len(r["out"]) for r in self.rounds) or len(self.text)
        print("→ 已保存 %s（%d 轮 / %d 字）" % (p, len(self.rounds), chars))
        return True

    def cmd_tok(self, canon, arg):
        txt = arg if (arg and arg.strip()) else self.text
        if not txt:
            print("（没有可分词的内容：:tok 一段文字，或先写点东西）")
            return True
        ids = self.sp.encode(txt)
        pieces = self.sp.encode(txt, out_type=str)
        print("   %d 字 / %d token（%.2f 字每 token）"
              % (len(txt), len(ids), len(txt) / max(1, len(ids))))
        show = list(zip(pieces, ids))[:40]
        print("   " + " | ".join("%s(%d)" % (p.replace("\u2581", "_"), i) for p, i in show))
        if len(pieces) > 40:
            print("   …（共 %d 个）" % len(pieces))
        return True

    def cmd_help(self, canon, arg):
        a = (arg or "").strip()
        if a:
            name, err = resolve(normalize(a))
            if err or name is None:
                print(err or "没有这个命令")
                return True
            if name in NUM:
                self.cmd_num(name, "")
                return True
            desc, usage = SPEC[name]
            print("  :%s  %s%s" % (name, desc, ("   用法 " + usage) if usage else ""))
            return True
        print(help_text(self.S, self.ctx_len, self.watcher is not None, self.block))
        return True

    def cmd_quit(self, canon, arg):
        return False

    # ------------------------------------------------------------------ 分发
    def dispatch(self, name, arg):
        canon, err = resolve(name)
        if err:
            print(err)
            return True
        if canon in ARG_FORBIDDEN and arg:
            print(":%s 不接受参数（你输入的是 %r）" % (canon, arg))
            return True
        fn = self.H.get(canon)
        if fn is None:
            print(":%s 还没实现" % canon)
            return True
        return fn(canon, arg)

    # ------------------------------------------------------------------ 主循环
    def loop(self):
        while True:
            try:
                raw = input(self.prompt())
                self.ctrl_c = 0
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                self.ctrl_c += 1
                print()
                if self.ctrl_c >= 2:
                    break
                print("（再按一次 Ctrl+C 或输入 :q 退出）")
                continue

            self.n_input += 1
            if self.n_input % 10 == 0:
                save_history()

            line = normalize(raw)
            if not line:
                # 空行 = 接着写。不用 normalize 过的原文，保留用户自己敲的标点
                self.cmd_cont("cont", "")
                continue

            name, arg = parse_command(line)
            if name is None:
                # 普通文字：整行原文当开头（不做全角→半角，标点原样保留）
                self.try_reload()
                self.generate(raw.strip())
                self.try_reload()
                continue
            if not self.dispatch(name, arg):
                break


# ---------------------------------------------------------------- readline
def setup_readline():
    global _RL
    try:
        import readline                                    # noqa: F401
    except Exception:                                      # noqa: BLE001
        return None
    _RL[0] = True
    hist = os.path.expanduser(HISTORY_PATH)
    try:
        if os.path.exists(hist):
            readline.read_history_file(hist)
        readline.set_history_length(500)
    except Exception:                                      # noqa: BLE001
        pass
    return hist


def save_history():
    try:
        import readline
        readline.write_history_file(os.path.expanduser(HISTORY_PATH))
    except Exception:                                      # noqa: BLE001
        pass


# ---------------------------------------------------------------- 参数 / 自检
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="miniNLM 交互续写")
    p.add_argument("--ckpt", default="", help="检查点路径；不给就自动选 out/ 里最新的")
    p.add_argument("--out_dir", default=ROOT + "/out")
    p.add_argument("--data_dir", default=ROOT + "/data_v4", help=":eval 用的验证集目录")
    p.add_argument("--spm", default=ROOT + "/data_v4/spm48k_v4.model")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=1.0, help="核采样，1 = 关闭")
    p.add_argument("--min_p", type=float, default=0.0, help="最低概率，0 = 关闭")
    p.add_argument("--n", type=int, default=0,
                   help="生成长度，0 = 一直写到结束符（硬上限 %d）" % AUTO_CAP)
    p.add_argument("--rep_penalty", type=float, default=1.15)
    p.add_argument("--seed", type=int, default=-1, help="随机种子，-1 = 每次随机")
    p.add_argument("--stop", action="append", default=[], help="停止词，可重复给")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    p.add_argument("--once", default=None,
                   help="非交互：续写这段文字后退出（'-' 表示从 stdin 读）")
    p.add_argument("--num", type=int, default=1, help="--once 时生成几个样本")
    p.add_argument("--stats", action="store_true", help="--once 时也打印统计")
    p.add_argument("--selftest", action="store_true", help="只测命令解析，不加载模型")
    p.add_argument("--no_auto_reload", action="store_true", help="关闭自动加载新检查点")
    p.add_argument("--watch_interval", type=float, default=3.0, help="查新检查点的间隔（秒）")
    return p.parse_args(argv)


def selftest():
    """不加载模型，只查：命令表对不对、解析会不会崩、数值命令有没有提示"""
    problems = []
    if set(HANDLER_NAMES) != set(SPEC):
        problems.append("HANDLER_NAMES 和 SPEC 不一致：%s"
                        % (set(HANDLER_NAMES) ^ set(SPEC)))
    for canon, m in HANDLER_NAMES.items():
        if not hasattr(Repl, "cmd_" + m):
            problems.append("缺方法 cmd_%s（命令 :%s）" % (m, canon))
    for a, c in ALIAS.items():
        if c not in SPEC:
            problems.append("别名 :%s → :%s 不存在" % (a, c))

    S = SimpleNamespace(temperature=0.9, top_k=50, top_p=1.0, min_p=0.0,
                        n=200, rep=1.15, seed=None, stop=[])
    cases = [
        ":temp 1.0", ":temp=1.1", "：temp 1.2", ":TEMP 1.3", ":t 0.5",
        ":temp", ":temp abc", ":temp 9", ":temp -1", ":tmp 0.8", ":emperature 1.0",
        ":k 0", ":k 99999", ":k 32", ":topk 10", ":K 20",
        ":p 0.95", ":p 1", ":top_p 0.9", ":p 2",
        ":minp 0.05", ":minp", ":mp 3",
        ":n 300", ":n 0", ":n 99999", ":len 100",
        ":rep 1.2", ":rep 5", ":rep x",
        ":seed 42", ":seed random", ":seed -3", ":seed abc", ":种子 7",
        ":stop", ":stop 第", ":stop -", ":stop 一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十",
        ":help", ":?", ":h", ":help temp", ":info", ":i", ":q", ":quit", ":exit",
        ":auto", ":auto on", ":auto 关", ":auto xyz", ":a",
        ":new", ":c", ":cont", ":undo", ":retry", ":last", ":hist", ":hist 3",
        ":try", ":try 4", ":cand 2", ":pick", ":pick 2", ":采用 1",
        ":save", ":file", ":paste", ":tok", ":eval", ":load",
        ":", "::", ":unknown", ":  ", "  :temp 0.7  ",
        "你好世界", "   ", "",
        ":temp ０.８",           # 全角数字
        ":temp 0.8 0.9",         # 多余参数
    ]
    for c in cases:
        name, arg = parse_command(c)
        if name is None:
            print("  %-26r → 正文" % c)
            continue
        canon, err = resolve(name)
        if err:
            print("  %-26r → %s" % (c, err))
            continue
        try:
            if canon in NUM:
                print("  %-26r → %s" % (c, num_preview(canon, arg, S)))
            else:
                print("  %-26r → 命令 %s（arg=%r）" % (c, canon, arg))
        except Exception as e:                            # noqa: BLE001
            problems.append("用例 %r 抛异常 %s: %s" % (c, type(e).__name__, e))
    print("\n设置: " + settings_line(S))
    print("命令 %d 个，别名 %d 个" % (len(SPEC), len(ALIAS)))
    if problems:
        print("\n发现 %d 个问题：" % len(problems))
        for p in problems:
            print("  · " + p)
        return 1
    print("自检完成：命令表一致，全部输入都被接住了，没有抛异常")
    return 0


def num_preview(canon, arg, S):
    """自检里不建 Repl，单独算一下数值命令的结果文本"""
    spec = NUM[canon]
    cur = getattr(S, spec["attr"])
    if not arg:
        return ":%s = %s（用法 :%s <数字>）" % (canon, spec["fmt"] % cur, canon)
    try:
        val = spec["kind"](float(arg))
    except (TypeError, ValueError):
        return ":%s 需要数字，收到 %r" % (canon, arg)
    if not (spec["lo"] <= val <= spec["hi"]):
        return ":%s %s 超范围 %s~%s" % (canon, val, spec["lo"], spec["hi"])
    setattr(S, spec["attr"], val)
    return "→ %s = %s" % (spec["label"], spec["fmt"] % val)


def main():
    args = parse_args()
    if args.selftest:
        return selftest()

    setup_readline()
    repl = Repl(args)

    if args.once is not None:
        text = sys.stdin.read() if args.once == "-" else args.once
        text = (text or "").strip()
        if not text:
            sys.exit("--once 需要一段开头（或者 --once - 从 stdin 读）")
        n = max(1, args.num)
        for i in range(n):
            if n > 1:
                print("─── 样本 %d / %d ───" % (i + 1, n))
            repl.text = ""
            repl.rounds = []
            if repl.S.seed is not None:   # 固定种子时错开，否则 N 个样本一模一样
                repl.S.seed = args.seed + i
            repl.generate(text, stats=args.stats, prefix="")
            if n > 1:
                print()
        return 0

    try:
        repl.loop()
    finally:
        if repl.watcher is not None:
            repl.watcher.stop()
        save_history()
    return 0


if __name__ == "__main__":
    sys.exit(main())
