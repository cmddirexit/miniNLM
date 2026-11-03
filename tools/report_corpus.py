#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料体检：扫一遍 books 目录，统计脏东西，输出 JSONL + 汇总。

用法:
    python tools/report_corpus.py --dir data_v3/books --out data_v3/_report.jsonl [--limit 300]

输出每本书一行 JSON（供筛选），并对若干指标打印分布。
"""
import argparse, glob, json, os, random, re, sys
from collections import Counter
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RE_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,9}(?:\s[^<>]{0,80})?/?>")
RE_ENT = re.compile(r"&[A-Za-z0-9#]{1,10};")
RE_URL = re.compile(r"https?://|www\.|\.com|\.net|\.cc")
RE_AD = re.compile(
    r"笔趣阁|顶点小说|书包网|txt下载|一秒记住|请收藏|求月票|求推荐票|作者有话|作者的话"
    r"|本章未完|点击下一页|手机用户请|最新章节|全文阅读|无弹窗|广告|加群|QQ群|微信号"
)
RE_META = re.compile(
    r"^(《.*》|作者[:：].*|书名[:：].*|简介[:：].*|内容简介[:：]?|目录|序言|前言|后记"
    r"|声明[:：].*|版权.*|ISBN|出版社.*|字数[:：].*|更新时间[:：].*|正文$|正文\s.*|楔子|引子)$"
)
RE_CHAPTER = re.compile(r"^\s*(第[0-9零一二三四五六七八九十百千两]{1,7}[章回节卷集]|正文\s|番外|序章|终章|大结局)")
RE_DONE = re.compile(r"全书完|全本完|完结|完本|(全书|本文|本书)?结束[。！]?$")
RE_FULLWIDTH = re.compile(r"[，。！？；：“”‘’（）《》、…—]")
RE_HALF = re.compile(r"[,.!?;:\"()<>]")
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_REPL = re.compile(r"\ufffd")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def one(path):
    try:
        raw = open(path, "rb").read()
        t = raw.decode("utf-8", "replace")
    except Exception as e:
        return {"id": os.path.basename(path), "err": str(e)}
    lines = t.split("\n")
    n = len(t)
    nonempty = [l for l in lines if l.strip()]
    short = sum(1 for l in nonempty if len(l.strip()) <= 4)
    dup = 0
    seen = set()
    for l in nonempty:
        s = l.strip()
        if len(s) >= 8:
            if s in seen:
                dup += 1
            else:
                seen.add(s)
    d = {
        "id": os.path.basename(path),
        "chars": n,
        "lines": len(lines),
        "blank": round(1 - len(nonempty) / max(1, len(lines)), 4),
        "short_lines": short,
        "dup_lines": dup,
        "cjk": len(RE_CJK.findall(t)),
        "repl": len(RE_REPL.findall(t)),
        "ctrl": len(RE_CTRL.findall(t)),
        "tag": len(RE_TAG.findall(t)),
        "ent": len(RE_ENT.findall(t)),
        "url": len(RE_URL.findall(t)),
        "ad": len(RE_AD.findall(t)),
        "meta": sum(1 for l in nonempty if RE_META.match(l.strip())),
        "chap": sum(1 for l in nonempty if RE_CHAPTER.match(l)),
        "done": len(RE_DONE.findall(t)),
        "fw": len(RE_FULLWIDTH.findall(t)),
        "hw": len(RE_HALF.findall(t)),
        "maxline": max((len(l) for l in lines), default=0),
    }
    d["cjk_ratio"] = round(d["cjk"] / max(1, n), 4)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=ROOT + "/data_v3/books")
    ap.add_argument("--out", default=ROOT + "/data_v3/_report.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.txt")))
    if a.limit and a.limit < len(files):
        random.seed(0)
        files = random.sample(files, a.limit)
    print("扫描 %d 个文件，%d 进程" % (len(files), a.workers), flush=True)

    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, files, chunksize=4), 1):
            rows.append(r)
            if i % 500 == 0:
                print("  %d/%d" % (i, len(files)), flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("明细 -> %s" % a.out)

    ok = [r for r in rows if "err" not in r]
    tot = sum(r["chars"] for r in ok)
    print("\n=== 汇总（%d 本，%.2f 亿字）===" % (len(ok), tot / 1e8))

    def pct(key, q=0.5):
        v = sorted(r[key] for r in ok)
        return v[int(q * (len(v) - 1))]

    for key, label in [
        ("cjk_ratio", "汉字占比"), ("blank", "空行比例"), ("short_lines", "≤4字短行"),
        ("dup_lines", "重复行"), ("repl", "替换符"), ("tag", "HTML标签"),
        ("ent", "HTML实体"), ("url", "URL"), ("ad", "广告词"), ("meta", "元信息行"),
        ("chap", "章节标记"), ("done", "完结标记"), ("maxline", "最长行"),
    ]:
        hit = sum(1 for r in ok if r[key] > 0)
        print("  %-8s 有问题的书 %5d/%d (%5.1f%%)   中位 %s   最大 %s"
              % (label, hit, len(ok), 100 * hit / len(ok), pct(key), max(r[key] for r in ok)))

    fw = sum(r["fw"] for r in ok)
    hw = sum(r["hw"] for r in ok)
    print("  全角标点 %d  半角标点 %d   比值 %.2f" % (fw, hw, fw / max(1, hw)))
    tiny = [r for r in ok if r["chars"] < 20000]
    print("  <2 万字的书 %d 本（%d 字）" % (len(tiny), sum(r["chars"] for r in tiny)))
    big = sorted(ok, key=lambda r: -r["chars"])[:5]
    print("  最大的 5 本: " + ", ".join("%s %.1fM" % (r["id"], r["chars"] / 1e6) for r in big))
    lowcjk = sorted(ok, key=lambda r: r["cjk_ratio"])[:5]
    print("  汉字占比最低: " + ", ".join("%s %.2f" % (r["id"], r["cjk_ratio"]) for r in lowcjk))


if __name__ == "__main__":
    main()
