#!/usr/bin/env python3
"""把 HuggingFace wdndev/webnovel-chinese 的 jsonl 分片整理成按书切分的 txt。

用法：
    python ingest_hf.py             # 处理全部 10 个分片
    python ingest_hf.py 9           # 只处理第 9 个分片（测试用）
"""
import json, os, re, sys, hashlib, time
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = ROOT + "/data/hf_novels"
OUT = ROOT + "/data_v3/books"
SHARDS = ["webnovel_%d.jsonl" % i for i in range(10)]

RE_AD = re.compile(
    r"https?://|www\.|\.com|笔趣阁|顶点小说|书包网|txt下载|一秒记住|请收藏"
    r"|求月票|求推荐票|作者有话|作者的话|PS[:：]|本章未完|点击下一页"
)
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff]")
RE_WS = re.compile(r"[ \t\u3000]+")


def clean(t):
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = RE_CTRL.sub("", t)
    out = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        if RE_AD.search(line):
            continue
        out.append(line)
    return "\n".join(out)


def process(shard):
    path = os.path.join(SRC, shard)
    sid = shard.split("_")[1].split(".")[0]
    t0 = time.time()
    n_lines = n_bad = n_ch = n_drop = 0
    total_chars = 0
    idx = 0
    cur_title = None
    buf = []
    seen = set()
    manifest = []
    os.makedirs(OUT, exist_ok=True)

    def flush():
        nonlocal idx, total_chars, cur_title
        if cur_title is None or not buf:
            return
        text = "\n".join(buf)
        if len(text) < 3000:      # 整本太短的丢弃（短篇碎片）
            return
        name = "%s_%05d.txt" % (sid, idx)
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(text)
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        manifest.append({"id": name, "title": cur_title, "shard": shard,
                         "chars": len(text), "chapters": len(buf), "hash": h})
        total_chars += len(text)
        idx += 1

    with open(path, "rb") as f:
        for raw in f:
            n_lines += 1
            try:
                d = json.loads(raw)
            except Exception:
                n_bad += 1
                continue
            title = (d.get("title") or "").strip()
            text = d.get("text") or ""
            if title != cur_title:
                flush()
                cur_title = title
                buf = []
                seen = set()
            c = clean(text)
            if len(c) < 100:
                n_drop += 1
                continue
            hh = hashlib.blake2b(c.encode("utf-8"), digest_size=8).digest()
            if hh in seen:
                n_drop += 1
                continue
            seen.add(hh)
            buf.append(c)
            n_ch += 1
    flush()

    with open(os.path.join(OUT, "_manifest_%s.jsonl" % sid), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return {"shard": shard, "lines": n_lines, "bad": n_bad, "chapters": n_ch,
            "dropped": n_drop, "books": idx, "chars": total_chars,
            "sec": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    targets = SHARDS if len(sys.argv) < 2 else ["webnovel_%s.jsonl" % sys.argv[1]]
    t0 = time.time()
    tot = {"lines": 0, "bad": 0, "chapters": 0, "dropped": 0, "books": 0, "chars": 0}
    with Pool(min(4, len(targets))) as pool:
        for r in pool.imap_unordered(process, targets):
            print("[%s] 行 %d  章节 %d  丢弃 %d  书 %d  %.2f 亿字  %.0fs"
                  % (r["shard"], r["lines"], r["chapters"], r["dropped"],
                     r["books"], r["chars"] / 1e8, r["sec"]), flush=True)
            for k in tot:
                tot[k] += r[k]
    print("\n=== 合计 ===")
    print("行数 %d，损坏 %d" % (tot["lines"], tot["bad"]))
    print("章节 %d，丢弃 %d" % (tot["chapters"], tot["dropped"]))
    print("书 %d，总字数 %.2f 亿" % (tot["books"], tot["chars"] / 1e8))
    print("用时 %.0f 分钟" % ((time.time() - t0) / 60))
