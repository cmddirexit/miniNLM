#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 二次净化：在已切好章的 books 上做「行内」清理，不重新解析 39GB jsonl。

清理项（prep 阶段漏掉、且只影响个别书的脏东西）：
  1. 行内 HTML 标签：`</br>“…”` → `“…”`，`<script …></script></br>` → 空行丢弃；
  2. HTML 实体：`&#8226;` `&ldquo;` → 真正的字符（prep 只处理了带字母的实体）；
  3. 行内脚本残渣：`javascript：` `document.write(` 等；
  4. 纯符号分隔行：`---------` `=====` 之类。

只删行、不新增内容；某章被清空时整章连同它的 EOS 一起丢掉，保证 EOS 数 = 章数。

用法：
    python tools/sanitize_v4.py --books data_v4/books --workers 4
    python tools/sanitize_v4.py --books data_v4/books --limit 50 --dry-run
"""
import argparse
import hashlib
import html
import json
import os
import re
import time
from multiprocessing import Pool

ROOT = ROOT + ""
EOS = "<|endoftext|>"
DRY = False

RE_TAG = re.compile(r"<[^>]{0,300}>")
RE_ENT = re.compile(r"&(?:#\d{1,7}|#x[0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,9})[;；]")
RE_ENT_NUM = re.compile(r"&#(\d{1,7})[;；]")
RE_ENT_HEX = re.compile(r"&#[xX]([0-9a-fA-F]{1,6})[;；]")
RE_ENT_NAME = re.compile(r"&([a-zA-Z][a-zA-Z0-9]{1,9})[;；]")
RE_SCRIPT = re.compile(r"javascript[:：]?|document\.write|window\.open|alert\(|on(?:click|load)=|<!--|-->", re.I)
RE_DASH = re.compile(r"^[\s\-—_=*~·・\.。，、]{6,}$")
RE_WS = re.compile(r"[ \t\u3000]{2,}")
RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
RE_LATIN = re.compile(r"[A-Za-z]")


def unescape_entities(ln):
    """半角/全角分号的实体都还原；认不出的保持原样"""
    def num(m):
        try:
            v = int(m.group(1))
        except ValueError:
            return m.group(0)
        return chr(v) if 0 < v < 0x110000 else m.group(0)

    def hx(m):
        try:
            v = int(m.group(1), 16)
        except ValueError:
            return m.group(0)
        return chr(v) if 0 < v < 0x110000 else m.group(0)

    def named(m):
        return html.unescape("&%s;" % m.group(1))

    for _ in range(3):                     # &amp;#8226； 这种二次转义要还原两轮
        prev = ln
        ln = RE_ENT_HEX.sub(hx, ln)
        ln = RE_ENT_NUM.sub(num, ln)
        ln = RE_ENT_NAME.sub(named, ln)
        if ln == prev:
            break
    return ln


def clean_line(ln, st):
    """返回清洗后的行（None = 丢弃）"""
    if "<" in ln and RE_TAG.search(ln):
        new = RE_TAG.sub("", ln)
        if new != ln:
            ln = new
            st["tags"] += 1
    if "&" in ln and RE_ENT.search(ln):
        new = unescape_entities(ln)
        if new != ln:
            ln = new
            st["ents"] += 1
    if RE_SCRIPT.search(ln):
        new = RE_SCRIPT.sub("", ln)
        if new != ln:
            ln = new
            st["scripts"] += 1
    ln = RE_WS.sub(" ", ln.strip())
    if not ln:
        return None
    if RE_DASH.match(ln):
        return None
    if not RE_CJK.search(ln) and not RE_LATIN.search(ln) and len(ln) > 12:
        return None
    return ln


def process(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    st = {"tags": 0, "ents": 0, "scripts": 0, "removed": 0}
    chapters, cur = [], []
    for ln in text.split("\n"):
        if ln.strip() == EOS:
            chapters.append(cur)
            cur = []
            continue
        if not ln.strip():
            continue
        new = clean_line(ln, st)
        if new is None:
            st["removed"] += 1
            continue
        cur.append(new)
    if cur:
        chapters.append(cur)
    n_old_ch = text.count(EOS)
    chapters = [c for c in chapters if c]          # 章节被清空 → 整章丢
    out = ["\n".join(c) for c in chapters]
    body = ("\n" + EOS + "\n").join(out) + "\n" + EOS + "\n"
    st["chapters_dropped"] = n_old_ch - len(out)
    chars = sum(len(c) for c in out)               # 不含 EOS 的纯正文字数
    st.update(changed=False, removed=st["removed"])
    if body == text:
        return {"path": path, "changed": False}
    if DRY:
        return {"path": path, **{**st, "changed": True, "chars": chars, "chapters": len(out), "hash": ""}}
    tmp = path + ".san"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
    os.replace(tmp, path)
    st["changed"] = True
    st["chars"] = chars
    st["chapters"] = len(out)
    st["hash"] = hashlib.blake2b(body.encode(), digest_size=8).hexdigest()
    return {"path": path, **st}


def _init(dry):
    global DRY
    DRY = dry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=os.path.join(ROOT, "data_v4", "books"))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    man_path = args.manifest or os.path.join(args.books, "_manifest_v4.jsonl")

    rows = [json.loads(l) for l in open(man_path, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[:args.limit]
    by_id = {r["id"]: r for r in rows}
    paths = [os.path.join(args.books, r["id"]) for r in rows]

    print(f"待净化 {len(rows)} 本，workers={args.workers}，dry_run={args.dry_run}")
    t0 = time.time()
    tot = {"changed": 0, "removed": 0, "tags": 0, "ents": 0, "scripts": 0, "chapters_dropped": 0}
    with Pool(max(1, args.workers), initializer=_init, initargs=(args.dry_run,)) as pool:
        for i, r in enumerate(pool.imap_unordered(process, paths, chunksize=2), 1):
            for k in tot:
                tot[k] += r.get(k, 0)
            if r["changed"] and not args.dry_run:
                row = by_id[os.path.basename(r["path"])]
                row["chars"] = r["chars"]
                row["chapters"] = r["chapters"]
                row["eos"] = r["chapters"]
                row["hash"] = r["hash"]
                row["sanitized"] = 1
                row["lines_removed"] = r["removed"]
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    print(f"用时 {time.time()-t0:.0f}s  {tot}")
    if args.dry_run:
        return
    rows.sort(key=lambda r: r["id"])
    with open(man_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("manifest 已更新")


if __name__ == "__main__":
    main()
