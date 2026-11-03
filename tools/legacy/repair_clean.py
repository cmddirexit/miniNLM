# -*- coding: utf-8 -*-
"""就地清理 data/clean 里的 HTML 残留（<p> </p> <br> &nbsp; 等）

用法:
    python tools/repair_clean.py            # 只统计，不改文件
    python tools/repair_clean.py --apply    # 真正改写
"""
import argparse, glob, html, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,9}(?:\s[^<>]{0,80})?/?>")
ENT = re.compile(r"&#?\w{1,8};")


def repair(text):
    n0 = len(text)
    for _ in range(3):                  # 反转义可能生成新的标签，反复剥几轮
        new = html.unescape(TAG.sub("", text))
        if new == text:
            break
        text = new
    text = re.sub(r"&[A-Za-z0-9#]{1,10};", "", text)   # 残缺实体 &1t; &quouot; &q;
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, n0 - len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=ROOT + "/data/clean")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.txt")))
    rows, total_cut = [], 0
    for f in files:
        s = open(f, encoding="utf-8").read()
        if not TAG.search(s) and not ENT.search(s):
            continue
        t, cut = repair(s)
        hits = len(TAG.findall(s)) + len(ENT.findall(s))
        rows.append((hits, cut, f))
        total_cut += cut
        if a.apply and cut > 0:
            open(f, "w", encoding="utf-8").write(t)

    rows.sort(reverse=True)
    print(f"受影响 {len(rows)} 本 / 共 {len(files)} 本，删除字符 {total_cut:,}")
    for hits, cut, f in rows[:10]:
        print(f"  {hits:7d} 处  -{cut:8,d} 字  {os.path.basename(f)}")
    if not a.apply:
        print("\n（未改动；加 --apply 执行）")
    else:
        print("\n已就地改写")
        for _, _, f in rows:
            s = open(f, encoding="utf-8").read()
            if TAG.search(s) or ENT.search(s) or re.search(r"&[A-Za-z0-9#]{1,10};", s):
                print("  !! 仍有残留:", f)


if __name__ == "__main__":
    main()
