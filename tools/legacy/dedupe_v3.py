#!/usr/bin/env python3
"""统计 data_v3 语料，并和本地 data/clean 语料做重复检查。"""
import json, os, re, glob, hashlib, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V3 = ROOT + "/data_v3/books"
CLEAN = ROOT + "/data/clean"

RE_JUNK = re.compile(r"[《》（）()【】\[\]·・\s\-—_0-9０-９]|完结|全本|校对|精校|番外|合集|未删|文字版|TXT|txt")


def norm_title(t):
    t = re.sub(r"(作者|著)[:：].*$", "", t)
    t = t.split("-")[0]
    t = RE_JUNK.sub("", t)
    return t.strip().lower()


def load_v3():
    books = []
    for mf in glob.glob(os.path.join(V3, "_manifest_*.jsonl")):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                books.append(json.loads(line))
    return books


def load_clean():
    out = []
    mf = os.path.join(CLEAN, "manifest.jsonl")
    if os.path.exists(mf):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                d["_name"] = d.get("file") or d.get("id")
                out.append(d)
        return out
    for p in glob.glob(os.path.join(CLEAN, "*.txt")):
        out.append({"_name": os.path.basename(p), "chars": os.path.getsize(p)})
    return out


def shingles(text, k=12, n=400):
    t = text[:200000]
    return set(t[i:i + k] for i in range(0, max(0, len(t) - k), 7)) or {""}


def jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    v3 = load_v3()
    cl = load_clean()
    print("data_v3：%d 本书，%.2f 亿字" % (len(v3), sum(b["chars"] for b in v3) / 1e8))
    print("data/clean：%d 本，%.2f 亿字" % (len(cl), sum(b.get("chars", 0) for b in cl) / 1e8))

    # 1) v3 内部：完全相同
    byhash = collections.defaultdict(list)
    for b in v3:
        byhash[b["hash"]].append(b)
    ex = [v for v in byhash.values() if len(v) > 1]
    print("\n[1] v3 内部完全相同的书：%d 组，涉及 %d 本，浪费 %.2f 千万字"
          % (len(ex), sum(len(v) for v in ex), sum(v[0]["chars"] for v in ex) * (1 - 1 / 2) / 1e7))

    # 2) v3 内部：同名不同文
    bytitle = collections.defaultdict(list)
    for b in v3:
        bytitle[norm_title(b["title"])].append(b)
    same = {k: v for k, v in bytitle.items() if len(v) > 1}
    print("[2] v3 内部同名（归一化后）：%d 组" % len(same))
    for k, v in list(same.items())[:8]:
        print("    %-22s %d 本，字数 %s" % (k, len(v), [b["chars"] for b in v]))

    # 3) v3 vs clean：书名交集
    cl_titles = {}
    for c in cl:
        nm = c["_name"].rsplit(".", 1)[0]
        nm = re.sub(r"^\d+_", "", nm)
        cl_titles.setdefault(norm_title(nm), []).append(c)
    hit = [(k, bytitle[k], cl_titles[k]) for k in bytitle if k in cl_titles]
    print("\n[3] 与本地 529 本书名相同的：%d 组" % len(hit))

    # 对交集做内容相似度
    strong, weak, distinct = [], [], []
    for k, a, b in hit:
        ta = open(os.path.join(V3, a[0]["id"]), encoding="utf-8", errors="replace").read(200000)
        sa = shingles(ta)
        best = 0.0
        for c in b:
            p = os.path.join(CLEAN, c["_name"])
            if not os.path.exists(p):
                continue
            try:
                tb = open(p, encoding="utf-8", errors="replace").read(200000)
            except Exception:
                continue
            best = max(best, jacc(sa, shingles(tb)))
        if best > 0.5:
            strong.append((k, best))
        elif best > 0.15:
            weak.append((k, best))
        else:
            distinct.append((k, best))
    print("    内容高度重复(>0.5)：%d 本" % len(strong))
    print("    部分重复(0.15-0.5)：%d 本" % len(weak))
    print("    其实不同(<0.15)：%d 本" % len(distinct))
    if strong:
        print("    高度重复示例：", [(k, round(s, 2)) for k, s in strong[:10]])
    if weak:
        print("    部分重复示例：", [(k, round(s, 2)) for k, s in weak[:10]])

    # 4) 章节数分布
    ch = sorted(b["chapters"] for b in v3)
    n = len(ch)
    print("\n[4] 每本章节数：中位 %d，25%% %d，75%% %d，最大 %d"
          % (ch[n // 2], ch[n // 4], ch[3 * n // 4], ch[-1]))
    big = sum(1 for x in ch if x >= 50)
    print("    >=50 章的书：%d 本（占字数 %.1f%%）"
          % (big, 100.0 * sum(b["chars"] for b in v3 if b["chapters"] >= 50)
             / sum(b["chars"] for b in v3)))


if __name__ == "__main__":
    main()
