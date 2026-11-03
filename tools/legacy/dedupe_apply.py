#!/usr/bin/env python3
"""对 data_v3 做去重：删掉完全重复的书和同名近似重复的版本。

输出：
  data_v3/books/_manifest_final.jsonl  最终保留清单
  data_v3/books/_dropped.jsonl         被删清单（含原因）
  实际删除对应的 txt 文件
"""
import json, os, re, glob, hashlib, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V3 = ROOT + "/data_v3/books"
RE_JUNK = re.compile(r"[《》（）()【】\[\]·・\s\-—_0-9０-９]|完结|全本|校对|精校|番外|合集|未删|文字版|TXT|txt")


def norm_title(t):
    t = re.sub(r"(作者|著)[:：].*$", "", t)
    t = t.split("-")[0]
    return RE_JUNK.sub("", t).strip().lower()


def shingles(path, k=12, n=200000):
    with open(path, encoding="utf-8", errors="replace") as f:
        t = f.read(n)
    return set(t[i:i + k] for i in range(0, max(0, len(t) - k), 7))


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    books = []
    for mf in glob.glob(os.path.join(V3, "_manifest_*.jsonl")):
        if "_final" in mf or "_dropped" in mf:
            continue
        with open(mf, encoding="utf-8") as f:
            for line in f:
                books.append(json.loads(line))
    print("载入 %d 本" % len(books))

    drop = {}

    # 1) 完全重复：同 hash 只留第一本
    byhash = collections.defaultdict(list)
    for b in books:
        byhash[b["hash"]].append(b)
    for h, v in byhash.items():
        v.sort(key=lambda x: -x["chars"])
        for b in v[1:]:
            drop[b["id"]] = "完全重复(%s)" % v[0]["id"]

    # 2) 同名近似重复
    bytitle = collections.defaultdict(list)
    for b in books:
        if b["id"] in drop:
            continue
        bytitle[norm_title(b["title"])].append(b)
    near = 0
    for k, v in bytitle.items():
        if len(v) < 2:
            continue
        v.sort(key=lambda x: -x["chars"])
        keep_sh = None
        for b in v:
            if b["id"] in drop:
                continue
            s = shingles(os.path.join(V3, b["id"]))
            if keep_sh is None:
                keep_sh = (b, s)
                continue
            j = jacc(s, keep_sh[1])
            if j > 0.5:
                drop[b["id"]] = "近似重复 %s (jaccard %.2f)" % (keep_sh[0]["id"], j)
                near += 1
            else:
                keep_sh = (b, s)

    keep = [b for b in books if b["id"] not in drop]
    kept_chars = sum(b["chars"] for b in keep)
    drop_chars = sum(b["chars"] for b in books if b["id"] in drop)
    print("删除 %d 本（其中近似重复 %d 本），保留 %d 本，%.2f 亿字"
          % (len(drop), near, len(keep), kept_chars / 1e8))
    print("删掉 %.2f 亿字（占 %.2f%%）" % (drop_chars / 1e8, 100.0 * drop_chars / sum(b["chars"] for b in books)))

    with open(os.path.join(V3, "_manifest_final.jsonl"), "w", encoding="utf-8") as f:
        for b in keep:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(os.path.join(V3, "_dropped.jsonl"), "w", encoding="utf-8") as f:
        for b in books:
            if b["id"] in drop:
                f.write(json.dumps({"id": b["id"], "title": b["title"],
                                    "chars": b["chars"], "rev": drop[b["id"]]},
                                   ensure_ascii=False) + "\n")
    for b in books:
        if b["id"] in drop:
            p = os.path.join(V3, b["id"])
            if os.path.exists(p):
                os.remove(p)
    print("已删除文件，剩余 txt %d 个" % len(glob.glob(os.path.join(V3, "[0-9]*.txt"))))


if __name__ == "__main__":
    main()
