#!/usr/bin/env python3
"""同名书版本重复检测：用 20-gram 包含率代替 jaccard。

同名的两本常是同一本书的不同版本（删改/和谐/补全），
它们中间大段文字一致但不是逐字相同，jaccard 会偏低，包含率更准。
"""
import json, os, re, glob, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V3 = ROOT + "/data_v3/books"
RE_JUNK = re.compile(r"[《》（）()【】\[\]·・\s\-—_0-9０-９]|完结|全本|校对|精校|番外|合集|未删|文字版|TXT|txt")
K = 20
STEP = 200


def norm_title(t):
    t = re.sub(r"(作者|著)[:：].*$", "", t)
    t = t.split("-")[0]
    return RE_JUNK.sub("", t).strip().lower()


def grams(text, step, k=K):
    return set(hash(text[i:i + k]) for i in range(0, max(0, len(text) - k), step))


def containment(a_grams, b_text):
    """b 有多大比例的 20-gram 出现在 a 里"""
    bs = grams(b_text, STEP)
    if not bs:
        return 0.0
    hit = sum(1 for h in bs if h in a_grams)
    return hit / len(bs)


def main():
    books = [json.loads(l) for l in open(os.path.join(V3, "_manifest_final.jsonl"), encoding="utf-8")]
    by = collections.defaultdict(list)
    for b in books:
        by[norm_title(b["title"])].append(b)
    groups = [v for v in by.values() if len(v) > 1]
    print("同名组 %d 个" % len(groups))

    drop = {}
    for v in groups:
        v.sort(key=lambda x: -x["chars"])
        kept = []   # [(book, text)]
        for b in v:
            p = os.path.join(V3, b["id"])
            t = open(p, encoding="utf-8", errors="replace").read()
            dup_of = None
            for kb, kt, kg in kept:
                c = containment(kg, t)
                if c > 0.5:
                    dup_of = (kb, c)
                    break
            if dup_of:
                drop[b["id"]] = (dup_of[0]["id"], dup_of[1], b["title"])
            else:
                kept.append((b, t, grams(t, 1)))

    print("\n判定为版本重复：%d 本" % len(drop))
    for bid, (kid, c, title) in sorted(drop.items(), key=lambda x: -x[1][1])[:20]:
        kb = next(b for b in books if b["id"] == kid)
        print("  %-16s %10d → 保留 %s %10d  (包含率 %.2f)" % (bid, next(b for b in books if b["id"] == bid)["chars"], kid, kb["chars"], c))

    saved = sum(next(b for b in books if b["id"] == bid)["chars"] for bid in drop)
    print("\n删除后节省 %.2f 亿字" % (saved / 1e8))

    if drop:
        for bid in drop:
            p = os.path.join(V3, bid)
            if os.path.exists(p):
                os.remove(p)
        keep = [b for b in books if b["id"] not in drop]
        with open(os.path.join(V3, "_manifest_final.jsonl"), "w", encoding="utf-8") as f:
            for b in keep:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
        with open(os.path.join(V3, "_dropped_ver.jsonl"), "w", encoding="utf-8") as f:
            for bid, (kid, c, title) in drop.items():
                f.write(json.dumps({"id": bid, "title": title, "kept": kid, "containment": round(c, 3)},
                                   ensure_ascii=False) + "\n")
        print("保留 %d 本，共 %.2f 亿字" % (len(keep), sum(b["chars"] for b in keep) / 1e8))


if __name__ == "__main__":
    main()
