#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 语料预处理：把 HF 分片（一行一章）清洗成按书切分的 txt，并保住章节边界。

和 v3 的 ingest_hf.py 相比，关键区别：
  1. 章节之间插入 `<|endoftext|>` 标记行，让打包时能真正学到「文档结束」；
  2. 清洗更严格：广告行/元信息行/版权行/纯符号行/相邻重复段落全部剔除，
     并保留 v3 已有的「书内重复章节」去重结论；
  3. 书单沿用 v3 去重后的结论（data_v3/books/_manifest_final.jsonl），
     只按 title + 出现顺序对齐，不重新跑 shingle 去重。

用法：
    python tools/prep_v4.py --shards 0 --allow-part   # 冒烟测试
    python tools/prep_v4.py                           # 全部 10 个分片
    python tools/prep_v4.py --workers 6 --limit-books 200
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
SRC = os.path.join(ROOT, "data", "hf_novels_v4")
V3_MANIFEST = os.path.join(ROOT, "data_v3", "books", "_manifest_final.jsonl")
OUT = os.path.join(ROOT, "data_v4", "books")
EOS = "<|endoftext|>"

MIN_BOOK_CHARS = 20000     # 整本不足 2 万字：碎片，丢弃
MIN_CHAPTER_CHARS = 200    # 章节不足 200 字：广告/占位章，丢弃
MIN_BOOK_CJK = 0.55        # 汉字占比过低：疑似翻译/乱码
MIN_CHAPTER_CJK = 0.40

RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff\u00ad]")
RE_INDENT = re.compile(r"^[ \t\u3000]+")
RE_WS = re.compile(r"[ \t\u3000]{2,}")
RE_CJK = re.compile(r"[\u4e00-\u9fff]")

# 广告/站点/求票：短行（<=60 字）命中即删；长行只在命中强特征时删。
# 注意：像「手打」「首发」「尽在」这种词在正文里也会出现，必须绑更长的组合。
RE_AD_SHORT = re.compile(
    r"https?://|www\.|[a-z0-9\-]+\.(com|net|cc|org|la|me)\b"
    r"|笔趣阁|顶点小说|书包网|琥珀小说|飘天文学|全本小说网|看书网|读读窝|无弹窗"
    r"|txt下载|免费下载|全集下载|电子书下载|一秒记住|请收藏|加入书签|投推荐票|求月票"
    r"|求推荐票|求订阅|求打赏|求收藏|作者有话|作者的话|PS[:：]" r"|本章未完|点击下一页"
    r"|上一章|下一章|返回目录|章节目录|全文阅读|最新章节|手机版|请记住本站|本站网址"
    r"|更新最快|首发于|首发地址|手打版|手打更新|手打小说|本书由|更多精彩.{0,6}尽在"
)
RE_AD_STRONG = re.compile(r"https?://|www\.|笔趣阁|请收藏|求月票|求推荐票|求订阅|求打赏|本章未完|点击下一页")

# 元信息（书头/书尾常见）
RE_META = re.compile(
    r"^(作者|著者|译者|责任编辑|出版社|出版|文案|简介|内容简介|书籍简介|作品简介|标签|分类|"
    r"状态|字数|总字数|更新时间|最后更新|最新章节|正文|类型|来源|声明|版权|ISBN|isbn)\s*[:：]"
)
RE_INFO = re.compile(
    r"作者\s*[:：]|书号|字数\s*[:：]|录入|校对|整理|排版|下载|转载|版权|本书由|本书为|"
    r"电子书|手机阅读|免费阅读|全文阅读|更新于|更新时间"
)
RE_BOOK_TITLE = re.compile(r"^《[^《》]{1,40}》[^\n]{0,20}$")
RE_COPYRIGHT = re.compile(r"^(版权所有|版权声明|Copyright|©|本书由|未经许可|严禁转载|转载自)")
RE_ISBN = re.compile(r"^(ISBN|isbn)\s*[:：]?\s*[0-9\-Xx ]{9,20}$|^97[89][\- ]?[0-9\- ]{9,15}$")
RE_END_MARK = re.compile(
    r"^[—\-~·*_=]{0,8}\s*[（(【\[]?\s*(全书完|完本|完结|未完待续|待续|结文|大结局|全书终|本书完)"
)
RE_ONLY_PUNCT = re.compile(r"^[\W_0-9]*$", re.UNICODE)
RE_REPEAT_PUNCT = re.compile(r"^([\W_·—…※☆★○●◎◇◆□■△▲※→←↑↓])\1{2,}$", re.UNICODE)
RE_REPEAT_CJK = re.compile(r"^([\u4e00-\u9fff])\1{5,}$")
RE_SCENE_BREAK = re.compile(r"^(…{2,}|—{2,}|\.{3,}|·{3,}|\*{3,})$")
RE_LATIN = re.compile(r"[A-Za-z0-9]")

# 独立成行的分类标签
RE_TAG = re.compile(
    r"^(种田文|爽文|系统文|无限流|都市|玄幻|仙侠|言情|穿越|重生|网游|科幻|悬疑|历史|军事|"
    r"校园|末世|女强|男强|宫斗|宅斗|甜文|虐文|宠文|灵异|修真|修仙|异界|恐怖|惊悚|轻松|搞笑|"
    r"同人|快穿|种马|后宫|热血|青春|古言|现言|耽美|百合|耽美小说|言情小说)$"
)

# 章标题（本数据集没有，但保留：万一分片里带了标题行，别当正文删）
RE_CHAPTER_HEAD = re.compile(r"^第[0-9零一二三四五六七八九十百千两]{1,10}[章节卷回集部篇].{0,40}$")


def tail_junk(p):
    n = len(p)
    if not p:
        return True
    if RE_TAG.match(p) or RE_REPEAT_CJK.match(p) or RE_REPEAT_PUNCT.match(p):
        return True
    if n <= 40 and RE_END_MARK.search(p):
        return True
    if n <= 60 and RE_INFO.search(p):
        return True
    if not RE_CJK.search(p):
        return True
    return False


def trim_tail(body, max_lines=25):
    """书尾常挂「（全书完）/分类标签/啦啦啦/站点信息」，从尾部剥掉这些段落。"""
    parts = body.split("\n")
    cut = 0
    while parts and cut < max_lines and tail_junk(parts[-1]):
        parts.pop()
        cut += 1
    return "\n".join(parts), cut


def cjk_ratio(s):
    if not s:
        return 0.0
    return len(RE_CJK.findall(s)) / len(s)


def clean_chapter(text):
    """清洗一章 -> 段落列表。只折叠「相邻」重复段落，书内跨章去重交给章节哈希。"""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = RE_CTRL.sub("", t)
    out = []
    prev2 = ("", "")
    for raw in t.split("\n"):
        line = RE_INDENT.sub("", raw).rstrip()
        line = RE_WS.sub(" ", line).strip()
        if not line:
            continue
        if "&" in line and ";" in line:
            line = html.unescape(line).strip()
            if not line:
                continue
        n = len(line)
        if (n <= 60 and RE_AD_SHORT.search(line)) or RE_AD_STRONG.search(line):
            continue
        if RE_META.search(line) or RE_ISBN.search(line) or RE_COPYRIGHT.search(line):
            continue
        if RE_TAG.match(line) or (n <= 60 and RE_INFO.search(line)):
            continue
        if n <= 40 and RE_END_MARK.search(line):
            continue
        if n <= 30 and RE_BOOK_TITLE.search(line):
            continue
        if RE_CHAPTER_HEAD.search(line):
            out.append(line)
            continue
        if RE_REPEAT_PUNCT.search(line) or RE_REPEAT_CJK.search(line):
            continue
        if not RE_CJK.search(line):
            if RE_SCENE_BREAK.search(line):
                out.append(line)
                continue
            if n > 8 and RE_ONLY_PUNCT.search(line):
                continue                      # 长串标点/数字噪声
            if n <= 12 and RE_LATIN.search(line):
                continue                      # 短英文/数字行（站点名、编号）
        if line == prev2[0] or line == prev2[1]:
            continue                          # 相邻重复段落（源站常见）
        prev2 = (line, prev2[0])
        out.append(line)
    return out


def source_path(shard, allow_part):
    final = os.path.join(SRC, shard)
    if os.path.exists(final):
        return final
    part = final + ".part"
    if allow_part and os.path.exists(part):
        return part
    return None


def load_kept(manifest_path):
    """按 shard 分组的 v3 保留书单，保持原顺序。"""
    kept = {}
    if not os.path.exists(manifest_path):
        return kept
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sid = r["id"].split("_")[0]
            kept.setdefault(sid, []).append(r)
    return kept


_G = {}


def init_worker(kept_sidecar):
    with open(kept_sidecar, encoding="utf-8") as f:
        _G["kept"] = json.load(f)


def process_shard(job):
    shard, limit_books, allow_part = job
    sid = shard.split("_")[1].split(".")[0]
    kept = _G["kept"].get(sid, [])
    path = source_path(shard, allow_part)
    st = {"shard": shard, "lines": 0, "bad": 0, "src_books": 0, "kept_books": 0,
          "missing": 0, "chapters_kept": 0, "chapters_dropped": 0, "dup_chapters": 0,
          "chars": 0, "books_dropped_short": 0, "books_dropped_cjk": 0,
          "books_empty": 0, "eos": 0, "sec": 0}
    if path is None:
        st["missing"] = len(kept)
        return {"stats": st, "manifest": []}
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    kp = 0                                   # kept 指针
    cur_title = None
    buf = []
    seen_chapters = set()
    manifest = []

    def finish(title):
        nonlocal kp
        if title is None:
            return
        if kp >= len(kept) or kept[kp]["title"] != title:
            return                            # 这本在 v3 被去重掉了
        rec = kept[kp]
        kp += 1
        if not buf:
            st["books_empty"] += 1
            return
        body = ("\n" + EOS + "\n").join(buf)
        body, trimmed = trim_tail(body)
        text = body + "\n" + EOS + "\n"
        chars = len(body)
        ratio = cjk_ratio(body)
        if chars < MIN_BOOK_CHARS:
            st["books_dropped_short"] += 1
            return
        if ratio < MIN_BOOK_CJK:
            st["books_dropped_cjk"] += 1
            return
        name = rec["id"]
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(text)
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        manifest.append({"id": name, "title": title, "shard": shard, "chars": chars,
                         "chapters": len(buf), "eos": len(buf), "hash": h,
                         "cjk": round(ratio, 4),
                         "v3_chars": rec["chars"], "v3_chapters": rec["chapters"]})
        st["kept_books"] += 1
        st["chapters_kept"] += len(buf)
        st["chars"] += chars
        st["eos"] += len(buf)

    with open(path, "rb") as f:
        for raw in f:
            st["lines"] += 1
            try:
                d = json.loads(raw)
            except Exception:
                st["bad"] += 1
                continue
            title = (d.get("title") or "").strip()
            text = d.get("text") or ""
            if title != cur_title:
                finish(cur_title)
                cur_title = title
                buf = []
                seen_chapters = set()
                st["src_books"] += 1
                if limit_books and st["src_books"] > limit_books:
                    cur_title = None
                    break
            chapter = "\n".join(clean_chapter(text))
            if len(chapter) < MIN_CHAPTER_CHARS or cjk_ratio(chapter) < MIN_CHAPTER_CJK:
                st["chapters_dropped"] += 1
                continue
            ch = hashlib.blake2b(chapter.encode("utf-8"), digest_size=8).digest()
            if ch in seen_chapters:
                st["dup_chapters"] += 1
                continue
            seen_chapters.add(ch)
            buf.append(chapter)
    if cur_title is not None:
        finish(cur_title)

    st["sec"] = round(time.time() - t0, 1)
    return {"stats": st, "manifest": manifest}


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*", default=None, help="分片号，如 0 3；默认全部")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-books", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--v3-manifest", default=V3_MANIFEST)
    ap.add_argument("--allow-part", action="store_true", help="分片还没下完也先跑（冒烟用）")
    args = ap.parse_args()

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    kept = load_kept(args.v3_manifest)
    if not kept:
        print(f"[!] 找不到 v3 书单 {args.v3_manifest}，将不做对齐（全部保留）")
    sidecar = os.path.join(OUT, "_kept_cache.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(kept, f)

    nums = args.shards if args.shards else list(range(10))
    shards = ["webnovel_%s.jsonl" % s for s in nums]
    jobs = [(s, args.limit_books, args.allow_part) for s in shards]
    print(f"分片 {len(shards)} 个，workers={args.workers}，输出 {OUT}")
    t0 = time.time()
    tot = {}
    all_manifest = []
    with Pool(max(1, min(args.workers, len(jobs))), initializer=init_worker,
              initargs=(sidecar,)) as pool:
        for r in pool.imap(process_shard, jobs):
            st = r["stats"]
            all_manifest.extend(r["manifest"])
            print("[%s] 行 %d 坏 %d 源书 %d -> 留 %d（章节 %d/丢 %d/重 %d）%.2f 亿字 %.0fs"
                  % (st["shard"], st["lines"], st["bad"], st["src_books"], st["kept_books"],
                     st["chapters_kept"], st["chapters_dropped"], st["dup_chapters"],
                     st["chars"] / 1e8, st["sec"]), flush=True)
            for k, v in st.items():
                if isinstance(v, (int, float)) and k != "sec":
                    tot[k] = tot.get(k, 0) + v

    all_manifest.sort(key=lambda r: r["id"])
    mpath = os.path.join(OUT, "_manifest_v4.jsonl")
    with open(mpath, "w", encoding="utf-8") as f:
        for r in all_manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== 合计 ===")
    print("源书 %d，留下 %d（丢短 %d / 丢低汉字 %d / 无正文 %d）"
          % (tot.get("src_books", 0), tot.get("kept_books", 0),
             tot.get("books_dropped_short", 0), tot.get("books_dropped_cjk", 0),
             tot.get("books_empty", 0)))
    print("章节 %d，丢弃 %d，重复章 %d；字数 %.2f 亿；EOS %d"
          % (tot.get("chapters_kept", 0), tot.get("chapters_dropped", 0),
             tot.get("dup_chapters", 0), tot.get("chars", 0) / 1e8, tot.get("eos", 0)))
    print("manifest -> %s（%d 行）" % (mpath, len(all_manifest)))
    print("用时 %.1f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
