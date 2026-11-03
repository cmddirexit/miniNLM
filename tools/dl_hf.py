#!/usr/bin/env python3
"""把 HF wdndev/webnovel-chinese 的 10 个分片下载到 data/hf_novels_v4/，支持断点续传。

每个分片一行 = 一章（jsonl），这是恢复章节边界的关键。
"""
import os
import sys
import time
import urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEST = ROOT + "/data/hf_novels_v4"
BASE = "https://huggingface.co/datasets/wdndev/webnovel-chinese/resolve/main/data"
EXPECTED = {
    "webnovel_0.jsonl": 4183.4e6,
    "webnovel_1.jsonl": 4198.3e6,
    "webnovel_2.jsonl": 4191.2e6,
    "webnovel_3.jsonl": 4191.4e6,
    "webnovel_4.jsonl": 4190.9e6,
    "webnovel_5.jsonl": 4187.3e6,
    "webnovel_6.jsonl": 4187.3e6,
    "webnovel_7.jsonl": 4206.2e6,
    "webnovel_8.jsonl": 4219.1e6,
    "webnovel_9.jsonl": 1235.0e6,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(name):
    url = f"{BASE}/{name}"
    final = os.path.join(DEST, name)
    part = final + ".part"
    total = EXPECTED[name]
    if os.path.exists(final) and os.path.getsize(final) >= total * 0.999:
        log(f"{name} 已完整，跳过")
        return True
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have >= total * 0.999:
        os.replace(part, final)
        log(f"{name} .part 已完整，改名完成")
        return True
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    n0 = have
    last = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                have += len(b)
                mb = (have - n0) / 1e6
                if mb - last >= 500:
                    last = mb
                    speed = mb / max(time.time() - t0, 1e-6)
                    log(f"  {name}: {have/1e9:.2f}/{total/1e9:.2f} GB  {speed:.1f} MB/s")
    except Exception as e:
        log(f"{name} 中断: {type(e).__name__}: {e}（保留 .part 可续传）")
        return False
    if have >= total * 0.999:
        os.replace(part, final)
        log(f"{name} 完成 {have/1e9:.2f} GB")
        return True
    log(f"{name} 不完整: {have/1e6:.0f} MB / {total/1e6:.0f} MB")
    return False


def main():
    os.makedirs(DEST, exist_ok=True)
    order = sorted(EXPECTED, key=lambda s: int(s.split("_")[1].split(".")[0]))
    if len(sys.argv) > 1:
        order = [n for n in order if n in sys.argv[1:]]
    for (i, name) in enumerate(order, 1):
        for attempt in range(6):
            log(f"== [{i}/{len(order)}] {name} 第 {attempt + 1} 次")
            if fetch(name):
                break
            time.sleep(5)
    log("全部结束")
    for name in order:
        p = os.path.join(DEST, name)
        log(f"  {name}: {os.path.getsize(p)/1e9:.2f} GB" if os.path.exists(p) else f"  {name}: 缺失")


if __name__ == "__main__":
    main()
