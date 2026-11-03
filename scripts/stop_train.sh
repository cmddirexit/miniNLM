#!/usr/bin/env bash
# 优雅停止训练。
#
# 训练进程是用 `setsid ... &` 起的，继承 SIG_IGN，**SIGINT / SIGTERM 都没用**
# （实测 /proc/PID/status 里 SigIgn 含 bit1）。所以用文件当信号：
#   out/STOP     -> train.py 每步检查，看到就存盘并 exit(0)
#   .stop_train  -> run_train.sh 看到就不再自动拉起（40 分钟 TTL）
#
# 用法：bash scripts/stop_train.sh
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/out}

if ! pgrep -u "$(id -u)" -f 'train\.py --data_dir' >/dev/null 2>&1; then
    echo "[stop] $(date '+%F %T') 没有在跑的 train.py"
    exit 0
fi

touch "$ROOT/.stop_train"
mkdir -p "$OUT"
touch "$OUT/STOP"
echo "[stop] $(date '+%F %T') 已放置停止标记，等训练存盘退出…"

for _ in $(seq 1 300); do
    if ! pgrep -u "$(id -u)" -f 'train\.py --data_dir' >/dev/null 2>&1; then
        echo "[stop] $(date '+%F %T') 已退出并完成存盘"
        sync
        exit 0
    fi
    sleep 2
done

echo "[stop] $(date '+%F %T') 10 分钟未退出（不建议强杀，会丢掉最近的 checkpoint）"
sync
