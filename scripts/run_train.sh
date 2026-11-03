#!/usr/bin/env bash
# 训练看护循环：train.py 退出后 15 秒重起（--resume auto 接着跑），最多 500 轮。
#   - out/DONE 存在      -> 计划步数已跑完，直接退出（否则 resume 会重放末尾几十步空转）
#   - .stop_train 未过期 -> 明确停止过，不拉起（TTL 默认 2400s，断电过夜回来标记自动失效）
#   - 已有 train.py 在跑 -> 不重复起（避免两份显存）
#
# 手动启动：rm -f .stop_train && setsid nohup bash scripts/run_train.sh >/dev/null 2>&1 &
# 换配置：   TRAIN_ARGS="--data_dir data_v4 --preset base --batch_size 8 ..." bash scripts/run_train.sh
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}                 # 例：PY=/path/to/conda/envs/rocm/bin/python
OUT=${OUT:-$ROOT/out}
LOG=${LOG:-$OUT/train.log}
FLAG=$ROOT/.stop_train
FLAG_TTL=${FLAG_TTL:-2400}        # 停止标记有效期：40 分钟内不自动重启

ARGS=${TRAIN_ARGS:---data_dir "$ROOT/data_v4" --out_dir "$OUT" --preset thin \
    --n_kv_head 2 --qk_norm 1 --z_loss 1e-4 --warmup 500 \
    --batch_size 4 --grad_accum 6 --eval_interval 1000 --eval_batches 200 \
    --save_interval 250 --resume auto}

cd "$ROOT" || exit 1
mkdir -p "$OUT" "$(dirname "$LOG")"

if [ -f "$FLAG" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$FLAG") ))
    if [ "$age" -lt "$FLAG_TTL" ]; then
        echo "[wrap] $(date '+%F %T') 停止标记还有效（${age}s < ${FLAG_TTL}s），不启动"
        exit 0
    fi
    echo "[wrap] $(date '+%F %T') 停止标记已过期（${age}s），清掉并继续" >>"$LOG"
    rm -f "$FLAG"
fi

if [ -f "$OUT/DONE" ]; then
    echo "[wrap] $(date '+%F %T') 训练已完成（$(cat "$OUT/DONE")），不再重启" >>"$LOG"
    exit 0
fi

if pgrep -u "$(id -u)" -f 'train\.py --data_dir' >/dev/null 2>&1; then
    exit 0
fi

# 编译缓存放持久目录：/tmp 重启会被清空，放这里能省掉每次开机后约 6 分钟的重新编译
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export PYTHONUNBUFFERED=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

for i in $(seq 1 500); do
    if [ -f "$OUT/DONE" ]; then
        echo "[stop] $(date '+%F %T') 训练已完成（$(cat "$OUT/DONE")），退出循环" >>"$LOG"
        break
    fi
    if [ -f "$FLAG" ]; then
        echo "[stop] $(date '+%F %T') 看到停止标记，不再重启" >>"$LOG"
        break
    fi
    rm -f "$OUT/STOP"            # 清掉上一轮优雅停止留下的标记，否则刚启动就退出
    echo "==== [wrap] $(date '+%F %T') 第 $i 轮启动（resume auto）====" >>"$LOG"
    # shellcheck disable=SC2086  # ARGS 故意按空白拆开
    "$PY" -u train.py $ARGS >>"$LOG" 2>&1
    rc=$?
    echo "==== [wrap] $(date '+%F %T') 退出 rc=$rc ====" >>"$LOG"
    [ -f "$FLAG" ] && break
    sleep 15
done
