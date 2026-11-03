# 无人值守训练：断电、崩溃、重启之后自己接着跑

一台会断电的机器（宿舍、实验室共享机）上跑 5 天训练，靠的就是这几个文件。
本机场景：Windows 11 + WSL2 Ubuntu 22.04 + 一块 AMD RX 9060 XT；但思路和平台无关。

## 一个关键事实：信号杀不掉训练进程

训练是用 `setsid ... &` 在后台起的，进程继承 `SIG_IGN`，所以 `kill -INT` / `Ctrl-C` / `SIGTERM`
**全都无效**（`/proc/PID/status` 的 `SigIgn` 里就有 bit1）。`kill -9` 能用，但会丢掉最近的 checkpoint。

所以优雅停止只能**用文件当信号**：

| 标记 | 谁看 | 作用 |
|---|---|---|
| `out/STOP` | `train.py` 每步检查 | 看到就存盘、写 `DONE`、`sys.exit(0)` |
| `.stop_train` | `run_train.sh` 每轮检查 | 看到就不再自动拉起，并带上 40 分钟 TTL |

## 三个脚本

### `scripts/run_train.sh` — 看护循环

```bash
TRAIN_ARGS="--preset thin --n_kv_head 2 ..." bash scripts/run_train.sh
```

- 启动前检查：已完成的 `DONE` → 退出；`.stop_train` 未过期 → 退出；已有训练进程 → 退出（避免两份显存）；
- 死后 15 秒重起，最多 500 轮；
- 把 inductor / triton 的编译缓存放进 `~/.cache`：否则每次重启都要重编 6 分钟。

### `scripts/stop_train.sh` — 优雅停止

```bash
bash scripts/stop_train.sh
```

放两个标记 → 每 2 秒轮询 → 训练自己存盘退出 → `sync`。最多等 10 分钟。

### `scripts/miniNLM-train.service` — systemd（可选）

给已经配好免密 sudo / `loginctl enable-linger` 的机器用，替代 Windows 任务计划。

## Windows 任务计划（本机实际用法）

| 任务 | 触发 | 动作 |
|---|---|---|
| `miniNLM-train` | 登录时 + 之后每 **5 分钟** | `wsl.exe -d Ubuntu-22.04 -- bash /path/to/repo/scripts/run_train.sh` |
| `miniNLM-stop` | 每天 **22:52** | `wsl.exe -d Ubuntu-22.04 -- bash /path/to/repo/scripts/stop_train.sh` |

`miniNLM-train` 每 5 分钟看一眼：训练在跑就直接退出，没在跑就拉起来 ——
它既是「开机自启」，也是「崩溃自愈」。WSL 不会随 Windows 自启，这一步必须有。

> `schtasks` 在工作目录是 `\\wsl.localhost\...` 时会报
> `The directory name is invalid`，记得 `cd C:\` 或给 `workdir` 显式指定 Windows 路径。

## 行为细节

- `--save_interval 250` ≈ 每 12–15 分钟一个 checkpoint，真断电最多丢 15 分钟。
- 停止标记只有 **40 分钟**有效期：22:52 停完不会立刻被拉起来；断电过夜回来标记早过期，
  开机后任务计划照常 `--resume auto` 续跑。
- 跑完计划步数后 `train.py` 写 `out/DONE`，看护循环看到就退出 —— 否则 `--resume auto` 会从最后一个
  `ckpt.pt` 反复重放末尾几十步，一轮 10 分钟地空转。要继续加训，先删 `out/DONE`。

## 手动操作

```bash
# 停止
bash scripts/stop_train.sh
# 续跑
rm -f .stop_train out/STOP && setsid nohup bash scripts/run_train.sh >/dev/null 2>&1 &
# 看进度
tail -f out/train.log
```

## 必须手动做的两件事（脚本改不了）

1. BIOS 里 `Restore on AC Power Loss`（停电恢复）设为 **Power On**，否则来电后机器不自己开机；
2. 开自动登录（`Win+R` → `netplwiz` → 取消「要使用本计算机，用户必须输入用户名和密码」），
   否则任务计划要等你早上登录才触发。

于是断电的一晚是：22:52 存盘退出 → 23:00 断电 → 早上来电 → Windows 自动登录 →
任务计划拉起 WSL → `--resume auto` 从最后一个 checkpoint 继续。
