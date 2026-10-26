# HLS 核心模块消融实验

`run_core_ablation.sh` 编译独立的 `build/ablation/production_scheduling`，然后按变体顺序调用 SmartLab 动态资源调度器。原有 `build/optimized/production_scheduling` 不会被覆盖。

已存在且确认与当前源码一致的消融二进制可用 `SKIP_BUILD=1` 跳过重新编译；默认仍会编译，避免误用旧程序。

不设置新增环境变量时，Greedy、GA、Tabu、Guided 和自适应选择全部保持开启，行为与原 HLS 一致。

## 默认消融变体

| 变体 | 关闭内容 |
| --- | --- |
| `full` | 不关闭模块，完整 HLS 对照组 |
| `no_greedy` | Greedy 及 Greedy RL |
| `no_genetic` | GA |
| `no_tabu` | Tabu 及 Tabu RL |
| `no_guided` | Guided Insertion |
| `no_rl` | Greedy RL 和 Tabu RL |
| `greedy_rl_only` | 仅保留 Greedy RL |
| `tabu_rl_only` | 仅保留 Tabu RL |
| `fixed_selector` | 用固定初始权重随机选择替代自适应评分 |
| `no_restarts` | 统一停滞重启 |
| `no_dispatch_initial` | 动态 Dispatch 初始解 |

扩展变体 `no_greedy_repair`、`no_tabu_continuity` 和 `no_ga_enhancement` 已定义，但不包含在默认核心实验中，可通过 `--variants` 显式选择。

所有变体使用同一份 `paired_seeds.csv`，按顺序运行，避免不同算法变体同时竞争 CPU 和内存。每个变体保存独立的程序、模型、数据和配置快照。

同一变体内部允许多个 instance/run 并行。默认最大并发为 14，每个任务绑定 1 个逻辑 CPU；调度器每 5 秒读取 cgroup CPU 配额、实时 CPU 占用、可用内存和任务 RSS，在资源充足时每轮最多增加 2 个任务，资源不足时停止启动新任务。默认 CPU 使用上限为 95%，每个待启动任务预留 2 GiB，额外保留 4 GiB 内存。

消融实验默认每 300 秒在终端和 `scheduler.log` 输出一次当前活动进程，同时追加到各变体运行目录的 `processes_10min.csv`（文件名为兼容历史结果而保留）。每条记录包括 PID、instance、run、seed、已运行秒数、累计 CPU 秒数、RSS 内存和 CPU 绑定。可使用 `--process-report-interval` 调整间隔。

## 正式实验范围

当前核心模块消融实验只在 Brandimarte 上执行：10 个 instance、每个 10 次、每次最多 3600 秒。直接运行脚本时默认 family 已固定为 `brandimarte`：

```bash
cd /home/luopw24/HLS && ./run_core_ablation.sh
```

默认 11 个核心变体，因此正式实验包含 `10 × 10 × 11 = 1100` 个独立任务。其他数据集不属于本轮消融结果，不应混入 `ablation_summary.csv`。

调整并发和资源保护参数：

```bash
cd /home/luopw24/HLS && ./run_core_ablation.sh --max-parallel 10 --cpu-usage-limit 0.90 --reserve-cpus 1 --job-memory-gb 2 --reserve-memory-gb 6
```

`--max-parallel` 只是上限，实际并发由资源调度器动态决定。不同消融变体仍然顺序执行，避免变体之间的硬件竞争破坏配对比较。

例如每 120 秒输出一次活动进程：

```bash
cd /home/luopw24/HLS && ./run_core_ablation.sh --process-report-interval 120
```

## 小规模验证

先用两个实例、两个 seed、60 秒检查全部核心变体：

```bash
cd /home/luopw24/HLS && ./run_core_ablation.sh --families brandimarte --instances Mk01 Mk02 --runs 2 --total-time 60 --max-parallel 4
```

只运行指定变体：

```bash
cd /home/luopw24/HLS && ./run_core_ablation.sh --variants full no_rl no_guided fixed_selector
```

查看计划但不运行：

```bash
cd /home/luopw24/HLS && ./run_core_ablation.sh --dry-run
```

中断后使用原目录恢复：

```bash
cd /home/luopw24/HLS && ./run_core_ablation.sh --resume --output-dir /home/luopw24/HLS/ablation_runs/core_<运行时间>
```

## 输出

```text
ablation_runs/core_<运行时间>/
├── ablation_manifest.json
├── paired_seeds.csv
├── ablation_summary.csv
├── full/
│   ├── run/
│   └── outputs/
└── <variant>/
    ├── run/
    └── outputs/
```

`ablation_summary.csv` 包含每个变体和 Brandimarte 实例的 mean、min、max、标准差，以及相对 `full` 的平均 makespan 差值。正的 `delta_vs_full` 表示关闭模块后结果变差。

## 新增控制变量

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `HLS_ENABLE_GREEDY` | `1` | 是否允许选择 Greedy |
| `HLS_ENABLE_GENETIC` | `1` | 是否允许选择 GA |
| `HLS_ENABLE_TABU` | `1` | 是否允许选择 Tabu |
| `HLS_ENABLE_GUIDED_SEARCH` | 原规则 | `0` 关闭，`force` 忽略规模门槛 |
| `HLS_ADAPTIVE_METHOD_SELECTION` | `1` | `0` 时按固定初始权重随机选择启用方法 |

至少需要启用 Greedy、GA、Tabu 中的一种。新增变量不设置时不会改变原有 HLS 行为。
