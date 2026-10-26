# HLS 参数说明

本文说明当前 HLS 的运行参数、默认值、作用和覆盖方式。参数分为两套口径：

1. **当前 SmartLab 批量配置**：通过 `run_smartlab_hls.sh` 运行 Brandimarte、Barnes 和 Dauzere 时使用，是当前重复实验的推荐配置。
2. **C++ 直接运行默认值**：直接执行 `build/production_scheduling` 且省略位置参数时使用，不等同于当前批量配置。

实验结果应以对应运行目录中的 `config.json` 为最终依据。代码中的默认值只能说明新建批次时的初始配置，不能反推历史批次实际使用的参数。

## 1. 当前推荐运行入口

```bash
cd /home/luopw24/HLS && ./run_smartlab_hls.sh
```

默认测试 SmartLab 的三个数据集：

- Brandimarte：10 个 instance；
- Barnes：21 个 instance；
- Dauzere：18 个 instance；
- 每个 instance 运行 10 次，共 490 个独立任务；
- 每个任务使用独立随机 seed；
- 优先使用 `build/optimized/production_scheduling`，不存在时回退到 `build/production_scheduling`。

批量参数定义在 `run_smartlab_hls.py::parse_env_parameters()`，搜索机制开关定义在同一文件的 `EXPERIMENT_ENV`。

## 2. 核心搜索参数

| 参数/环境变量 | C++ 直接运行默认值 | SmartLab 批量默认值 | 含义 |
| --- | ---: | ---: | --- |
| `SEARCH_MODE` / `search_mode_param` | `0.5` | `0.60` | Greedy 之外的 GA/Tabu 初始偏好。当前使用自适应方法选择器，因此它不是运行期间固定不变的选择概率。 |
| `RANDOM_STRATEGY` / `random_search_strategy_param` | `0.5` | `0.50` | 在 GA/Tabu 初始偏好中分配给 GA 的比例，其余分给 Tabu。 |
| `MAX_REPEAT` / `max_repeat_count` | `100` | `1000` | 所有当前可用搜索方法完成一轮后仍无全局改进的轮数上限。达到该阈值可提前停止。 |
| `MAX_ITER` / `max_iter_count` | `30` | `45` | 单次局部搜索调用的基准迭代数。强化阶段、剩余时间不足等场景会自动缩短实际迭代数。 |
| `POPULATION_SIZE` / `population_size` | `20` | `32` | GA 的目标种群大小。GA 会跨多次调用保留种群和适应度缓存。 |
| `TABU_LENGTH` / `tabu_list_length` | `25` | `30` | Tabu 动态禁忌期限的配置基值；实际期限还会结合工序数量和停滞状态随机调整。 |
| `COPT_TIME` / `copt_time_limit_seconds` | `300` 秒 | `0` 秒 | 初始 COPT 解的时间预算。批量配置设为 0，表示跳过初始 COPT 求解。 |
| `TOTAL_TIME` / `total_time_limit_seconds` | `3600` 秒 | `3600` 秒 | 单次任务总时间上限，包括初始解生成和全部 HLS 搜索。 |
| `random_seed` | 当前 Unix 时间秒数 | 每任务随机 uint32 | HLS 根 seed；自适应控制器、GA、Tabu、Guided 和 RL 采样使用由它派生的独立随机流。 |

停止条件由总时间、`MAX_REPEAT` 和算法内部停止条件共同决定，先满足者生效。`COPT_TIME` 包含在 `TOTAL_TIME` 中，并不是额外时间。

## 3. C++ 命令行参数

直接运行二进制时，位置参数顺序如下：

```text
./build/optimized/production_scheduling input_file search_mode random_strategy \
  max_repeat max_iter population_size tabu_length copt_seconds total_seconds random_seed
```

例如使用当前 SmartLab 核心数值运行一个实例：

```bash
./build/optimized/production_scheduling \
  datasets/SmartLab_Challenge_2020/fjsp.brandimarte.Mk03.m8j15c5.txt \
  0.60 0.50 1000 45 32 30 0 3600 100003
```

直接运行 C++ 程序时不会自动获得 SmartLab 批量入口设置的全部 `HLS_*` 机制开关。仅传入上述数值参数，算法行为仍可能与批量实验不同。

## 4. RL 邻域筛选参数

当前批量配置同时在 Greedy 和 Tabu 中启用 RL。两者默认使用同一个已训练 Actor：

```text
rl_greedy/models/greedy_policy_brandimarte.txt
```

| 批量变量 | 对应 C++ 环境变量 | C++ 默认值 | 批量默认值 | 含义 |
| --- | --- | ---: | ---: | --- |
| `GREEDY_RL_TOP_K` | `HLS_GREEDY_RL_TOP_K` | `96` | `256` | Greedy 中优先精确评价的候选数量。 |
| `TABU_RL_TOP_K` | `HLS_TABU_RL_TOP_K` | `256` | `1024` | Tabu 中优先精确评价的候选数量。 |
| `GREEDY_RL_RULE_FRACTION` | `HLS_GREEDY_RL_RULE_FRACTION` | `0.25` | `0.30` | Greedy shortlist 中由规则排序保留的比例。 |
| `GREEDY_RL_EXPLORATION` | `HLS_GREEDY_RL_EXPLORATION` | `0.15` | `0.20` | Greedy shortlist 中探索候选的比例。 |
| `TABU_RL_RULE_FRACTION` | `HLS_TABU_RL_RULE_FRACTION` | `0.30` | `0.30` | Tabu shortlist 中由规则排序保留的比例。 |
| `TABU_RL_EXPLORATION` | `HLS_TABU_RL_EXPLORATION` | `0.20` | `0.25` | Tabu shortlist 中探索候选的比例。 |
| `TABU_RL_MIN_PER_NEIGHBORHOOD` | `HLS_TABU_RL_MIN_PER_NEIGHBORHOOD` | `16` | `24` | Tabu 为每类非空邻域保留的最少候选数。 |
| `TABU_RL_FALLBACK_ON_NON_IMPROVEMENT` | `HLS_TABU_RL_FALLBACK_ON_NON_IMPROVEMENT` | `0` | `0` | 是否因 shortlist 相对当前轨迹未改善而额外全扫。即使为 0，未改善方法历史最好值时仍保留必要补评。 |
| `GUIDED_TOP_K` | `HLS_GUIDED_TOP_K` | `48` | `96` | Guided 跨机插入搜索每轮优先精确评价的候选数量。 |

其他当前批量 RL 设置：

| C++ 环境变量 | 批量值 | 说明 |
| --- | ---: | --- |
| `HLS_GREEDY_RL` | `1` | 启用 Greedy RL 候选筛选。 |
| `HLS_TABU_RL` | `1` | 启用 Tabu RL 候选筛选。 |
| `HLS_GREEDY_RL_DYNAMIC_TOP_K` | `0` | 使用固定 Greedy Top-K。 |
| `HLS_TABU_RL_DYNAMIC_TOP_K` | `0` | 使用固定 Tabu Top-K。 |
| `HLS_GREEDY_RL_TEMPERATURE` | `1` | Greedy Actor 分数温度。 |
| `HLS_TABU_RL_TEMPERATURE` | `1` | Tabu Actor 分数温度。 |
| `HLS_GREEDY_RL_FALLBACK_ON_MISS` | `1` | 模型 shortlist 漏掉改进时允许规则回退。 |
| `HLS_TABU_RL_FALLBACK_ON_MISS` | `1` | Tabu 模型 shortlist 漏掉改进时允许规则回退。 |
| `HLS_GREEDY_RL_VERIFY_WITH_BASELINE` | `0` | 关闭每次调用与完整基线的开发期对照。 |
| `HLS_TABU_RL_VERIFY_FULL_NEIGHBORHOOD` | `0` | 关闭每次调用的全邻域开发期验证。 |

RL 只负责候选排序和筛选，不直接预测最终 makespan，不替代精确调度图评价，也不负责外层 Greedy/GA/Tabu/Guided 方法选择。

## 5. 当前优化机制开关

以下值由 `run_smartlab_hls.py::EXPERIMENT_ENV` 和批量环境构造逻辑设置：

| C++ 环境变量 | 批量值 | 作用 |
| --- | ---: | --- |
| `HLS_ENABLE_GREEDY` | `1` | 允许自适应控制器选择 Greedy；设为 0 用于模块消融。 |
| `HLS_ENABLE_GENETIC` | `1` | 允许自适应控制器选择 GA；设为 0 用于模块消融。 |
| `HLS_ENABLE_TABU` | `1` | 允许自适应控制器选择 Tabu；设为 0 用于模块消融。 |
| `HLS_ENABLE_GUIDED_SEARCH` | `force` | 允许所有规模的实例使用 Guided 跨机插入搜索。 |
| `HLS_ADAPTIVE_METHOD_SELECTION` | `1` | 使用在线收益评分选方法；设为 0 时按固定初始权重随机选择启用方法。 |
| `HLS_DISPATCH_INITIAL` | `1` | 启用动态分派初始解。 |
| `HLS_GREEDY_FLEXIBLE_REPAIR` | `1` | 启用 Greedy 柔性联合修复。 |
| `HLS_GREEDY_REPAIR_RELAYS` | `1` | 启用 Greedy relay 修复。 |
| `HLS_GREEDY_RELEASE_COUNT` | `32` | 扩展修复释放的候选工序数量。 |
| `HLS_GREEDY_REPAIR_SECONDS` | `8` | 扩展修复单次时间预算。 |
| `HLS_GREEDY_ROUTE_REPAIR` | `1` | 对匹配结构的实例启用路线片段修复。 |
| `HLS_ROUTE_AWARE_METHOD_SELECTION` | `1` | 结构匹配时调整 Greedy/Guided 的方法选择偏好。 |
| `HLS_GREEDY_COPT_REPAIR` | `0` | 关闭旧的 Greedy COPT 修复入口。 |
| `HLS_GA_WORKLOAD_REDISTRIBUTION` | `1` | 启用 GA 工作量重分配变异。 |
| `HLS_GA_ACTIVE_DECODE` | `0` | 关闭 GA 主动解码模式。 |
| `HLS_GA_LOCAL_RESTART` | `0` | 关闭 GA 自身的局部重启，由统一阶段重启负责。 |
| `HLS_TABU_CONTINUOUS_TRAJECTORY` | `1` | 跨 Tabu 调用保留轨迹、停滞和扰动进度。 |
| `HLS_ENABLE_RESTARTS` | `1` | 启用统一停滞重启。 |
| `HLS_RESTART_STAGNATION_CALLS` | `16` | 连续 16 次方法调用无全局改进后触发阶段重启。 |
| `HLS_TARGET_MAKESPAN` | `0` | 不使用实例专属目标值提前停止。 |

当前配置不按 instance 名称设置单独参数或固定 seed。算法可以根据实例结构和运行状态走不同分支，但配置值对所有实例一致。

## 6. 批量运行和资源参数

| 参数/环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `RUNS` / `--runs` | `10` | 每个 instance 启动的独立运行次数。 |
| `MAX_PARALLEL` / `--max-parallel` | `14` | 动态调度允许的最大并发任务数，不保证始终达到该值。 |
| `CPUS_PER_JOB` / `--cpus-per-job` | `1` | 每个 HLS 任务绑定的逻辑 CPU 数。 |
| `CPU_USAGE_LIMIT` / `--cpu-usage-limit` | `0.95` | 调度器允许使用的 CPU 配额比例。 |
| `CPU_RESERVE` / `--reserve-cpus` | `0` | 额外保留的 CPU 核数。 |
| `JOB_MEMORY_GB` / `--job-memory-gb` | `2` GiB | 尚无足够实测值时，每个新任务预留的内存。 |
| `MEMORY_RESERVE_GB` / `--reserve-memory-gb` | `4` GiB | 不分配给新任务的内存下限。 |
| `CHECK_INTERVAL` / `--check-interval` | `5` 秒 | CPU、内存和任务状态检查周期。 |
| `RAMP_UP` / `--ramp-up` | `2` | 每个检查周期最多增加的并发任务数。 |
| `TIMEOUT_GRACE` / `--timeout-grace` | `120` 秒 | 超过算法总时限后允许的退出宽限时间。 |

调度器会读取 CPU affinity、cgroup CPU 配额、cgroup 内存上限、实时 CPU 使用率和任务 RSS，因此实际并发数可能小于 `MAX_PARALLEL`。并行单位是独立的 instance/run 进程，不是单条 HLS 搜索轨迹内部并行。

## 7. 参数覆盖规则

### 7.1 覆盖核心数值和资源参数

在启动命令前设置批量入口支持的变量：

```bash
RUNS=10 MAX_PARALLEL=10 TOTAL_TIME=1800 MAX_ITER=40 \
  ./run_smartlab_hls.sh --families brandimarte barnes
```

`run_smartlab_hls.py` 使用 `os.environ.setdefault()` 填入默认值，因此调用者预先设置的 `SEARCH_MODE`、`MAX_ITER`、`TOTAL_TIME`、RL Top-K 和资源变量会生效。

### 7.2 覆盖算法开关

SmartLab Runner 会清理父进程继承的普通 `HLS_*` 变量，再设置当前固定方案。不能仅通过下面这种方式可靠覆盖：

```bash
HLS_ENABLE_RESTARTS=0 ./run_smartlab_hls.sh
```

审计或消融实验需要使用 `SMARTLAB_HLS_ENV` JSON：

```bash
SMARTLAB_HLS_ENV='{"HLS_ENABLE_RESTARTS":"0","HLS_TABU_RL":"0"}' \
  ./run_smartlab_hls.sh --families barnes
```

正式重复实验应尽量保留统一默认配置，避免把不同开关、时间预算或程序版本的运行拼接到同一组十次结果中。

当前核心模块消融实验只在 Brandimarte 的 Mk01～Mk10 上执行。`run_core_ablation.sh` 默认不创建 Barnes 或 Dauzere 任务；11 个默认变体各使用相同的 100 个配对 seed 任务。

## 8. 随机种子

- 新批次使用 `secrets.randbits(32)` 为每个任务生成独立 seed；
- 同一批次若随机生成重复 seed，会重新生成；
- seed 在求解前写入 `seeds.csv` 和 `state.json`；
- 默认不会根据 makespan 重抽、筛选或替换 seed；
- `--resume` 会沿用原任务 seed，不会生成新 seed；
- 配对实验可通过 `PAIRED_SEEDS_FILE` 提供完整 seed 表。

`RUNS=10` 表示创建 10 个独立进程和 10 个 seed，不是让同一个进程中的局部搜索循环执行 10 次。

## 9. 输出和参数审计

每次 SmartLab 批次会创建独立运行目录：

```text
run_logs/smartlab_<运行时间>/
```

关键文件：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 本批次参数、算法开关、调度配置、程序/模型/数据快照和 SHA-256。 |
| `seeds.csv` | 每个 family、instance、run 对应的随机 seed。 |
| `state.json` | 每个任务状态、PID、makespan、耗时和失败原因。 |
| `scheduler.log` / `scheduler.csv` | 动态并发和资源使用记录。 |
| `processes_10min.csv` | 每 10 分钟记录活动进程的 CPU、RSS 和绑定核。 |
| `HLS_<family>.csv` | 对应数据集的 `instance,run1,...,run10` 结果。 |

复现实验时应优先读取具体运行目录的 `config.json`，并验证其中程序、模型和数据快照的 SHA-256。根目录中的 `HLS_brandimarte.csv`、`HLS_barnes.csv` 和 `HLS_dauzere.csv` 只保存汇总结果，不足以单独证明运行参数和代码版本。

## 10. 相关实现文件

- `run_smartlab_hls.py`：当前 SmartLab 批量默认参数、算法开关和输出管理；
- `run_smartlab_hls.sh`：Python 版本检查和批量启动入口；
- `run_adaptive_brandimarte.py`：核心参数映射、模型环境、动态资源调度和任务状态管理；
- `main.cpp`：C++ 位置参数及直接运行默认值；
- `GreedyRlPolicy.cpp` / `TabuRlPolicy.cpp`：RL 参数解析；
- `GreedySearch.cpp` / `RandomSearch.cpp` / `TabuSearch.cpp` / `GuidedSearch.cpp`：各搜索方法参数的实际使用；
- `design.md`：算法流程、预算分配和内部实现的详细说明。
