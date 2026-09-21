# HLS-FJSP

HLS-FJSP 是一个使用 C++ 实现的柔性作业车间调度问题（FJSP）混合搜索程序。当前版本先调用 COPT 为实例生成一个初始调度方案，然后使用贪心邻域搜索、遗传搜索和禁忌搜索继续优化 makespan。

当前算法的详细设计见 [design.md](design.md)。

## 当前算法概述

一次运行的处理流程如下：

1. 读取 `.fjs` 实例，得到工件、工序、可选机器和加工时间。
2. 在总计时开始后调用 COPT，求解一个以 makespan 最小为目标的 FJSP 混合整数模型。
3. 如果 COPT 返回可行解，将其转换为 HLS 的机器序列和工序关系图；如果 COPT 在时间限制内没有可行解，则使用基于最早完工时间的多起点 HLS 启发式初始解继续运行。
4. 贪心搜索先围绕零松弛关键工序执行同机重插入和跨机迁移；只有关键邻域没有改进时，才用较小预算检查近关键工序。
5. 自适应调度器根据刷新全局最优的近期收益、单位时间收益和失败冷却选择贪心、遗传或禁忌搜索。深度停滞时增加探索和多样化起点，但仍由实测收益决定方法；最后 10% 时间继续保留短 GA。遗传分支使用 incumbent 注入、结构距离筛选、停滞重建、自适应变异和受限局部强化。
6. 每个候选方案都重新构造有向图，并通过拓扑排序计算最长路径作为 makespan。

默认 COPT 初始解时间为 `300` 秒，总优化时间为 `3600` 秒；总时间从 COPT 求解前开始计时，因此 COPT 时间包含在总限制内。

## 环境要求

- Linux
- C++ 编译器（项目曾使用 GCC/G++ 11.4）
- CMake 3.22 或更高版本
- COPT 8.x，并且已配置许可证

先确认 COPT 环境变量：

```bash
export COPT_HOME="$HOME/copt80"
export COPT_LICENSE_DIR="$HOME/copt80"
export PATH="$COPT_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$COPT_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果 `~/.zshrc` 属于 `root`，普通用户不能直接写入。可以先在当前终端执行上面的命令；需要永久生效时，应修复该文件的所有者，或让管理员将这些变量写入用户可读写的 shell 配置文件。

## 编译

```bash
cd /home/luopw24/HLS
mkdir -p build
cd build
cmake .. -DCOPT_HOME="${COPT_HOME:-$HOME/copt80}"
make -j"$(nproc)"
```

默认可执行文件为：

```text
./build/production_scheduling
```

如果 CMake 没有找到 COPT，请检查 `COPT_HOME` 下是否存在 COPT 的头文件和库文件，以及当前用户是否有执行权限。

## 单次运行

```bash
./build/production_scheduling [input_file] [search_mode] [random_strategy] [max_repeat] [max_iter] [population_size] [tabu_length] [copt_time_limit_seconds] [total_time_limit_seconds] [random_seed]
```

命令行参数及默认值：

| 参数 | 含义 | 默认值 |
| --- | --- | ---: |
| `input_file` | FJSP 输入文件 | `/data/luopw/production_scheduling/input.txt` |
| `search_mode` | 自适应调度器中遗传/禁忌搜索的初始偏好；其余偏向贪心 | `0.5` |
| `random_strategy` | 遗传/禁忌初始偏好中分配给遗传搜索的比例 | `0.5` |
| `max_repeat` | 三种搜索方法均未改进的完整轮数上限 | `100` |
| `max_iter` | 单次方法调用的基准迭代数；贪心和强化调用会自动缩短 | `30` |
| `population_size` | 遗传搜索种群大小 | `20` |
| `tabu_length` | 禁忌表长度 | `25` |
| `copt_time_limit_seconds` | COPT 初始解时间限制 | `300` |
| `total_time_limit_seconds` | COPT 加 HLS 的总时间限制 | `3600` |
| `random_seed` | HLS 根随机种子；调度器、遗传和禁忌使用独立派生随机流 | 当前时间 |

例如运行 Brandimarte 的 Mk03：

```bash
cd /home/luopw24/HLS && COPT_HOME="$HOME/copt80" COPT_LICENSE_DIR="$HOME/copt80" LD_LIBRARY_PATH="$HOME/copt80/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./build/production_scheduling datasets/Brandimarte_Data/Text/Mk03.fjs 0.5 0.5 100 30 20 25 300 3600 100003
```

程序输出中：

- `COPT initial makespan:` 表示 COPT 生成的初始解；
- `HLS fallback initial makespan:` 表示 COPT 没有产生可行解时使用的启发式初始解；
- `mode: intensification/adaptive/diversified` 表示强化、自适应选择或精英起点多样化调用；
- `summary: calls=...` 汇总每种搜索的调用次数、改进次数、相对改善和耗时；
- `Final makespan:` 表示 HLS 结束时的最终结果；
- 最终单独输出的整数也是最终 makespan，便于脚本提取。

## Brandimarte 批量实验

`run_brandimarte_experiments.sh` 默认以 `10` 路并行运行 `Mk01` 到 `Mk10`，每个实例运行 `1` 次，每次 COPT 限制 `300` 秒、总时间限制 `3600` 秒。每次运行会生成时间目录，完整日志和最终汇总分别保存为 `run_logs/<运行时间>/<instance>_run01.txt` 与 `run_logs/<运行时间>/result.csv`。可通过 `RUNS`、`START_RUN`、`PARALLEL_JOBS` 和 `RUN_TIMESTAMP` 调整总次数、起始次数、并行任务数及运行目录。设置 `START_RUN=2` 可以保留已有的第 1 次结果，只补跑第 2 次到最后一次。

使用默认配置：

```bash
cd /home/luopw24/HLS && COPT_HOME="$HOME/copt80" COPT_LICENSE_DIR="$HOME/copt80" LD_LIBRARY_PATH="$HOME/copt80/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" EXECUTABLE=./build/production_scheduling ./run_brandimarte_experiments.sh
```

脚本默认跳过已经包含 `Final makespan:` 的日志，适合中断后继续运行。需要强制重新运行全部实验时设置 `SKIP_COMPLETED=0`。也可以用 `INSTANCES` 指定本次实际运行的实例，用 `AGGREGATE_INSTANCES` 指定写入 CSV 的实例集合。

例如只运行尚未测试的 Mk08、Mk09、Mk10：

```bash
INSTANCES="Mk08 Mk09 Mk10" AGGREGATE_INSTANCES="Mk01 Mk02 Mk03 Mk04 Mk05 Mk06 Mk07 Mk08 Mk09 Mk10" EXECUTABLE=./build/production_scheduling ./run_brandimarte_experiments.sh
```

从已有日志中提取 COPT 初始 makespan 到 `copt.csv`：

```bash
./collect_copt_results.sh
```

对 `Mk01` 到 `Mk10` 各运行一次 COPT，时间限制为 `3600` 秒，并每 `300` 秒记录一次当前 incumbent makespan：

```bash
./check_copt_initial_solution.sh
```

默认并发数为 `5`。进度明细写入 `copt_progress_checks/copt_progress.csv`，最终结果写入 `copt_progress_checks/copt_final_results.csv`，单次运行日志写入同目录下的 `logs/`。脚本通过 COPT incumbent 回调记录求解过程中的可行解，并在 COPT 初始解输出后停止后续 HLS 搜索。可通过 `PARALLEL_JOBS`、`COPT_TIME`、`CHECKPOINT_SECONDS` 和 `INSTANCES` 覆盖默认值。

该脚本使用 `CoptInitialSolution.cpp` 中新增的回调支持，因此修改源码后需要先重新编译 `build/production_scheduling`。

当某次 COPT 没有可行 incumbent 时，`copt.csv` 对应位置记录为 `NA`，不会把 HLS fallback 初始解误记为 COPT 结果。

## 主要源文件

- `main.cpp`：参数解析、COPT 初始化、HLS 外层循环和最终输出。
- `AdaptiveSearch.cpp`：按局部单位时间收益、成功率、停滞程度和探索奖励选择搜索方法。
- `SearchDeadline.h`：向各搜索方法传递统一的单次调用和总优化截止时间。
- `CoptInitialSolution.cpp`：COPT FJSP 初始解模型及解提取。
- `DataProc.cpp`：数据读取、调度图构造、makespan 计算和停止判断。
- `GreedySearch.cpp`：交换和迁移邻域的贪心搜索。
- `RandomSearch.cpp`：持久化遗传搜索、关键路径变异、编码解码及兼容旧调用的搜索分派接口。
- `TabuSearch.cpp`：关键块 N5/N6、独立邻域配额、跨调用轨迹、移动属性禁忌和停滞恢复。
- `Schedule.cpp`：调度对象及关键路径工序计算。
- `run_brandimarte_experiments.sh`：并行批量实验和 `result.csv` 汇总。
- `collect_copt_results.sh`：从日志汇总 COPT 初始解到 `copt.csv`。

## 设计文档

关于 FJSP 建模、调度图、COPT 初始解、贪心邻域、遗传搜索、禁忌搜索以及停止条件的中文说明，请阅读 [design.md](design.md)。
