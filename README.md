# HLS-FJSP

HLS-FJSP 是一个使用 C++ 实现的柔性作业车间调度问题（FJSP）混合搜索程序。当前版本先调用 COPT 为实例生成初始调度，再使用贪心、遗传、禁忌和可选的引导式跨机插入搜索继续优化 makespan。

当前算法的详细设计见 [design.md](design.md)。

## 当前算法概述

一次运行的处理流程如下：

1. 读取 `.fjs` 实例，得到工件、工序、可选机器和加工时间。
2. 在总计时开始后调用 COPT，求解一个以 makespan 最小为目标的 FJSP 混合整数模型。
3. 如果 COPT 返回可行解，将其转换为 HLS 的机器序列和工序关系图；如果 COPT 在时间限制内没有可行解，则使用基于最早完工时间的多起点 HLS 启发式初始解继续运行。
4. 贪心搜索先围绕零松弛关键工序执行同机重插入和跨机迁移；只有关键邻域没有改进时，才用较小预算检查近关键工序。
5. 自适应调度器根据刷新全局最优的近期收益、单位时间收益和失败冷却选择搜索方法。深度停滞时增加探索和多样化起点，但仍由实测收益决定；最后 10% 时间继续保留短 GA。
6. 对至少 80 道工序的实例，停滞后可调用 Guided Insertion：从最多 6 条持久轨迹生成关键/近关键跨机插入动作，分层选择候选做精确评价。控制器在 Guided 连续 12 次调用无全局改善后才切换为周期探测，避免过早放弃需要多步穿越平台的轨迹。`HLS_ENABLE_GUIDED_SEARCH=0` 可关闭，`force` 可忽略规模门槛。
7. Greedy 可选用 HLS 自主训练的 PPO Actor 对同机重插入、跨机插入和同机交换动作排序，只精确评价 Top-K；Tabu 也可暂时复用同一模型筛选五类邻域。两者默认关闭，模型不可用时分别回退原始搜索。
8. 每个被选中的候选方案都重新构造有向图，并通过拓扑排序计算最长路径作为 makespan。

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

## 强化学习开发环境

用于后续 Greedy/Tabu 候选排序、图策略网络和 PPO 实验的独立 Conda 环境名为 `hls-rl`。当前机器上的环境使用 Python 3.11、PyTorch 2.5.1（CUDA 12.4 Runtime）和 PyG 2.6.1，已在 Tesla V100-PCIE-32GB（`sm_70`）上通过图卷积前向、反向传播测试。环境自带 CUDA Runtime，不要求系统安装 `nvcc`，但仍依赖可用的 NVIDIA 驱动。

激活环境：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hls-rl
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

在其他机器上可根据仓库中的 `environment-rl.yml` 重建：

```bash
conda env create -f environment-rl.yml
conda activate hls-rl
```

该环境同时保留 `gym==0.26.2` 以兼容传统 Gym API，并提供 `gymnasium` 供新代码使用。`torch-geometric` 用于 GIN/GAT 图模型，TensorBoard 用于训练监控，ONNX 用于后续模型导出。Jupyter kernel 名称为 `Python (hls-rl)`。

HLS 自主实现的 Greedy PPO 训练、测试、模型导出和 C++ 开关见 [rl_greedy/README.md](rl_greedy/README.md)。该实现不依赖 L2S 的 Python 模块或已有模型。

当前推荐的正式验证使用一套统一参数，并在运行前为十个实例分别随机生成 seed：

```bash
cd /home/luopw24/HLS && PARALLEL_JOBS=5 ./run_uniform_rl_experiments.sh
```

该脚本保持 Greedy、GA、Tabu 和 Guided，启用 Greedy RL 与 Tabu RL；Mk01～Mk10 使用完全相同的搜索、候选和时间参数，最大并发数为 5。未提供 `SEEDS_FILE` 时脚本从 `/dev/urandom` 生成十个独立 seed，随后写入 `best-config.env` 供复现。历史每实例 profile 的最佳观测组合及证据日志仍保存在 [best-rl-results.md](best-rl-results.md)，但不再作为统一参数实验的最终验收结果。

筛查实例子集时可设置 `INSTANCES`；参数仍对所选实例保持统一，脚本会按实例数量生成并校验 seed。例如优先验证 Mk02、Mk04、Mk05：

```bash
cd /home/luopw24/HLS && INSTANCES="Mk02 Mk04 Mk05" PARALLEL_JOBS=3 COPT_TIME=30 TOTAL_TIME=300 ./run_uniform_rl_experiments.sh
```

脚本默认把 `Mk01.txt`～`Mk10.txt`、`result.csv`、`comparison_l2s.csv`、随机 seed 和 `best-config.env` 写入 `run_logs/<运行时间>/`。默认统一参数为 COPT 300 秒、总时限 3600 秒、`search=0.60`、`random=0.50`、`max_repeat=1000`、`max_iter=45`、种群 32、禁忌长度 30、Greedy RL Top-K 256、Tabu RL Top-K 1024，以及每类 Tabu 邻域至少保留 24 个候选。候选预算、规则/探索比例和 Tabu 非改善回退可通过对应的全局环境变量统一覆盖。

### 10 个实例各运行 10 次，动态并发

使用当前固定可执行文件和统一参数进行 100 次实验：

```bash
cd /home/luopw24/HLS && ./run_adaptive_brandimarte.sh
```

脚本在前台运行，每个实例默认运行 10 次，启用 Greedy RL、Tabu RL 和 Guided，保留贪心、遗传和禁忌搜索。默认 COPT 时限 300 秒、每次总时限 3600 秒，其他搜索参数与上述统一参数入口一致。100 个 seed 在启动时随机生成并保存，不按结果挑选或更换。C++ 模型推理使用 CPU，不需要 GPU 或激活 Conda；启动器会自动寻找 Python 3.8 及以上版本，只使用标准库，也可通过 `PYTHON` 指定解释器。

调度器每 5 秒检查 CPU 利用率、可用内存及当前实验的内存占用，并识别容器的 CPU quota 和内存限制。它逐步增加并发，每次最多增加 2 个任务；当资源不足时停止补充新任务，待正在运行的任务结束后自然降低并发。每个实验默认绑定 1 个逻辑 CPU，COPT 创建的线程也继承这个限制；这一资源设置会影响求解速度，因此本轮会记录实际 CPU 分配。默认每个实验预留 2 GiB，并根据已观测的最大内存占用提高估计；系统预留至少 4 GiB 或内存上限的 10%，取较大值。

可先查看当前资源配额与预计并发上限，不创建结果目录，也不执行实验：

```bash
cd /home/luopw24/HLS && ./run_adaptive_brandimarte.sh --dry-run
```

如需限定最多同时运行 10 个实验，仍由脚本在该上限内动态调整：

```bash
cd /home/luopw24/HLS && MAX_PARALLEL=10 ./run_adaptive_brandimarte.sh
```

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `RUNS` | `10` | 每个实例运行次数 |
| `INSTANCES` | `Mk01`～`Mk10` | 实例集合 |
| `MAX_PARALLEL` | `0` | 并发上限，0 表示按资源计算；也兼容 `PARALLEL_JOBS` |
| `CPUS_PER_JOB` | `1` | 每个实验绑定的 CPU 数 |
| `CPU_USAGE_LIMIT` | `0.85` | 目标 CPU 利用率，此外预留至多 1 核 |
| `JOB_MEMORY_GB` | `2` | 每个实验初始内存预留，单位 GiB |
| `MEMORY_RESERVE_GB` | `4` | 系统内存预留，同时至少保留内存上限的 10% |
| `CHECK_INTERVAL` | `5` | 资源检查间隔，单位秒 |
| `RAMP_UP` | `2` | 每次检查最多新增的实验数 |

所有结果实时写入 `run_logs/<北京时间>/`；具体绝对路径会在启动时输出。`result.csv` 每个实例一行，后接 `run1`～`run10` 的 makespan；`summary.csv` 包含均值、最好值、最差值、样本标准差和当前 `l2s_brandimarte.csv` 的均值对比，未跑满的实例明确标记为 `INCOMPLETE`。`details.csv` 记录每次 seed、状态、耗时和 CPU 分配；`MkXX_runYY.txt` 保存各次完整日志；`scheduler.log`、`scheduler.csv` 和 `status.md` 记录调度过程与实时进度；`seeds.csv`、`config.json` 和 `inputs/` 保存随机种子、统一参数以及可执行文件、模型、数据和 L2S 参考结果的副本与 SHA-256，保证本批实验使用相同版本。

按 `Ctrl+C` 会停止本批启动的实验进程，保存已完成结果。中断或出现失败后，在同一目录续跑时保持原 seed、模型和搜索参数，并跳过已成功完成的实验：

```bash
cd /home/luopw24/HLS && LOG_DIR=/home/luopw24/HLS/run_logs/<原运行时间> RESUME=1 ./run_adaptive_brandimarte.sh
```

续跑仅对未成功完成的实验重新执行，原未完成日志保存在 `attempts/`，失败和中断不作为有效 makespan 参与均值。若仍有失败任务，脚本返回非零状态，不把部分结果标记为完整实验。

### 2026-10-13 优化候选的重复验证

新的优化入口保留贪心、遗传、禁忌和 Guided，启用两种 RL，以及动态工序分派、贪心有界联合邻域、GA 工作量重分配、连续禁忌轨迹和统一停滞重启。标准联合邻域失败后间隔尝试包含非关键交换伙伴的扩展邻域。实现细节见 [design.md](design.md) 的可选优化方案。

```bash
cd /home/luopw24/HLS && ./run_optimized_hls.sh
```

默认 Mk01～Mk10 各 10 次、随机 seed、每次总时限 600 秒、最多动态并发 14，使用 `build/optimized/production_scheduling`，原固定程序保持可用。初始 COPT 时限为 0，COPT 仅用于 Greedy 内部的有界邻域修复，因此运行仍需 COPT 库及许可证。正式结果须查看对应目录的完整 `summary.csv`，不能用短测或单次最好值替代；本轮正在进行 100 次验证，尚未宣称所有实例的均值达到 L2S。这个时限与先前 3600 秒批次不同。

查看资源计划可加 `--dry-run`。所有实验使用同一套全局设置，`COPT_TIME`、`TOTAL_TIME`、`MAX_PARALLEL` 等可统一覆盖。中断续跑时设置 `LOG_DIR=<原目录> RESUME=1`，读取原配置与 seed，并跳过成功任务。

### Brandimarte、Barnes、Dauzere 全量重复实验

运行三组 SmartLab FJSP 数据的全部 instance，每个 instance 默认运行 10 次，共 490 次：Brandimarte 10 个、Barnes 21 个、Dauzere 18 个。三个 family 共享一个资源调度器，按当前 CPU quota、CPU 利用率、可用内存和已运行任务的内存占用动态调整并发；默认最多 14 个并发、每个任务绑定 1 个逻辑 CPU。每个实验使用独立随机 seed，seed、完整日志和调度状态保存到时间目录，支持中断续跑。

```bash
cd /home/luopw24/HLS && ./run_smartlab_hls.sh
```

默认使用当前固定优化方案：Greedy、遗传、Tabu、Guided 和两种 RL 均开启，初始 COPT 时限为 0，单次总时限为 600 秒。输出文件直接写入 HLS 根目录：`HLS_brandimarte.csv`、`HLS_barnes.csv` 和 `HLS_dauzere.csv`，格式为 `instance,run1,...,run10`。完整日志和 `seeds.csv` 位于 `run_logs/smartlab_<运行时间>/`。

需要将三个 CSV 写入其他目录时可设置 `OUTPUT_DIR=/path/to/output`；默认仍写入 HLS 根目录。

先查看 instance 数量、资源上限和预计任务数，不启动实验：

```bash
cd /home/luopw24/HLS && ./run_smartlab_hls.sh --dry-run
```

也可以统一调整运行次数、总时限或并发上限，例如每个 instance 运行 10 次，最多允许 10 个并发：

```bash
cd /home/luopw24/HLS && RUNS=10 MAX_PARALLEL=10 TOTAL_TIME=600 ./run_smartlab_hls.sh
```

只运行指定 family：

```bash
cd /home/luopw24/HLS && ./run_smartlab_hls.sh --families barnes dauzere
```

默认不会覆盖已有的三个输出文件；确认需要重新生成时设置 `OVERWRITE_OUTPUT=1`。中断后使用原运行目录续跑，已成功的任务会跳过并沿用原 seed：

```bash
cd /home/luopw24/HLS && LOG_DIR=/home/luopw24/HLS/run_logs/smartlab_<运行时间> RESUME=1 ./run_smartlab_hls.sh
```

### Greedy PPO 训练与测试

小批量训练：

```bash
cd /home/luopw24/HLS && source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate hls-rl && python -m rl_greedy.train --epochs 5 --episodes-per-epoch 4 --horizon 10 --ppo-epochs 2 --ranking-epochs 2 --ranking-temperature 0.05 --hidden-size 32 --learning-rate 0.0003 --min-jobs 10 --max-jobs 12 --min-machines 4 --max-machines 8 --min-operations 3 --max-operations 15 --max-machine-options 6 --min-processing-time 1 --max-processing-time 20 --device cuda --seed 20260928 --checkpoint rl_greedy/models/greedy_policy_small.pt --export rl_greedy/models/greedy_policy_small.txt
```

测试小批量模型：

```bash
python -m rl_greedy.evaluate datasets/Brandimarte_Data/Text/Mk01.fjs datasets/Brandimarte_Data/Text/Mk02.fjs --model rl_greedy/models/greedy_policy_small.txt --top-k 96 --steps 10 --device cuda
```

完整训练：

```bash
cd /home/luopw24/HLS && source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate hls-rl && python -m rl_greedy.train --epochs 200 --episodes-per-epoch 8 --horizon 30 --ppo-epochs 4 --ranking-epochs 2 --ranking-temperature 0.05 --ranking-state-batch-size 8 --hidden-size 64 --learning-rate 0.0002 --min-jobs 10 --max-jobs 20 --min-machines 4 --max-machines 15 --min-operations 3 --max-operations 15 --max-machine-options 6 --min-processing-time 1 --max-processing-time 20 --device cuda --seed 20260928 --checkpoint rl_greedy/models/greedy_policy_brandimarte.pt --export rl_greedy/models/greedy_policy_brandimarte.txt
```

在 Brandimarte 的 Mk01～Mk10 上各测试一次，并将结果写入独立时间目录：

```bash
python -m rl_greedy.evaluate datasets/Brandimarte_Data/Text/Mk*.fjs --model rl_greedy/models/greedy_policy_brandimarte.txt --top-k 96 --dynamic-top-k --rule-fraction 0.25 --exploration 0.15 --temperature 1.0 --steps 10 --device cuda --seed 20260928 --output "rl_greedy/results/$(date +%Y%m%d_%H%M%S)/result.csv"
```

模型文件只会在所有训练 epoch 完成后输出。`Mk*.fjs` 中的 `*` 不要使用反斜杠转义。评估器对展开后的每个实例只执行一次，并在 CSV 中记录 `run=1`、实例 seed、模型绝对路径及 SHA-256、Top-K、最大步数、初始/基线/RL makespan、精确评价次数和实际搜索步数。

使用训练后的模型依次运行完整 HLS，Mk01～Mk10 各运行一次：

```bash
cd /home/luopw24/HLS && COPT_HOME="$HOME/copt80" COPT_LICENSE_DIR="$HOME/copt80" LD_LIBRARY_PATH="$HOME/copt80/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" HLS_GREEDY_RL=1 HLS_GREEDY_RL_MODEL=/home/luopw24/HLS/rl_greedy/models/greedy_policy_brandimarte.txt HLS_GREEDY_RL_TOP_K=96 HLS_GREEDY_RL_DYNAMIC_TOP_K=1 HLS_GREEDY_RL_RULE_FRACTION=0.25 HLS_GREEDY_RL_FALLBACK_ON_MISS=1 HLS_GREEDY_RL_TEMPERATURE=1.0 HLS_GREEDY_RL_EXPLORATION=0.15 HLS_TABU_RL=1 HLS_TABU_RL_MODEL=/home/luopw24/HLS/rl_greedy/models/greedy_policy_brandimarte.txt HLS_TABU_RL_TOP_K=256 HLS_TABU_RL_DYNAMIC_TOP_K=1 HLS_TABU_RL_RULE_FRACTION=0.30 HLS_TABU_RL_EXPLORATION=0.20 HLS_TABU_RL_MIN_PER_NEIGHBORHOOD=16 HLS_TABU_RL_FALLBACK_ON_MISS=1 RUNS=1 PARALLEL_JOBS=1 SKIP_COMPLETED=0 EXECUTABLE=./build/production_scheduling ./run_brandimarte_experiments.sh
```

该命令同时启用 Greedy PPO 与 Tabu RL 邻域筛选，并保留 GA 和默认规模门控下的 Guided 搜索。Tabu 当前复用 Greedy 模型，但有独立的候选预算、30% 规则保底、20% 探索、五类邻域最低配额和 fallback。`RUNS=1` 表示每个实例运行一次，`PARALLEL_JOBS=1` 保证 Mk01～Mk10 顺序执行。完整日志和汇总结果分别写入 `run_logs/<运行时间>/MkXX_run01.txt` 与 `run_logs/<运行时间>/result.csv`。模型推理由 C++ 原生执行，不需要激活 Conda，也不使用 GPU。启动日志应同时出现 `greedy_rl_enabled: true`、`tabu_rl_enabled: true` 和对应的 `model loaded` 状态；模型不可用时各方法自动回退原始搜索。

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
| `max_repeat` | 所有当前启用搜索方法均未改进的完整轮数上限 | `100` |
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
- `GreedyRlPolicy.cpp`：Greedy PPO Actor 文本模型加载、原生打分及 Top-K/探索候选选择。
- `rl_greedy/`：HLS 自主实现的 FJSP 强化学习环境、PPO 训练、评估和模型导出。
- `RandomSearch.cpp`：持久化遗传搜索、关键路径变异、编码解码及兼容旧调用的搜索分派接口。
- `TabuSearch.cpp`：关键块 N5/N6、独立邻域配额、跨调用轨迹、移动属性禁忌和停滞恢复。
- `TabuRlPolicy.cpp`：复用 Greedy Actor 权重的 Tabu 候选筛选、动态预算、规则保底和探索。
- `GuidedSearch.cpp`：关键/近关键跨机插入、分层 Top-K 精确评价、多轨迹和访问记忆。
- `Schedule.cpp`：调度对象及关键路径工序计算。
- `run_brandimarte_experiments.sh`：并行批量实验和 `result.csv` 汇总。
- `run_uniform_rl_experiments.sh`：统一参数、按实例生成随机 seed、支持实例子集、最多五进程的双 RL 实验及 L2S 对比汇总。
- `collect_copt_results.sh`：从日志汇总 COPT 初始解到 `copt.csv`。

## 设计文档

关于 FJSP 建模、调度图、COPT 初始解、贪心邻域、遗传搜索、禁忌搜索以及停止条件的中文说明，请阅读 [design.md](design.md)。
