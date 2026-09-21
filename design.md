# HLS-FJSP 设计说明

本文以当前 `main.cpp`、`DataProc.cpp`、`CoptInitialSolution.cpp`、`GreedySearch.cpp`、`RandomSearch.cpp`、`TabuSearch.cpp` 和 `AdaptiveSearch.cpp` 的**实际实现**为准，说明 HLS-FJSP 如何求解柔性作业车间调度问题（FJSP）。主程序先构造完整可行解，再交替调用贪心、遗传和禁忌三种搜索；始终单独保留全局最小 makespan。

本文只描述已经运行的 HLS 代码。`l2s-design.md` 记录的是后续拟引入的学习引导搜索方案，目前**未接入**主程序，也不是第四种已实现的搜索方法。

## 1. FJSP 问题定义

设有若干工件，每个工件包含一串有先后顺序的工序。与普通 JSP 不同，FJSP 中一条工序通常可以选择多台机器，每台候选机器有自己的加工时间。

对工件 `j` 的第 `k` 道工序，记为 `O(j,k)`，其可选机器集合为：

```text
{ (m, p(j,k,m)) }
```

其中 `m` 是候选机器，`p(j,k,m)` 是在机器 `m` 上的加工时间。一个完整调度方案必须满足：

1. 每道工序只能选择一台允许的机器。
2. 同一工件的工序必须按照工艺顺序执行。
3. 同一台机器上的工序必须有确定的先后顺序，且不能重叠。

当前程序只把 makespan 作为搜索目标：

```text
min Cmax
```

其中 `Cmax` 是所有工件完成时间的最大值。代码还会计算机器故障率，但该指标目前没有参与 makespan 的比较目标。

## 2. 数据结构与调度表示

输入文件由 `DataProc.cpp::Init` 读取。文件首行为工件数、机器数及可为小数的第三个说明字段，随后每个工件行依次列出工序数、各工序的可用机器数和 `(机器编号, 加工时间)` 列表。若输入使用 1 开始的机器编号且最大编号等于机器数，读取阶段转换为内部从 0 开始的编号；否则保持原有编号。程序建立一个包含各工件各 1 份的订单，通过 `OrderToJobList` 得到本次调度的 `jobList`。核心数据结构如下：

- `Job`：工件编号、名称和工序列表。
- `Job_process`：一道工序的全部机器候选。
- `Process_item`：一个机器候选及其加工时间。
- `Schedule_item`：一台机器上的工序序列。
- `Schedule`：完整调度方案、机器序列、工序关系图、makespan 和开始时间。
- `Machine`：机器编号、加工工序数及故障率等属性；故障率不属于当前 makespan 目标。

`Schedule` 的主要表示方式是“按机器存储工序序列”：

```text
machine 0: O(2,0) -> O(1,1) -> ...
machine 1: O(1,0) -> O(3,0) -> ...
```

`Schedule_process` 用 `(job_id, process_id)` 标识工序，`Schedule_item.process_count` 与其 `schedule_process` 长度同步。`Schedule` 另存 `processList`、图的邻接矩阵、最早开始时间向量、`TotalTime` 和故障率。机器序列决定机器约束，工件编号和工序编号决定工艺约束。遗传搜索还会使用两条编码：

- 机器选择编码：每道工序选择其候选机器列表中的第几个选项。
- 工序排序编码：保存工件编号序列；同一个工件出现多次时，解码器依次取该工件的下一道工序。

因此一组机器分配和机器序列描述的是**已经包含全部工序的完整方案**，不是逐道工序尚待填充的部分调度。搜索修改完整方案之后，重新求所有工序的开始时间。

## 3. 用有向图计算 makespan

`ScheduleItemsToGraph` 遍历机器序列建立工序位置及所选机器的映射，使用各 `(工件, 工序, 机器)` 的实际加工时间，把调度方案转换为邻接矩阵；矩阵值 `-1` 表示没有边。图中包含：

- `start` 节点；
- 每道实际工序节点，例如 `job2-0`；
- `end` 节点。

图边表示两类约束：

1. 工件边：同一工件的第 `k` 道工序指向第 `k+1` 道工序，边权是前一道工序在所选机器上的加工时间。
2. 机器边：同一机器序列中，前一道工序指向后一道工序，边权是前一道工序的加工时间。

工序完成后通向 `end` 的边也带有当前加工时间；可行方案的最后一个节点就是 `end`。若机器边和工件边指向同一对节点，它们表示相同前序工序的完工约束。

`CalculateTotalTime` 先扫描邻接矩阵计算入度，然后用零入度队列进行拓扑排序，同时动态规划最早开始时间：

```text
earliest[v] = max(earliest[v], earliest[u] + weight(u,v))
```

`end` 节点的最早时间就是 makespan，同时写回 `Schedule.start_time` 和 `Schedule.TotalTime`。访问节点数不等于图的节点数时说明存在环，函数抛出异常；Greedy/Tabu 候选评价捕获后跳过该方案，遗传解码则必须产生可拓扑排序的方案。

`Schedule::GetKeyProcess()` 通过最早/最迟时间找所有零松弛工序；`GetProcessSlack()` 返回各工序松弛时间，供近关键邻域及等 makespan 解的次级比较使用。它们读取已生成的图和工序列表，不是额外的目标函数。

当前图为工序数加两个虚拟节点的**稠密邻接矩阵**。每次候选通常复制 `Schedule` / 机器序列、重建图并重新拓扑计算；图构造与评估在大实例上有明显的内存和计算开销。稀疏图、增量 makespan 或学习模型均尚未作为当前求值器实现。

## 4. 初始解：COPT 加启发式回退

### 4.1 COPT 初始解模型

`CoptInitialSolution.cpp::GenerateCoptInitialSolution` 将所有工序摊平，设 `horizon` 为每道工序**最长候选加工时间之和**（至少 1）；该值作为连续变量上界和机器互斥约束的 Big-M。模型为每道工序建立：

- 连续开始时间变量 `start`；
- 连续完成时间变量 `finish`；
- 对每个候选机器建立一个二进制分配变量；
- 一个全局连续 makespan 变量，以及可能使用同一机器的每对工序之间的二进制前后顺序变量。

模型约束包括：

1. 每道工序的机器分配变量之和等于 1。
2. `finish` 等于 `start` 加所选机器的加工时间。
3. 相邻工序满足工件先后约束。
4. makespan 不小于每道工序的完成时间。
5. 对可能使用同一台机器的工序对，使用二进制顺序变量和 Big-M 约束决定先后关系；两工序未同时分配到这台机器时，分配变量使对应约束失效。

目标函数是最小化 makespan。调用 `Solve()` 后检查 `HASSOL` 而不是要求全局最优；只要在时限内找到可行 incumbent 即可使用。通过大于 0.5 的机器分配变量确定机器，从 COPT 开始时间重建机器序列，开始时间几乎相同时以工件和工序 ID 作确定性排序，最后重建 HLS 图并重新计算 makespan。

只有设置 `HLS_COPT_PROGRESS_FILE` 时才注册 COPT incumbent 回调：回调在 `COPT_CBCONTEXT_INCUMBENT` 读取 `COPT_CBINFO_BESTOBJ`，按相对于回调创建时刻的秒数写入 `elapsed_seconds,makespan`；正常求解无需此文件。COPT 无可行解、模型/许可证异常或结果提取失败时抛异常并触发回退。

### 4.2 COPT 失败时的回退

COPT 的作用是提供较好的起点，并不要求整个程序必须依赖 COPT 找到可行 incumbent。若 COPT 抛出异常，`main.cpp` 的 `BuildMultiStartHeuristicInitialSolution` 多次调用 `DataProc.cpp::GenerateInitialSolution`：

1. 先在原 `jobList` 顺序上生成一份解，再最多尝试 `min(32, max(8, 2 * 工件数)) - 1` 个随机打乱工件顺序的起点，始终检查总优化截止时间。
2. 每个起点按选定工件顺序遍历；一个工件内部始终按工序先后顺序处理。对工序的候选机器，按 `finish = max(job_ready, machine_available) + process_time` 计算预计完工时间，优先选择更早完工的机器；相同时依次比较加工时间、机器可用时间和故障率。
3. 将工序追加到选中机器的序列，更新工件/机器的就绪时间；全部工序分配后重建图并计算 makespan，跨起点保留严格更好的方案。

多起点的 `std::mt19937` 种子由 `random_seed ^ 0xD1B54A32U` 导出。此回退发生在 COPT 用时之后，也占用同一个总时限；若回退仍失败，则报错退出，而不是返回空结果。

默认 COPT 时间为 `300` 秒，总时间为 `3600` 秒。优化计时点位于 COPT 调用之前，所以 COPT 消耗的时间会占用总时间预算。实际传给 COPT 的时间限制还会取 `copt_time_limit_seconds` 与总时间限制中的较小值。

## 5. HLS 外层搜索流程

`main.cpp` 分开维护：`output_schedule` 为全局最好方案，`input_schedule` 为后续搜索的工作起点。候选严格优于全局最好时同时更新二者；候选与全局最好 makespan 相同但机器序列不同，也可只更新工作解，供后续搜索穿越等值平台。精英库最多保留 8 个签名互异的方案，按加入顺序淘汰最旧项；它包含严格更好的解，也可能包含等值工作解。深停滞且选中 GA 时可从精英库选择不同起点：65% 选择最新项，其他情况均匀抽取更旧项（只有一项则使用工作解）。

搜索方法不再由两层固定概率随机分支决定。`AdaptiveSearchController` 为贪心、遗传和禁忌分别记录调用次数、改进次数、累计相对改善、累计耗时和近期单位时间收益。

`AdaptiveSearchController::Record()` 区分方法起点的局部改进与是否刷新全局最好。设 `local_before`/`local_after` 为该方法的调用前后 makespan，`global_before`/`global_after` 为外层全局最好值：

```text
local_gain  = max(0, (local_before - local_after) / local_before)
global_gain = max(0, (global_before - global_after) / global_before)
reward = min(1, (global_gain + 0.05 * local_gain) / max(0.05, 调用秒数))
recent_reward = 第一次调用时 reward，否则 0.70 * recent_reward + 0.30 * reward
```

仅在严格刷新全局最优时才重置该方法的 `consecutive_failures`。方法选择的评分项可直接写成：

```text
global_success_rate = global_improvements / calls
local_success_rate  = local_improvements / calls
time_efficiency = min(1, 20 * total_global_relative_gain / max(0.1, total_seconds))
exploration_bonus = 0.18 * sqrt(log(total_calls + 2) / (calls + 1))
failure_penalty   = min(0.20, 0.02 * consecutive_failures)
recency_bonus     = 从未调用 ? 0.10 : min(0.12, 0.01 * (total_calls - last_called_at))
score = 0.40 * recent_reward + 0.25 * time_efficiency
      + 0.15 * global_success_rate + 0.05 * local_success_rate
      + 0.05 * initial_preference + exploration_bonus
      - failure_penalty + recency_bonus + 阶段偏置
```

当 `calls=0` 时两个成功率均为 0。`阶段偏置` 包括：Tabu 在连续至少 5 次外层停滞后最多加 0.10，GA 在连续至少 12 次停滞后加 0.12，剩余总时间比例不超过 10% 时 Greedy 加 0.08。固定偏置并不强制选用某一种方法。

`search_mode_param` 给定遗传/禁忌相对于 Greedy 的初始偏好；`random_search_strategy_param` 再把该部分分给遗传与禁忌。三个偏好先各设至少 0.05 再归一化；它们**不是**每次固定的选中概率。控制器在常规评分前尽量让每个未冷却的方法得到至少 2 次采样。

搜索阶段规则如下：

- COPT 或启发式初始解生成后，先执行最多 3 轮贪心和最多 15 轮禁忌强化；
- 自适应调用找到新全局最优解后，再排入一次短贪心和短禁忌强化；
- 方法评分以刷新全局 incumbent 的近期收益和累计单位时间收益为主，历史精英上的局部改善只获得少量多样性信用；
- 只要还有其他方法未持续失败，连续 6 次调用未刷新全局最优的方法就被暂时排除； `(total_calls + 4 * method_index) % 12 == 0` 时允许该方法再探测。若三种方法都达到冷却条件，则恢复全部可选，避免无方法可用；
- 每轮以 `(停滞次数 >= 12 ? 0.20 : 0.08) + 0.08 * 剩余时间比例` 的概率进行偏好加权随机探索，否则使用最高分；深停滞时选中 GA 则设置多样化精英起点；
- 剩余时间不足总预算的 10% 时将单次调用缩短到 2 至 5 秒，遗传搜索仍可继续演化持久化种群。

三种方法都有**内部迭代数**和**单次墙钟截止时间**两层限制；总截止时间取两者中更早者。默认 `max_iter_count=30` 时，正常 Greedy 最多 6 轮、强化 Greedy 最多 3 轮；正常 Tabu 最多 30 轮、强化 Tabu 最多 15 轮；GA 最多 30 代。非强化单次预算：Greedy 10 秒、Tabu 20 秒、GA 30 秒；强化预算：Greedy 3 秒、Tabu 10 秒。剩余总时长不超过 10% 时，Greedy 最多 2 轮和 2 秒、Tabu 最多 10 轮和 5 秒、GA 仍允许最多 30 代但最多 5 秒。所有调用均受总体 `SearchDeadline` 约束，并在候选枚举和遗传迭代内部检查截止时间。

外层满足任一条件就停止：

- 从 COPT 开始计时的总时间达到 `total_time_limit_seconds`；
- 外层搜索次数超过 `1200`；
- 贪心、遗传和禁忌三种方法在某段连续搜索中都至少失败过一次时，累计一个“无全局改进完整轮”；完整轮数超过 `max_repeat_count`。任一方法刷新全局最好值就将停滞计数、完整轮数和各方法失败标志清零。

调度器、遗传算法和禁忌搜索分别使用由命令行 `random_seed` 异或不同常量派生的随机流。遗传种群、代数、停滞计数和适应度缓存会在同一进程/实例的多次调用之间保留；禁忌搜索会保留当前非最优轨迹、短期禁忌属性、绝对迭代号和移动频率。跨实例时由问题签名重置这些内部状态。批量脚本每次调用单独启动可执行文件，因此**不同运行之间不共享**搜索统计或解的记忆。

## 6. 贪心局部搜索

贪心搜索位于 `GreedySearch.cpp`，采用迭代 best-improvement：每次从当前方案生成同机重插入和跨机迁移两个邻域，选出最优候选作为下一次搜索起点。主目标仍是 makespan；当候选 makespan 相同，使用“关键工序数量更少、总松弛时间更大”作为次目标，并最多接受 4 次等值结构移动。单步邻域没有继续改善时，还会对关键工序执行受限的双工序同机交换，再决定是否停止。

### 6.1 关键工序筛选

搜索通过 `GetProcessSlack()` 识别零松弛关键工序；候选按 slack、机器当前总加工负载、工序加工时间、开始时间排序。跨机搜索要求工序拥有不止一个机器候选，且在相同 slack 下优先考虑当前机器时间减去最短可选加工时间的潜在收益。关键邻域最多选择 28 道工序，其中排名前 8 道始终保留，剩余席位由跨调用递增的窗口轮换；近关键邻域最多选择 12 道工序，其中前 3 道固定。每次 `GreedySearch` 调用选择一个窗口，在该调用的各轮复用。

仅当本轮精确关键邻域没有**严格降低 makespan** 时，才额外检查 `0 < slack <= max(6, min(20, Cmax / 40))` 的近关键工序；近关键移动仍需要通过图评价和截止时间限制。

同机重插入和跨机迁移分别选取 best-improvement 候选。`ScheduleQuality` 的比较顺序为 `(makespan, 零松弛工序数, -工序总松弛)`；后两项**仅用于 makespan 相等时**选择结构不同的工作解，最终输出仍按 makespan 判断全局改进。评估一个候选时复制调度对象、更新机器序列、重建 `processList`/图、执行拓扑最长路；成环抛异常并跳过。

### 6.2 同机重插入邻域

对每道候选关键工序，从当前机器序列中移除并枚举该机器上除原位置外的全部插入位置，既覆盖相邻交换，也覆盖较远距离的前移和后移。候选进入精确图评价后按上述 `ScheduleQuality` 比较；没有硬编码的“同机只能交换前一个工序”限制。

### 6.3 跨机迁移邻域

对每道柔性关键工序，搜索会：

1. 从当前机器序列中删除该工序，包括位于序列第一个位置的工序。
2. 枚举该工序允许的所有其他机器。
3. 在每台目标机器上枚举从开头到末尾的全部插入位置。
4. 重建调度图并通过拓扑排序过滤有环方案。
5. 使用精确计算出的 makespan 比较所有可行候选。

目标机器先按“机器累计加工负载 + 该工序在目标机器上的加工时间”排序，再逐一枚举目标机器从开头到结尾的**所有**插入槽位；排序只是有限预算下的尝试顺序，最终优劣仍由重建图后的精确 makespan 决定。故障率只作为初始化平局时的选择依据和方案记录，不是主搜索目标。

### 6.4 搜索规模控制

一次下降迭代最多提交 600 个零松弛同机重插入候选、1800 个零松弛跨机迁移候选；近关键补充阶段最多 180 个同机候选和 480 个跨机候选。若仍无严格 makespan 改进，再从关键工序对中尝试同机双工序交换，预算最多 240 次（计数发生在同机检查之前，可能包含不适用的工序对）。所有阶段都受单次截止时间限制，任一候选成环则继续寻找其他候选。接受等 makespan 的更优结构最多连续 4 次；出现严格 makespan 改进即清零该计数。

## 7. 遗传与禁忌搜索

主循环由自适应调度器直接调用 `AggressiveSearch`。原有 `RandomSearch` 包装接口继续保留以兼容其他调用者，但不再参与主程序的方法选择。

### 7.1 遗传搜索

`AggressiveSearch` 使用机器选择编码和工序排序编码共同表示一个 FJSP 解：机器基因是该工序候选机器列表的**索引**；工序基因是工件 ID，同一工件的第 N 次出现对应第 N 道工序。编码一个已有解时先对调度图排序形成可行工序顺序，不能顺利生成时回退到按工件排列的基因顺序。解码时把基因依次追加到选定机器序列，并重新建图；保证每个工件出现次数正确及机器选项合法，最终适应度是实际 makespan。`Decode(Encode(schedule))` 在机器分配不变时通常可恢复原顺序，具体解仍以解码后的图评价为准。

实现使用 `thread_local` 的持久状态，包含问题签名、种群、染色体适应度缓存、交叉/变异统计、累计代数和停滞计数；问题签名变化或种群为空时重建。单次调用将当前工作解编码注入存量种群，并以 `max(4, population_size)` 作为目标规模。缓存键为机器编码与工序编码的组合字符串，**仅在同一实例的进程内**复用。

初始种群包含：

- 当前 HLS 方案本身，保证遗传搜索不会丢失 incumbent；
- 约 `20%` 的 incumbent 局部扰动方案；
- 约 `40%` 的全局负载初始化方案：随机改变工件处理顺序，并按机器累计加工时间选择机器；
- 约 `20%` 的局部初始化方案：在每个工件内部按候选机器预计完成时间分配；
- 剩余位置使用随机机器选择和随机工序排序补齐。

初始化按染色体键去重，随机生成不足时继续在 incumbent 上做有限次扰动。若仍达不到目标规模，才会在初始化的最后一步复制首个个体以维持非空种群；后续代际选择会再次去重，因此不能简单假定种群始终有 20 个不同染色体。

每代遗传演化执行以下步骤：

1. 使用规模为 3 的锦标赛抽取父代，尽量避开重复父代下标；不保证两个父代的染色体一定不同。
2. 初始交叉概率为 `0.85`，采样足够多时由交叉成功率更新并限制在 `0.55～0.95`；机器编码和工序编码分别以 `0.65` 概率交叉，若两者都未被选中则强制交叉工序编码。
3. 两个子代分别保留与各自父代对应的另一条编码，不再丢弃第二父代的组合信息。
4. 按当前自适应变异率执行机器变异或局部工序变异。
5. 解码子代，通过调度图计算 makespan，并与父代共同进入候选集合。
6. 父代和子代按 makespan 排序，保留最优两个不同染色体，再对其余幸存者加最小结构距离筛选；根据停滞情况补充随机移民，最多保留 `population_size` 个方案。

机器编码交叉使用非空且非全集的随机位置子集交换。工序编码使用按工件集合划分的 POX 交叉，保持每个工件的出现次数。普通机器变异每次修改 1 至 3 道柔性工序，75% 的情况下从加工时间最短的两个其他机器选项中抽样，其余情况从全部其他选项抽样。申请变异时以 65% 概率先尝试关键工序变异：临时解码、获取零松弛关键工序，优先改其机器选项或与其他工件的关键基因交换；否则在普通机器/工序变异中二选一。普通工序变异执行不同工件基因的交换、插入或短区间反转，不打乱整条序列。

基础变异率为 `0.12`；连续 6 代无改进升至 `0.25`、约 20% 随机移民，12 代无改进升至 `0.45`、约 33% 随机移民。变异尝试至少 8 次后，以 `clamp(base_rate * (1.20 - 0.45 * 变异成功率), 0.05, 0.60)` 做自适应调整。交叉和变异尝试数达到 200 时各自计数和累计收益减半，避免过旧的统计长期主导参数。

每隔累计 10 代，或本次调用的最后一代，对子代中较好的个体做 memetic 局部强化：通常至多处理一个，调用的最后一代至多两个。偶数序号精英执行 1 轮 Greedy，奇数序号精英执行最多 6 轮 Tabu（禁忌长度固定为 25，且 `use_persistent_state=false`，不污染主 Tabu 的持续轨迹）。只有强化后确实比该个体更好才重新编码入选，所有嵌套调用共用 GA 的截止时间；达到截止时间时，末代强化可能不执行。

结构距离为两条染色体相同位置基因不同的数量除以所比较基因总数；除前两个精英外，优先要求候选与现存个体的距离均不少于 `0.06`。种群相对其最优个体的平均距离低于 `0.08`，或连续 4 次调用未获得新最好时，保留最多 `max(2, population_size / 5)` 个精英，以随机移民补齐、清空算子统计并提高下一轮变异基准。遗传方法内部只向外返回本次起点与整个调用期间找到的较优者；后续种群和缓存跨调用保存。

适应度计算使用染色体缓存避免重复解码完全相同的基因；关键变异或嵌套局部搜索仍会额外构造完整调度。遗传分支有独立的 `std::mt19937` 随机流，初始种群、父代抽样、交叉/变异及随机移民均使用这条流；每次批量运行是独立进程，不跨运行保留种群。

### 7.2 禁忌搜索

`TabuSearch` 分别维护当前轨迹 `current_schedule` 和**本次调用**的最优解 `global_best_schedule`。当前解可以接受有限幅度的非改进移动以跳出局部最优，最佳解只在 makespan 严格降低时更新，因此即使最后一次移动变差，返回值仍为本次调用找到的最好方案。其名字 `global_best_schedule` 是该方法内部的最佳值，外层 `output_schedule` 才是整个 HLS 运行的全局最好值。

默认 `use_persistent_state=true`：持久状态按问题签名保存当前非最优轨迹、各逆移动的禁忌到期迭代号、累计绝对迭代号和位置访问频率。若保存的轨迹比传入的起点差超过 `max(5, round(0.05 * 输入 makespan))`，则丢弃轨迹并清空旧短期禁忌；否则从该轨迹继续。GA 内嵌 Tabu 设置 `use_persistent_state=false`，使用独立临时状态，不影响主 Tabu 分支。更换问题签名时重置持久状态。

每轮围绕当前调度的关键工序生成邻域。`GetKeyProcess()` 使用零松弛识别关键工序；若方法内部停滞至少 6 次或处于扰动阶段，再补充 `slack <= max(6, min(20, Cmax / 40))` 的近关键工序。机器上连续两个**严格关键**工序且后者最早开始恰等于前者完工时形成紧致关键弧，连续弧组合为关键块；近关键工序不会被当作严格关键块成员。轮换工序列表起点，每轮机器迁移和顺序移动各最多处理 36 道候选关键工序。

每轮按如下顺序向 `NeighborEvaluator` 提交候选（相关停滞条件必须满足）：

1. **关键块移动**（至多 400）：块内部相邻交换，以及块首后移/块尾前移到关键块其他位置。
2. **路径引导**（至多 200，需停滞至少 4 次或处于扰动阶段且当前解与本次方法最好解不同）：向本次方法的最佳解靠近，一次将一条工序移动到引导解的对应机器和大致位置。这不是当前外层精英库之间的完整 path relinking。
3. **跨机迁移**（至多 1100）：只枚举该工序允许的其他机器，目标机器按该工序加工时间从短到长排序，每台目标机器枚举全部插入位置。
4. **关键工序同机双工序交换**（至多 250，需内部停滞至少 4 次）：同机两道工序换位，记录两道工序各自的禁忌属性。
5. **同机重插入**（至多 650）：把候选关键工序从原机器上移到该机器的其他位置。

各邻域的固定配额之和为 2600，同时还有“每轮最多精确评价 2600 个不同候选”的总上限与时间截止。生成候选先按机器序列签名去重，再精确重建图和计算 makespan；有环候选不参与选择。某邻域未用完的配额不会自动转给其他邻域；前序生成器遇到时间/总预算上限，后续邻域可能提前停止。

禁忌表记录工序原机器及旧前驱/后继位置对应的**逆移动属性**，不是保存完整调度字符串。双工序交换同时注册两道工序的逆属性。候选的目标位置/机器命中未过期属性时被禁止，除非 makespan 严格小于本次方法最佳值而满足 aspiration；候选即使未被禁忌，若比当前轨迹差过多也会被排除。允许变差的上限为 `max(3, round(0.02 * 本次方法最佳 makespan) + min(20, 2 * 内部停滞次数))`。若没有非禁忌可选解，则取未被变差阈值排除的最低 makespan 候选并清空短期禁忌，避免空邻域直接结束。

停滞至少 4 次后，在 `selection_score = makespan + frequency_weight * 已访问次数` 中施加访问频率惩罚，其中 `frequency_weight = max(0.1, best * 0.0005)`；否则不施加。扰动期维护最多 12 个可选候选，按评分排序，从前最多 6 个中随机选取，允许在有限范围内暂时变差。只在新解严格小于本次方法最佳值时重置内部停滞计数；否则继续当前轨迹。

`tabu_list_length` 是禁忌期限配置值。令 `size_tenure = max(7, round(sqrt(工序数)))`，中心 `center = max(5, (tabu_list_length + size_tenure) / 2)`，下界为 `max(3, center * 2 / 3)`，上界为 `max(下界, center * 4 / 3)`；方法内部停滞至少 5 次时上界再增加 `max(1, size_tenure / 2)`。实际期限在此区间随机抽取，原逆移动属性在 `absolute_iteration + tenure + 1` 到期。配置值不大于 0 时不注册逆移动禁忌。单次调用的内部重启阈值 `max(8, min(40, max(1, max_iter_count / 3)))`，在还有迭代次数时回到本次方法最好解，清空短期禁忌表，并执行 2～5 步扰动。每次实际选定的移动更新访问频率和当前轨迹；到绝对迭代数跨过 300 的倍数时将频率减半、去掉归零的记录。绝对迭代数与禁忌到期时间跨外层调用连续递增。

## 8. 可行性与评价闭环

所有搜索分支的实际可行性和 makespan 都回到同一张调度图；Greedy 与 Tabu 对单个候选典型执行以下闭环：

```text
改变机器选择或机器顺序
        |
        v
重建 ScheduleItemsToGraph
        |
        v
拓扑排序：同时检测是否有环并计算 makespan
        |
   无环才继续，有环则跳过/报错
        |
        v
按 makespan 选择或接受候选
```

`ScheduleItemsToGraph(..., flag=true)` 可同时设置图、工序列表、makespan 和故障率；Greedy/Tabu 候选一般用 `flag=false` 得到图后补齐工序列表，再调用 `CalculateTotalTime()`，仅在胜出时计算故障率。遗传解码走 `flag=true` 的统一路径；COPT 和启发式初始解同样转换为 `Schedule` 后评价。因此 COPT 目标值和方法内部候选最终都以 HLS 这套整数加工时间与图计算出的 makespan 为准。

## 9. 参数与实验输出

单次运行的主要参数默认值为：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `search_mode_param` | `0.5` | 遗传/禁忌相对于贪心的初始偏好 |
| `random_search_strategy_param` | `0.5` | 遗传/禁忌初始偏好中遗传搜索所占比例 |
| `max_repeat_count` | `100` | 三种搜索方法均未改进的完整轮数上限 |
| `max_iter_count` | `30` | 单次调用的基准迭代数，强化和贪心调用自动缩短 |
| `population_size` | `20` | 遗传搜索种群大小 |
| `tabu_list_length` | `25` | 动态禁忌期限的配置值，设为 0 可关闭禁忌限制 |
| `copt_time_limit_seconds` | `300` | COPT 初始解时间限制 |
| `total_time_limit_seconds` | `3600` | COPT 加 HLS 总时间限制 |
| `random_seed` | 当前 Unix 时间（秒） | 调度器、遗传、禁忌及启发式多起点的根随机种子 |

命令行位置参数依次为：

```text
./build/production_scheduling input_file search_mode random_strategy \
  max_repeat max_iter population_size tabu_length copt_seconds total_seconds random_seed
```

主程序只显式检查总时间 `total_time_limit_seconds > 0`；COPT 自身校验时间必须为有限正数，错误被外层异常捕获时会尝试启发式回退。设置 `copt_time_limit_seconds = total_time_limit_seconds` 意味着 COPT 可能耗尽 HLS 的全部时间，不能据此评价三种局部搜索的优化能力。

每次方法调用后输出方法名、秒数、模式（`intensification` / `adaptive` / `diversified`）、本次迭代上限及当前**全局最好** `Total time:`。结束时输出三种方法各自的 `calls`、`local_improvements`、`global_improvements`、累计 `relative_gain`（**局部**相对收益）和累计秒数；最后输出 `Final makespan:`、单独整数 makespan 和从 COPT 前开始的整数秒总耗时。

批量脚本 `run_brandimarte_experiments.sh` 默认以 **10 路并发**处理 `Mk01` 到 `Mk10`，每个实例默认 **1 次**；批量脚本中的 `RUNS` 才表示每个实例重复次数，不同于主程序单次方法调用的 `max_iter_count`。默认 `SEED_BASE=100000`，`seed = SEED_BASE + instance_index * 1000 + run_number`（run 从 1 开始）；`INSTANCES` 子集会改变 `instance_index`，比较实验时应固定实例列表与随机种子。脚本使用按 `PARALLEL_JOBS` 数量组成的批次等待，不保证两批之间无空闲。常用控制参数包括：

- `RUNS` / `START_RUN`：总次数及从第几次开始补跑（默认分别为 1 / 1）；补跑须使用同一个 `RUN_TIMESTAMP` 和现有日志目录。
- `PARALLEL_JOBS`：并发进程数，默认 10；**COPT 检查脚本**单独默认 5。
- `COPT_TIME` / `TOTAL_TIME`：每个进程的 COPT / 总优化秒数，默认 300 / 3600；分别映射主程序相应参数。
- `SEARCH_MODE`、`RANDOM_STRATEGY`、`MAX_REPEAT`、`MAX_ITER`、`POPULATION_SIZE`、`TABU_LENGTH`、`SEED_BASE`：覆盖主程序默认算法参数。
- `SKIP_COMPLETED=1`：已有日志含 `Final makespan:` 则跳过；`SKIP_COMPLETED=0`：重新运行并覆盖同名日志。
- `INSTANCES` / `AGGREGATE_INSTANCES`、`DATA_DIR`、`EXECUTABLE`、`LOG_DIR`、`RESULT_FILE`、`RUN_TIMESTAMP`：限定实例集合或覆盖输入、二进制、输出路径；默认时间目录由启动时的 `date +%Y%m%d_%H%M%S` 生成。

默认输出结构如下：

- `run_logs/<运行时间>/MkXX_runNN.txt`：一次运行的完整标准输出和错误输出；
- `run_logs/<运行时间>/result.csv`：`instance,run1,...,runN`，每个实例一行。脚本一开始若文件不存在，会填入 `PENDING` 占位；全部批次成功后才从日志提取 `Final makespan:` 并用临时文件原子替换最终 CSV，运行期间不会逐次更新该文件；
- `copt.csv`：`collect_copt_results.sh` 根据 `LOG_DIR` 和 `RUNS` 提取 COPT 初始 makespan，默认 `LOG_DIR` 为顶层 `run_logs/`，若日志位于时间目录须显式设置该变量；
- `copt_progress_checks/copt_progress.csv`：`check_copt_initial_solution.sh` 汇总的逐实例逐检查点记录，脚本写出字段 `instance,checkpoint_seconds,makespan,observed_elapsed_seconds,status`；现有文件若经过人工横向重排，不代表脚本原生的 CSV 结构；
- `copt_progress_checks/copt_final_results.csv`：`instance,final_makespan,status,copt_elapsed_seconds,log_file,progress_file`，对应 COPT 可行解或 `NA`；原始 incumbent 流另写到同目录 `logs/<instance>_incumbents.csv`，按 `elapsed_seconds,makespan` 记录。

`check_copt_initial_solution.sh` 默认每个实例 COPT 运行 3600 秒，每隔 300 秒从回调记录中取不晚于该检查点的最近 incumbent，最多 5 个实例并发；通过监听 `COPT initial makespan:` 或 `COPT initial solution unavailable:` 提前停止后续 HLS，使输出只反映 COPT。该脚本为分析初始解的独立实验，不能把回退 HLS 的 makespan 充作 COPT incumbent。

如果 COPT 没有产生可行 incumbent，日志会包含 `COPT initial solution unavailable:`，对应的 `copt.csv` 单元格记为 `NA`；此时日志中的 `HLS fallback initial makespan` 属于回退解，不属于 COPT 初始解。

## 10. 当前实现的边界

1. 当前最终目标是 makespan，机器故障率只被记录和计算，不参与主目标的加权优化。
2. 自适应收益统计只在单次程序运行内保留，不会跨越不同随机种子或实例学习参数。
3. 方法耗时和截止控制使用单调高精度时钟；最终兼容输出的总耗时仍按整数秒显示。
4. COPT 是初始解生成器；当实例较大或时间较短而 COPT 没有可行解时，程序会自动回退到启发式初始解，后续 HLS 仍可继续运行。
5. 当前 Greedy/Tabu 的候选多次复制调度并重建稠密图，尚无增量 makespan、统一的跨方法候选缓存或并行单实例搜索；批量脚本的并行是**不同进程/实验之间**的并行。
6. 精英库仅按签名去重且按加入顺序淘汰旧项；外层接纳等 makespan 工作解时只检查与现有工作解签名不同，**不**再次比较 Greedy 的结构次目标。遗传的结构距离筛选只用于种群，外层库没有基于距离的全局维护策略。
7. `l2s-design.md` 的学习引导候选、近邻状态记忆、第四种方法及模型训练/推理仍为后续设计，不属于当前 C++ 主程序。未经同实例、同总时限、同并发和同随机种子的实验，不应把其他仓库的 makespan 与 HLS 结果直接归因于某个邻域机制。
