# `/home/luopw24/fjsp` 多目标 FJSP 深度强化学习设计

## 1. 方法定位

本文对应的实现目录是 `/home/luopw24/fjsp`。训练入口为 `train.py`，环境为 `env/fjsp_env.py`，HGNN 策略和 PPO 更新位于 `PPO_model.py`。

该实现解决带机器选择、工序先后、作业就绪时间、交期、运输时间和能耗约束的多目标柔性作业车间调度问题（MO-FJSP）。三个目标是：

1. 最大完工时间（makespan）；
2. 加工与运输总能耗（energy）；
3. 作业提前或延期交付惩罚（penalty）。

训练阶段把三个目标的相对改进加权成一个标量 PPO 奖励；测试阶段保留三维目标值，并使用采样、非支配排序、膝点和超体积贡献等方法选择折中解。

## 2. 问题模型与多目标数据

一个实例包含 `J` 个作业、`M` 台机器和 `N` 道工序。每道工序只能在兼容机器集合中选择一台机器，不同机器对应不同加工时间和加工能耗；同一作业内的工序必须按顺序执行。相邻工序发生机器切换时，还会产生运输时间和运输能耗。

每个作业额外包含权重、就绪时间、交期、提前惩罚系数和延期惩罚系数。`env/load_data.py` 从 `raw` 文件读取加工时间、可行机器和工序图，从 `add` 文件读取加工能耗、作业属性、运输时间和运输能耗；`env/case_generator.py` 可以生成同样格式的随机实例。

因此，多目标实例的来源可以概括为：

```text
基础 FJSP = 工序顺序 + 可选机器 + 加工时间
附加数据 = 就绪/交期/惩罚 + 加工能耗 + 运输时间/能耗
```

在基础 FJSP 的每个可行工序—机器组合上补充能耗，在每个作业上补充交期属性，在机器对之间补充运输属性，就得到目录中使用的多目标 FJSP。

## 3. 环境输入与状态

环境主要维护以下张量：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `proc_times_batch` | `[B,N,M]` | 加工时间 |
| `ope_ma_adj_batch` | `[B,N,M]` | 工序—机器可行关系 |
| `ope_pre_adj_batch` | `[B,N,N]` | 工序前驱图 |
| `ope_sub_adj_batch` | `[B,N,N]` | 工序后继图 |
| `proc_energy_batch` | `[B,N,M]` | 加工能耗 |
| `job_feat_batch` | `[B,J,5]` | 权重、就绪时间、交期、提前/延期惩罚 |
| `transport_time_batch` | `[B,M,M]` | 运输时间 |
| `transport_energy_batch` | `[B,M,M]` | 运输能耗 |

当前 `config.json` 的工序特征维度为 14，依次包含：已调度标记、可选机器数、平均加工时间、作业剩余工序数、预计完成时间、开始时间、作业权重、就绪时间、交期、提前惩罚系数、延期惩罚系数、交付惩罚估计、累计加工/运输能耗和作业末工序完成能耗。

机器特征维度为 5：仍可处理的工序数、机器可用时间、机器利用率、平均运输时间和平均运输能耗。状态还包含正在加工/已完成掩码、每个作业当前工序、当前时间和部分调度记录。

## 4. 动作与约束转移

策略动作是三行张量：

```text
actions[0] = operation index
actions[1] = machine index
actions[2] = job index
```

`HGNNScheduler.act` 先选择作业和机器，再根据 `ope_step_batch` 找到该作业当前待调度工序。合法动作必须满足：作业未完成且未被占用、机器空闲、工序允许在该机器上加工。

环境执行动作时删除该工序在其他机器上的候选边，更新作业/机器可用时间、工序开始和结束时间、工序图、机器利用率、交期惩罚和累计能耗。若没有可执行的工序—机器组合，`if_no_eligible` 与 `next_time` 将时间推进到下一个完成事件。所有作业完成后 episode 结束；`validate_gantt` 检查工序顺序、机器重叠、加工时长和未调度工序。

## 5. 三个目标的计算方法

### 5.1 最大完工时间

对所有工序完成时间取最大值：

```text
Cmax = max_o(C_o)
```

开始时间包含作业就绪时间、前序工序约束、机器可用时间和运输影响。

### 5.2 总能耗

若工序 `o` 分配到机器 `m`，其加工能耗为 `E_proc(o,m)`。同一作业的相邻工序从机器 `m_prev` 切换到 `m` 时增加运输能耗 `E_trans(m_prev,m)`：

```text
E_job(j) = sum_o_in_j E_proc(o, machine(o))
         + sum_switches_in_j E_trans(previous_machine, machine)
E_total = sum_j E_job(j)
```

未完成工序的状态特征使用兼容机器平均能耗估计，动作确定机器后替换为真实加工能耗。

### 5.3 提前/延期惩罚

令作业 `j` 的末工序完成时间为 `C_j`，交期为 `d_j`，权重为 `w_j`，提前和延期单位惩罚分别为 `p_j^e`、`p_j^t`：

```text
early_j = max(d_j - C_j, 0)
tardy_j = max(C_j - d_j, 0)
P_j = w_j * (p_j^e * early_j + p_j^t * tardy_j)
P_total = sum_j P_j
```

## 6. 加权多目标奖励

环境保存上一步的三个目标值，每次动作后计算相对改进：

```text
delta_makespan = (old_makespan - new_makespan) / old_makespan
delta_energy   = (old_energy   - new_energy)   / old_energy
delta_penalty  = (old_penalty  - new_penalty)  / old_penalty
```

然后按 `config.json` 的权重融合：

```text
reward = (w_makespan * delta_makespan
        + w_energy   * delta_energy
        + w_penalty  * delta_penalty)
         / (w_makespan + w_energy + w_penalty)
```

默认三项权重均为 1.0。因此训练是**加权标量化的多目标 PPO**，而不是通过偏好向量训练一族策略。完整 Pareto 前沿主要在测试阶段的候选选择中产生。

## 7. HGNN 策略网络

`PPO_model.py:HGNNScheduler` 使用异构图神经网络编码调度状态：

1. `GATedge` 根据工序—机器关系更新机器嵌入；
2. `MLPs` 分别聚合机器邻居、工序前驱、工序后继和工序自身特征；
3. 对机器和工序嵌入做平均池化，得到全局机器/工序表示；
4. 对每个候选作业—机器组合拼接当前作业工序嵌入、机器嵌入和两个全局嵌入；
5. Actor MLP 为每个候选动作生成分数，并用合法动作掩码屏蔽非法组合；
6. Critic MLP 根据全局表示预测状态价值。

当前配置使用两轮 HGNN、机器和工序嵌入维度 16、工序隐藏层维度 128，Actor 和 Critic 都是多层 MLP。

## 8. PPO 训练流程

训练入口是 `train.py`：

1. 读取或生成一批 FJSP 实例并创建 `FJSPEnv`；
2. 用 `policy_old.act` 按动作分布采样工序—机器动作；
3. 执行 `env.step`，记录状态、动作 log probability、奖励和终止标志；
4. 一个更新周期结束后，`PPO.update` 计算折扣回报并标准化；
5. 使用新旧策略概率比构造 clipped PPO 策略损失，同时加入价值损失和熵正则；
6. 用 Adam 更新策略，再将 `policy` 参数同步给 `policy_old`；
7. 定期在验证集上分别记录 makespan、energy 和 penalty，并保存模型与 Excel 曲线。

PPO 主要参数来自 `config.json`：学习率 `2e-4`、折扣因子 `1.0`、更新轮数 `K_epochs=3`、裁剪系数 `0.2`、价值损失系数 `0.5`、熵系数 `0.01`。默认 batch 中并行训练 20 个实例。

## 9. 测试阶段的多目标解选择

`test.py` 对每个候选排程记录三元目标：

```text
(makespan, energy, penalty)
```

支持的选择流程包括：

- DRL-G：每一步选择最大概率动作；
- DRL-S：复制实例并按策略分布采样多个排程；
- FIFO、MOR、MWKR 等规则基线；
- NSGA-II、NSGA-III 和优先级染色体启发式；
- 快速非支配排序，提取 Pareto 前沿；
- 膝点选择，选取归一化 Pareto 前沿上的折中解；
- hypervolume 贡献选择，保留对三目标覆盖范围贡献较大的解。

采样模式默认用相对 FIFO 的三目标距离选择候选解。需要区分的是：训练阶段的奖励已经把三个目标标量化，完整 Pareto 前沿主要在测试候选选择阶段产生。

## 10. 代码模块映射

| 文件 | 作用 |
| --- | --- |
| `env/fjsp_env.py` | FJSP 环境、状态转移、三目标计算和可行性检查 |
| `env/load_data.py` | raw/add 实例及能耗、交期、运输数据解析 |
| `env/case_generator.py` | 随机 FJSP 与附加数据生成 |
| `PPO_model.py` | HGNN、Actor、Critic、Memory 和 PPO 更新 |
| `graph/hgnn.py` | 异构图注意力和消息传递层 |
| `train.py` | PPO 训练、验证和模型保存 |
| `validate.py` | 独立验证 |
| `test.py` | DRL 推理、规则基线和多目标候选选择 |
| `config.json` | 实例规模、特征维度、PPO 参数和目标权重 |

## 11. 运行与注意事项

```bash
cd /home/luopw24/fjsp
python train.py
python test.py
```

训练与测试时，`ope_feat_dim`、`ma_feat_dim`、`in_size_ope` 和 `in_size_ma` 必须与环境特征维度一致；当前配置分别为 14 和 5。若数据缺少 `add` 文件中的能耗、交期或运输字段，三目标扩展无法正确计算。

该实现的数据流是：

```text
FJSP 原始数据 + 能耗/交期/运输数据
        ↓
工序图、机器图和动态调度特征
        ↓
HGNN 编码 + 合法工序—机器动作掩码
        ↓
PPO 学习调度策略
        ↓
makespan / energy / penalty 加权奖励
        ↓
采样、Pareto、膝点或 HV 选择多目标方案
```

## 12. 从基础 FJSP 扩充多目标数据集

### 12.1 文件组织：raw 与 add 一一配对

当前数据集采用“基础实例 + 附加实例”的双文件结构。以 `data_test1/1508` 为例：

```text
data_test1/1508/raw/15j_8m_001.fjs
data_test1/1508/add/15j_8m_001.fjs
```

两个文件必须同名。`raw` 文件保存原始 FJSP 信息：作业数、机器数、每个作业的工序数、每道工序可选机器及加工时间。`add` 文件保存与 raw 中工序顺序严格对应的多目标属性。环境在读取 raw 文件时把路径中的 `raw` 替换为 `add`，因此不会改变原始加工时间和机器可行关系，只是为同一个实例加载附加目标数据。

### 12.2 raw 文件中的基础 FJSP

基础 FJSP 每道工序记录若干个 `(machine_id, processing_time)` 对。多目标扩充必须复用 raw 的：

- 作业顺序；
- 每个作业的工序数量；
- 每道工序的可选机器数量；
- 每道工序的机器 ID 顺序。

这样 `add` 文件中的加工能耗才能填入与 raw 相同的 `[operation, machine]` 位置。`load_fjs` 根据 raw 建立工序前驱图、后继图、工序—机器邻接矩阵和作业归属关系。

### 12.3 加工能耗的扩充

`add` 文件的前 `J` 行与 raw 的 `J` 个作业行结构相同，但每个可选项的第二个数不再是加工时间，而是加工能耗：

```text
作业工序数  每道工序的可选机器数  机器ID 能耗  机器ID 能耗 ...
```

`env/load_data.py:energy_detec` 按 raw 的工序偏移和机器 ID，将这些能耗填入 `matrix_proc_energy[N,M]`。不可行的工序—机器位置保持为 0。环境执行动作后，对已选机器使用真实能耗；对尚未调度的工序使用兼容机器平均能耗作为状态估计。

### 12.4 作业交期属性的扩充

加工能耗行之后追加 `J` 行，每行五个数：

```text
weight  ready_time  due_time  earliness_cost  tardiness_cost
```

它们分别表示作业权重、就绪时间、交期、提前交付单位惩罚和延期交付单位惩罚。加载后形成 `job_feat_batch[B,J,5]`，并映射到每个作业的全部工序特征中。

当前合成器 `CaseGenerator.get_case` 的默认生成规则是：

```text
weight          ~ randint(1, 5)
ready_time     ~ randint(0, 20)
earliness_cost ~ randint(1, 10)
tardiness_cost ~ randint(1, 10)
due_time        = ready_time + 1.5 * sum(mean_processing_time_of_job)
```

因此交期不是从原始 FJSP 文件推断出来的真实业务数据，而是利用原始实例的工序加工时间规模生成的合成交期；交期大约是就绪时间加上该作业平均加工工作量的 1.5 倍。

### 12.5 运输时间和运输能耗的扩充

作业属性行之后追加两个 `M × M` 矩阵：先是运输时间矩阵，再是运输能耗矩阵。矩阵主对角线为 0，非对角线表示从机器 `m1` 切换到机器 `m2` 的代价。`CaseGenerator` 生成对称矩阵：

```text
transport_time[m1,m2]   ~ randint(1, 3),  m1 != m2
transport_energy[m1,m2] ~ randint(1, 4),  m1 != m2
```

对角线保持为 0，并令 `T[m1,m2] = T[m2,m1]`、`E[m1,m2] = E[m2,m1]`。加载器把矩阵恢复为 `[M,M]` 张量；环境只有在相邻工序机器不同的时候才把对应运输时间/能耗计入调度。

### 12.6 当前仓库数据的生成流程

`utils/create_ins.py` 调用 `CaseGenerator` 生成 100 个实例。当前设置为 15 个作业、8 台机器，每个作业的工序数在 `0.8M` 到 `1.2M` 的整数范围内随机生成，即通常为 6～9 道工序。对每个实例，生成器在同一次调用中完成以下步骤：

1. 随机确定每个作业的工序数量；
2. 为每道工序随机确定 1～`M` 个可选机器；
3. 为每道工序采样平均加工时间，并在其 ±20% 范围内生成各机器加工时间；
4. 把机器选择和加工时间写入 `raw/<name>.fjs`；
5. 对相同的工序—机器选项生成 4～15 的加工能耗；
6. 生成作业权重、就绪时间、交期和提前/延期惩罚；
7. 生成对称的运输时间与运输能耗矩阵；
8. 把附加字段写入同名的 `add/<name>.fjs`。

训练时并不是从磁盘逐个读取 raw/add 文件，而是每隔 `parallel_iter` 次调用同一个 `CaseGenerator` 在线生成一批实例；验证和测试则读取已经落盘的 raw/add 配对文件。由此，基础 FJSP 的调度拓扑保持不变，多目标属性作为与实例绑定的附加层加入。

### 12.7 将已有基础 FJSP 扩展为多目标数据集的要求

如果从外部基础 FJSP 数据集扩充，应对每个 raw 文件生成一个同名 add 文件，并遵守以下顺序：

```text
J 个加工能耗行
J 个作业属性行
M 行运输时间矩阵
M 行运输能耗矩阵
```

加工能耗行必须与 raw 的工序数量、机器 ID 和机器选项顺序一致；作业属性行必须与 raw 的作业顺序一致；两个运输矩阵必须是 `M × M`。否则 `load_fjs` 虽然可能成功读取，但能耗会映射到错误的工序/机器，最终 makespan、energy 和 penalty 三个目标将不再对应同一个基础 FJSP 实例。
