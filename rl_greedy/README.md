# HLS Greedy Reinforcement Learning

本目录是 HLS 自主实现的 Greedy 邻域强化学习模块，不导入、不复制也不依赖 L2S 仓库、模型或 checkpoint。借鉴的只有“策略从合法动作中提出候选，求解器精确评价候选”的分工。

## 运行结构

```text
完整可行调度
  -> HLS 规则生成同机重插入、跨机插入、同机交换动作
  -> PPO Actor 为动作打分并采样/排序
  -> 只保留 Top-K 与少量探索动作
  -> HLS C++ 重建调度图、过滤有环解、精确计算 makespan
  -> Greedy 从精确评价结果中选择 best-improvement
```

模型不负责判断机器兼容性、构造最终调度、检测环或预测最终 makespan。关闭 RL 或模型加载失败时，C++ 自动执行原始 Greedy 全邻域逻辑。

## 环境

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hls-rl
```

环境定义位于仓库根目录 `environment-rl.yml`。训练代码使用 PyTorch；C++ 推理只执行一个 16-32-1 的 `tanh` MLP，不依赖 Python、PyTorch、ONNX Runtime 或 GPU。

## 动作和特征

动作覆盖：

- 同机重插入；
- 跨兼容机器插入；
- 同机关键工序交换。

Actor 输入为 16 维特征：三类动作 one-hot、是否零松弛、归一化 slack、源/目标加工时间、加工时间收益、源/目标负载、负载差、源/目标位置、位置距离及源/目标序列长度。Python 与 C++ 使用同样的特征顺序和归一化公式。

## 训练

默认使用随机合成 FJSP，避免把 Brandimarte Mk01～Mk10 同时用于训练和最终评价：

```bash
python -m rl_greedy.train --epochs 200 --episodes-per-epoch 8 --horizon 30 --ppo-epochs 4 --ranking-epochs 2 --ranking-temperature 0.05 --hidden-size 64 --learning-rate 0.0002 --min-jobs 10 --max-jobs 20 --min-machines 4 --max-machines 15 --min-operations 3 --max-operations 15 --max-machine-options 6 --min-processing-time 1 --max-processing-time 20 --device cuda --seed 20260928 --checkpoint rl_greedy/models/greedy_policy_brandimarte.pt --export rl_greedy/models/greedy_policy_brandimarte.txt
```

输出：

- `rl_greedy/models/greedy_policy.pt`：可继续训练或供 Python 测试的 checkpoint；
- `rl_greedy/models/greedy_policy.txt`：C++ 原生推理模型。

也可以显式提供其他训练实例，`--instance` 可重复：

```bash
python -m rl_greedy.train --instance /path/train01.fjs --instance /path/train02.fjs --device cuda
```

快速验证训练链路：

```bash
python -m rl_greedy.train --epochs 1 --episodes-per-epoch 2 --horizon 3 --ppo-epochs 1 --ranking-epochs 1 --min-jobs 4 --max-jobs 5 --min-machines 3 --max-machines 4 --min-operations 3 --max-operations 5 --max-machine-options 3 --device cuda --checkpoint /tmp/hls_greedy_rl_smoke.pt --export /tmp/hls_greedy_rl_smoke.txt
```

## Python 对照测试

评估器从相同初始调度分别运行全邻域 Greedy 和 RL Top-K Greedy，并报告 makespan 与精确候选评价数：

```bash
python -m rl_greedy.evaluate datasets/Brandimarte_Data/Text/Mk01.fjs --model rl_greedy/models/greedy_policy.txt --top-k 96 --steps 10 --device cuda
```

正式模型在 Mk01～Mk10 上各测试一次并保存结果：

```bash
python -m rl_greedy.evaluate datasets/Brandimarte_Data/Text/Mk*.fjs --model rl_greedy/models/greedy_policy_brandimarte.txt --top-k 96 --dynamic-top-k --rule-fraction 0.25 --exploration 0.15 --temperature 1.0 --steps 10 --device cuda --seed 20260928 --output "rl_greedy/results/$(date +%Y%m%d_%H%M%S)/result.csv"
```

每个实例只运行一次，实例 seed 为 `基础 seed + 实例序号`。CSV 记录模型路径和 SHA-256、运行参数、初始/基线/RL makespan、精确评价次数和搜索步数；结束时终端输出实际 `result_file` 路径。

单元测试：

```bash
python -m pytest -q rl_greedy/tests
```

## C++ 开关

```bash
HLS_GREEDY_RL=1 \
HLS_GREEDY_RL_MODEL=rl_greedy/models/greedy_policy.txt \
HLS_GREEDY_RL_TOP_K=96 \
HLS_GREEDY_RL_DYNAMIC_TOP_K=1 \
HLS_GREEDY_RL_RULE_FRACTION=0.25 \
HLS_GREEDY_RL_FALLBACK_ON_MISS=1 \
HLS_GREEDY_RL_TEMPERATURE=1.0 \
HLS_GREEDY_RL_EXPLORATION=0.15 \
./build/production_scheduling ...
```

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `HLS_GREEDY_RL` | `0` | 非零且非 `false/off/no` 时请求启用 |
| `HLS_GREEDY_RL_MODEL` | `rl_greedy/models/greedy_policy.txt` | C++ Actor 模型路径 |
| `HLS_GREEDY_RL_TOP_K` | `96` | 每个候选阶段最多精确评价的策略候选数 |
| `HLS_GREEDY_RL_DYNAMIC_TOP_K` | `1` | 按工序数将 Top-K 自动扩展为 96/192/256/384 |
| `HLS_GREEDY_RL_RULE_FRACTION` | `0.25` | 候选池中由原 Greedy 规则保底的比例 |
| `HLS_GREEDY_RL_FALLBACK_ON_MISS` | `1` | RL 三阶段均无可接受解时补跑原始 Greedy 步骤 |
| `HLS_GREEDY_RL_TEMPERATURE` | `1.0` | 探索候选的 softmax 温度，限制为 0.05～20 |
| `HLS_GREEDY_RL_EXPLORATION` | `0.15` | Top-K 中按概率无放回采样的比例，限制为 0～0.5 |

程序启动时输出 `greedy_rl_requested`、`greedy_rl_enabled` 和 `greedy_rl_status`；结束时输出生成候选数和实际选择数。模型不存在、版本不匹配或权重不完整时，`greedy_rl_enabled=false`，原始 Greedy 继续运行。

## 在 Tabu 中暂时复用当前模型

当前 Greedy Actor 也可以在 Tabu 的关键块、路径引导、跨机迁移、双工序交换和同机重插入邻域生成后进行候选筛选。禁忌属性、aspiration、频率惩罚、允许变差和扰动选择仍由原 Tabu 精确逻辑执行。

```bash
HLS_TABU_RL=1 \
HLS_TABU_RL_MODEL=rl_greedy/models/greedy_policy_brandimarte.txt \
HLS_TABU_RL_TOP_K=256 \
HLS_TABU_RL_DYNAMIC_TOP_K=1 \
HLS_TABU_RL_RULE_FRACTION=0.30 \
HLS_TABU_RL_EXPLORATION=0.20 \
HLS_TABU_RL_MIN_PER_NEIGHBORHOOD=16 \
HLS_TABU_RL_FALLBACK_ON_MISS=1 \
./build/production_scheduling ...
```

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `HLS_TABU_RL` | `0` | 是否启用 Tabu RL 候选筛选 |
| `HLS_TABU_RL_MODEL` | 继承 `HLS_GREEDY_RL_MODEL` | 模型路径 |
| `HLS_TABU_RL_TOP_K` | `256` | 动态扩展前的候选基数 |
| `HLS_TABU_RL_DYNAMIC_TOP_K` | `1` | 按工序数扩展为至少 256/384/512/768 |
| `HLS_TABU_RL_RULE_FRACTION` | `0.30` | 原 Tabu 规则候选比例 |
| `HLS_TABU_RL_EXPLORATION` | `0.20` | 概率探索比例 |
| `HLS_TABU_RL_MIN_PER_NEIGHBORHOOD` | `16` | 每种非空 Tabu 邻域最低保留数 |
| `HLS_TABU_RL_FALLBACK_ON_MISS` | `1` | 筛选结果没有可接受候选时评价剩余邻域 |

当前模型是按 Greedy 即时改善目标训练的，不理解 Tabu 的长期受控变差收益。因此该接入属于候选评价加速实验，必须通过相同 seed、相同初始解和相同总时限的 Tabu RL 开/关实验判断效果，不能假定已有 Greedy 权重会提高 Tabu 最终 makespan。

## 当前边界

- 当前 Actor 是便于 C++ 原生推理的小型 MLP，不是图神经网络。
- Python 训练环境与 C++ 使用同一动作语义和特征，但调度图实现彼此独立；正式实验前仍需用固定实例检查候选和 makespan 一致性。
- 每个状态的完整邻域精确 makespan 同时用于 listwise 监督排序和 PPO 多步微调；训练日志输出 Recall@96、ImprovementRecall@96、Regret@96 和真实最优动作中位排名。
- PPO 训练时会精确评价完整动作集合以取得环境反馈，训练成本高于部署推理；形成环的动作保留在动作集中，被策略选中时得到惩罚且不改变当前解，与 C++“筛选后再验证”的顺序一致。
- 生成模型和实验结果默认不纳入 Git；正式结果应记录训练数据划分、seed、模型哈希和 HLS 参数。
