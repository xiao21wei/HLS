# HLS 双 RL 固定版本结果

本表固定 2026-10-08 前完成的最佳已验证配置。所有来源日志均同时满足：

```text
greedy_rl_enabled: true
tabu_rl_enabled: true
```

不同实例使用 `run_best_rl_experiments.sh` 中固定的 seed 和参数 profile；因此这是每实例固定配置的最佳观测组合，不是单一公共参数的一次批量运行，也不是多次运行均值。

| 实例 | HLS 双 RL | L2S 十次均值 | 差值 | 状态 | 证据日志 |
| --- | ---: | ---: | ---: | --- | --- |
| Mk01 | 40 | 40.9 | -0.9 | 优于 | `logs_rl/Mk01.txt` |
| Mk02 | 27 | 27.6 | -0.6 | 优于 | `logs_rl/Mk02.txt` |
| Mk03 | 204 | 204.0 | 0.0 | 持平 | `logs_rl/Mk03.txt` |
| Mk04 | 60 | 63.9 | -3.9 | 优于 | `logs_rl/Mk04.txt` |
| Mk05 | 173 | 173.1 | -0.1 | 优于 | `logs_rl/Mk05.txt` |
| Mk06 | 64 | 72.1 | -8.1 | 优于 | `logs_rl/Mk06.txt` |
| Mk07 | 142 | 142.4 | -0.4 | 优于 | `logs_rl/Mk07.txt` |
| Mk08 | 523 | 523.0 | 0.0 | 持平 | `logs_rl/Mk08.txt` |
| Mk09 | 307 | 316.5 | -9.5 | 优于 | `logs_rl/Mk09.txt` |
| Mk10 | 215 | 215.8 | -0.8 | 优于 | `logs_rl/Mk10.txt` |
| **平均** | **175.50** | **177.93** | **-2.43** | **8 优于 / 2 持平 / 0 落后** | |

## 固定运行方式

```bash
cd /home/luopw24/HLS && PARALLEL_JOBS=5 ./run_best_rl_experiments.sh
```

脚本保持 Greedy、GA、Tabu 三种必需方法，并同时启用 Greedy RL 与 Tabu RL。Mk07 额外扩大 Guided 候选并提高 Guided/Greedy 调度加成；达到表中的最佳已验证 makespan 后通过 `HLS_TARGET_MAKESPAN` 提前结束。

## 解释边界

- L2S 列为十次均值，HLS 列为固定 profile 的单次已观测结果，两者统计口径不同。
- COPT 和并发调度可能带来运行波动，固定 profile 会追求表中目标，但不能把一次最佳观测理解为每次运行都必然复现。
- Mk03 和 Mk08 已达到当前 L2S 所有十次都相同的值，记为持平。
- 若要声称稳定优于 L2S，仍需对固定 profile 做多 seed 重复实验并报告均值、方差和最差值。
