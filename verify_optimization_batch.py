#!/usr/bin/env python3
"""Audit a completed ten-by-ten batch and generate its local handoff artifacts."""

import argparse
import csv
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess

import run_adaptive_brandimarte as batch


def audit(directory, reference):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if config["instances"] != batch.INSTANCES or config["runs"] != 10:
        raise ValueError("Expected all ten Brandimarte instances with ten runs each.")
    if state["phase"] != "COMPLETE" or len(state["tasks"]) != 100:
        raise ValueError("The complete 100-run experiment has not finished.")
    for artifact in config["artifacts"].values():
        path = (directory / artifact["snapshot"]).resolve()
        path.relative_to(directory)
        if batch.sha256(path) != artifact["sha256"]:
            raise ValueError("Changed artifact: %s" % path)
    with (directory / "seeds.csv").open(newline="") as handle:
        seeds = {(row["instance"], int(row["run"])): int(row["seed"]) for row in csv.DictReader(handle)}
    if len(seeds) != 100 or len(set(seeds.values())) != 100:
        raise ValueError("The seed manifest must contain 100 distinct experiment seeds.")
    values, logs = {}, {}
    parameter_log_names = ["search_mode_param", "random_search_strategy_param", "max_repeat_count", "max_iter_count",
                           "population_size", "tabu_list_length", "copt_time_limit_seconds", "total_time_limit_seconds"]
    for task in state["tasks"]:
        key = (task["instance"], task["run"])
        if key in values or task["status"] != "OK" or task["exit_code"] != 0 or key not in seeds:
            raise ValueError("Invalid, duplicate or incomplete task: %s" % task["label"])
        log = (directory / (task["label"] + ".txt")).read_text(encoding="utf-8")
        finals = batch.FINAL.findall(log)
        if len(finals) != 1 or int(finals[0]) != task["makespan"] or task["makespan"] <= 0:
            raise ValueError("Makespan/log mismatch: %s" % task["label"])
        seed_match = re.search(r"^random_seed:\s*(\d+)\s*$", log, re.MULTILINE)
        if not seed_match or int(seed_match[1]) != task["seed"] or task["seed"] != seeds[key]:
            raise ValueError("Seed/log mismatch: %s" % task["label"])
        for field, (parameter, *_) in zip(parameter_log_names, batch.SOLVER_PARAMETERS):
            match = re.search(r"^%s:\s*(\S+)\s*$" % field, log, re.MULTILINE)
            if not match or abs(float(match[1]) - float(config["parameters"][parameter])) > 1e-8:
                raise ValueError("Nonuniform parameter %s: %s" % (parameter, task["label"]))
        for method in ("Greedy Search", "Genetic Algorithm", "Tabu Search"):
            match = re.search(r"^%s summary: calls=(\d+)" % method, log, re.MULTILINE)
            if not match or int(match[1]) < 1:
                raise ValueError("Required method was not executed: %s %s" % (task["label"], method))
        for field in ("greedy_rl_enabled: true", "tabu_rl_enabled: true", "greedy_rl_status: model loaded",
                      "tabu_rl_status: Greedy model loaded for Tabu candidates", "batch_status=OK", "exit_code=0"):
            if field not in log:
                raise ValueError("Missing runtime evidence %s: %s" % (field, task["label"]))
        if task["instance"] in ("Mk03", "Mk08") and task["makespan"] < {"Mk03": 204, "Mk08": 523}[task["instance"]]:
            raise ValueError("Result contradicts known optimum: %s" % task["label"])
        values[key], logs[key] = task["makespan"], log
    with (directory / "result.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    expected = [["instance"] + ["run%d" % run for run in range(1, 11)]] + [
        [instance] + [str(values[(instance, run)]) for run in range(1, 11)] for instance in batch.INSTANCES]
    if rows != expected:
        raise ValueError("result.csv differs from the 100 authoritative logs.")
    reference_means = batch.reference_means(reference)
    comparisons = []
    for instance in batch.INSTANCES:
        scores = [values[(instance, run)] for run in range(1, 11)]
        mean = statistics.mean(scores)
        difference = mean - reference_means[instance]
        status = "BETTER" if difference < -1e-9 else "WORSE" if difference > 1e-9 else "TIE"
        comparisons.append({"instance": instance, "values": scores, "mean": mean,
                            "min": min(scores), "max": max(scores), "stddev": statistics.stdev(scores),
                            "l2s_mean": reference_means[instance], "difference": difference, "status": status})
    return config, state, expected, logs, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=batch.ROOT / "l2s_brandimarte.csv")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    directory, reference = args.run_dir.resolve(), args.reference.resolve()
    config, state, results, logs, comparisons = audit(directory, reference)
    output = (args.output_dir or directory.parent / "final").resolve()
    output.mkdir(parents=True, exist_ok=True)
    batch.write_csv(output / "result.csv", results)
    batch.write_csv(output / "comparison_l2s.csv", [["instance", "hls_mean", "l2s_mean", "hls_minus_l2s", "min", "max", "stddev", "status"]] + [
        [row["instance"], "%.4f" % row["mean"], "%.4f" % row["l2s_mean"], "%.4f" % row["difference"], row["min"], row["max"], "%.4f" % row["stddev"], row["status"]] for row in comparisons])
    for instance in batch.INSTANCES:
        text = "\n".join("===== %s run %02d =====\n%s" % (instance, run, logs[(instance, run)]) for run in range(1, 11))
        batch.atomic_text(output / (instance + ".txt"), text)
    hls_mean = statistics.mean(row["mean"] for row in comparisons)
    l2s_mean = statistics.mean(row["l2s_mean"] for row in comparisons)
    counts = {status: sum(row["status"] == status for row in comparisons) for status in ("BETTER", "TIE", "WORSE")}
    completed_at = max(task["finished_at"] for task in state["tasks"])
    goal_achieved = counts["WORSE"] == 0
    verdict = {"audited_runs": 100, "all_methods_executed": True, "both_rl_models_loaded": True,
               "result_csv_matches_logs": True, "distinct_seeds": 100, "hls_mean": hls_mean, "l2s_mean": l2s_mean,
               "counts": counts, "full_quality_goal_achieved": goal_achieved, "completed_at": completed_at,
               "reference_path": str(reference), "reference_sha256": batch.sha256(reference),
               "reference_matches_start_snapshot": batch.sha256(reference) == config["artifacts"]["l2s"]["sha256"],
               "comparisons": comparisons, "binary_sha256": config["artifacts"]["executable"]["sha256"]}
    batch.atomic_json(output / "audit.json", verdict)
    env = dict(config["parameters"], RUNS="10", INSTANCES=" ".join(batch.INSTANCES), MAX_PARALLEL="14",
               CPU_USAGE_LIMIT="0.95", CPU_RESERVE="0", CPUS_PER_JOB="1",
               EXECUTABLE=str(directory / config["artifacts"]["executable"]["snapshot"]),
               MODEL_PATH=str(directory / config["artifacts"]["model"]["snapshot"]), DATA_DIR=str(directory / "inputs"),
               L2S_FILE=str(directory / config["artifacts"]["l2s"]["snapshot"]),
               PAIRED_SEEDS_FILE=str(directory / "seeds.csv"),
               CANDIDATE_HLS_ENV=json.dumps(config.get("experimental_environment", {}), separators=(",", ":")))
    batch.atomic_text(output / "best-config.env", "\n".join("export %s=%s" % (key, shlex.quote(value)) for key, value in env.items()) + "\n")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(batch.ROOT), text=True).strip()
    worktree = subprocess.check_output(["git", "status", "--short"], cwd=str(batch.ROOT), text=True)
    batch.atomic_text(output / "source-state.txt", "head=%s\nbinary_sha256=%s\nmodel_sha256=%s\nsource_snapshot=%s\n\n%s" % (
        head, config["artifacts"]["executable"]["sha256"], config["artifacts"]["model"]["sha256"],
        directory.parent / "build-frozen/source", worktree))
    lines = ["# HLS 正式重复验证报告", "", "结束时间：%s。100 次全部通过日志、参数、seed、模型和三种必需方法调用审计。" % completed_at,
             "", "整体平均 HLS=%.4f，L2S=%.4f；%d 优于、%d 持平、%d 落后。完整质量目标：%s。" % (
                 hls_mean, l2s_mean, counts["BETTER"], counts["TIE"], counts["WORSE"], "达到" if goal_achieved else "未达到"),
             "", "| 实例 | HLS 十次均值 | L2S 十次均值 | 差值 | 最好 | 最差 | 结果 |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in comparisons:
        lines.append("| %s | %.2f | %.2f | %+.2f | %d | %d | %s |" % (row["instance"], row["mean"], row["l2s_mean"], row["difference"], row["min"], row["max"], row["status"]))
    lines += ["", "## 配置与解释", "", "同一冻结源码、同一全局参数、每实例 10 次独立随机 seed；每次总时限 600 秒，最高并发 14，每进程 1 个逻辑 CPU。初始 COPT 关闭，仅在 Greedy 内部用 COPT 求解有界联合邻域。上一轮固定版本使用 3600 秒，平均 177.34；本轮平均变化为 %+.4f，时限不同，不能把它描述为同预算的算法因果对照。" % (hls_mean - 177.34),
              "", "保留 Greedy、GA、Tabu、Guided 及两种 RL；新增动态工序分派、带合法 head/tail 界限的机器分配/顺序联合修复、停滞后间隔启用非关键交换伙伴、GA 工作量重分配、持续禁忌轨迹与统一阶段重启。所有变化均对全实例采用同一规则。",
              "", "验证：精确冻结源码通过联合修复与状态重置测试（含稀疏/完整图核对），Python RL 回归 3 项通过，脚本语法及 git diff 检查通过。原始失败/筛查候选见上一层 experiments.csv 和各候选目录，未用于拼接正式结果。",
              "", "原始逐次日志和不可变输入：%s。可复现实验入口为 run_optimized_hls.sh；best-config.env 保留本轮种子清单路径，重放时另设新的 LOG_DIR。" % directory]
    batch.atomic_text(output / "final-report.md", "\n".join(lines) + "\n")
    print(json.dumps({key: verdict[key] for key in ("audited_runs", "hls_mean", "l2s_mean", "counts", "full_quality_goal_achieved", "completed_at")}, ensure_ascii=False))
    print("Handoff directory: %s" % output)


if __name__ == "__main__":
    main()
