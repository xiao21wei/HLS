#!/usr/bin/env python3
"""Run paired core-module ablations through the SmartLab HLS runner."""

import argparse
import csv
import json
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
from datetime import datetime

import run_adaptive_brandimarte as batch
import run_smartlab_hls as smartlab


ROOT = Path(__file__).resolve().parent

VARIANTS = {
    "full": {},
    "no_greedy": {
        "HLS_ENABLE_GREEDY": "0",
        "HLS_GREEDY_RL": "0",
    },
    "no_genetic": {
        "HLS_ENABLE_GENETIC": "0",
    },
    "no_tabu": {
        "HLS_ENABLE_TABU": "0",
        "HLS_TABU_RL": "0",
    },
    "no_guided": {
        "HLS_ENABLE_GUIDED_SEARCH": "0",
    },
    "no_rl": {
        "HLS_GREEDY_RL": "0",
        "HLS_TABU_RL": "0",
    },
    "greedy_rl_only": {
        "HLS_GREEDY_RL": "1",
        "HLS_TABU_RL": "0",
    },
    "tabu_rl_only": {
        "HLS_GREEDY_RL": "0",
        "HLS_TABU_RL": "1",
    },
    "fixed_selector": {
        "HLS_ADAPTIVE_METHOD_SELECTION": "0",
    },
    "no_restarts": {
        "HLS_ENABLE_RESTARTS": "0",
    },
    "no_dispatch_initial": {
        "HLS_DISPATCH_INITIAL": "0",
    },
    "no_greedy_repair": {
        "HLS_GREEDY_FLEXIBLE_REPAIR": "0",
        "HLS_GREEDY_REPAIR_RELAYS": "0",
        "HLS_GREEDY_ROUTE_REPAIR": "0",
        "HLS_ROUTE_AWARE_METHOD_SELECTION": "0",
    },
    "no_tabu_continuity": {
        "HLS_TABU_CONTINUOUS_TRAJECTORY": "0",
    },
    "no_ga_enhancement": {
        "HLS_GA_WORKLOAD_REDISTRIBUTION": "0",
    },
}

DEFAULT_VARIANTS = (
    "full",
    "no_greedy",
    "no_genetic",
    "no_tabu",
    "no_guided",
    "no_rl",
    "greedy_rl_only",
    "tabu_rl_only",
    "fixed_selector",
    "no_restarts",
    "no_dispatch_initial",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", default=list(DEFAULT_VARIANTS),
        choices=tuple(VARIANTS), help="Ablations to execute sequentially."
    )
    parser.add_argument(
        "--families", nargs="+", default=["brandimarte"],
        choices=("brandimarte",)
    )
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--total-time", type=int, default=3600)
    parser.add_argument("--copt-time", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=14)
    parser.add_argument("--cpus-per-job", type=int, default=1)
    parser.add_argument("--cpu-usage-limit", type=float, default=0.95)
    parser.add_argument("--reserve-cpus", type=float, default=0.0)
    parser.add_argument("--job-memory-gb", type=float, default=2.0)
    parser.add_argument("--reserve-memory-gb", type=float, default=4.0)
    parser.add_argument("--check-interval", type=float, default=5.0)
    parser.add_argument("--ramp-up", type=int, default=2)
    parser.add_argument("--process-report-interval", type=float, default=300.0)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed-file")
    parser.add_argument("--executable", default=os.environ.get(
        "EXECUTABLE", str(ROOT / "build/ablation/production_scheduling")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.total_time < 1 or args.copt_time < 0:
        parser.error("runs and total-time must be positive; copt-time must be nonnegative")
    if args.copt_time > args.total_time:
        parser.error("copt-time cannot exceed total-time")
    if args.max_parallel < 1 or args.cpus_per_job < 1:
        parser.error("max-parallel and cpus-per-job must be positive")
    if not 0.1 <= args.cpu_usage_limit <= 1.0 or args.reserve_cpus < 0:
        parser.error("invalid CPU resource settings")
    if args.job_memory_gb <= 0 or args.reserve_memory_gb < 0:
        parser.error("invalid memory resource settings")
    if (args.check_interval < 0.1 or args.ramp_up < 1 or
            args.process_report_interval < 1):
        parser.error("invalid scheduler timing settings")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique")
    if len(set(args.families)) != len(args.families):
        parser.error("families must be unique")
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output-dir")
    args.executable = str(Path(args.executable).resolve())
    if not args.dry_run and (
        not Path(args.executable).is_file() or
        not os.access(args.executable, os.X_OK)
    ):
        parser.error("executable is missing or not executable: %s" % args.executable)
    return args


def selected_instances(families, requested_labels):
    discovered = smartlab.discover_instances(tuple(families))
    if requested_labels:
        requested = set(requested_labels)
        available = {
            label for family in families for label, _ in discovered[family]
        }
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError("unknown instance labels: %s" % " ".join(unknown))
        for family in families:
            discovered[family] = [
                item for item in discovered[family] if item[0] in requested
            ]
            if not discovered[family]:
                raise ValueError("no selected instances remain in %s" % family)
    return discovered


def expected_seed_keys(discovered, families, runs):
    return [
        (family, label, run)
        for family in families
        for label, _ in discovered[family]
        for run in range(1, runs + 1)
    ]


def read_seed_file(path, expected):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"family", "instance", "run", "seed"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("seed file must contain family, instance, run and seed")
        seeds = {}
        for row in reader:
            key = (row["family"], row["instance"], int(row["run"]))
            if key in seeds:
                raise ValueError("duplicate seed key: %s" % (key,))
            seed = int(row["seed"])
            if not 0 <= seed <= 4294967295:
                raise ValueError("seed outside uint32 range: %s" % seed)
            seeds[key] = seed
    if set(seeds) != set(expected):
        raise ValueError("seed file task set does not match this ablation plan")
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("seed values must be unique")
    return seeds


def prepare_seeds(path, expected, source=None):
    if source:
        seeds = read_seed_file(source, expected)
    elif path.is_file():
        seeds = read_seed_file(path, expected)
    else:
        used = set()
        seeds = {}
        for key in expected:
            seed = secrets.randbits(32)
            while seed in used:
                seed = secrets.randbits(32)
            used.add(seed)
            seeds[key] = seed
    rows = [["family", "instance", "run", "seed"]] + [
        [family, instance, run, seeds[(family, instance, run)]]
        for family, instance, run in expected
    ]
    batch.write_csv(path, rows)
    return seeds


def phase_complete(directory):
    state_path = directory / "state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get("phase") == "COMPLETE" and all(
        task.get("status") == "OK" for task in state.get("tasks", [])
    )


def read_numeric_results(path, runs):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    expected_header = ["instance"] + ["run%d" % run for run in range(1, runs + 1)]
    if not rows or rows[0] != expected_header:
        raise ValueError("unexpected result header: %s" % path)
    result = {}
    for row in rows[1:]:
        values = [float(value) for value in row[1:]]
        if len(values) != runs or not all(value > 0 for value in values):
            raise ValueError("incomplete result row in %s: %s" % (path, row[0]))
        result[row[0]] = values
    return result


def write_aggregate(root, variants, families, runs):
    rows = [[
        "variant", "family", "instance", "runs", "mean", "min", "max",
        "stddev", "full_mean", "delta_vs_full",
    ]]
    all_results = {}
    for variant in variants:
        variant_results = {}
        for family in families:
            path = root / variant / "outputs" / ("HLS_%s.csv" % family)
            if path.is_file():
                variant_results[family] = read_numeric_results(path, runs)
        if variant_results:
            all_results[variant] = variant_results
    full = all_results.get("full", {})
    for variant in variants:
        for family in families:
            for instance, values in all_results.get(variant, {}).get(family, {}).items():
                mean = statistics.mean(values)
                full_values = full.get(family, {}).get(instance)
                full_mean = statistics.mean(full_values) if full_values else None
                rows.append([
                    variant, family, instance, len(values), "%.4f" % mean,
                    min(values), max(values),
                    "%.4f" % statistics.stdev(values) if len(values) > 1 else "0.0000",
                    "%.4f" % full_mean if full_mean is not None else "",
                    "%.4f" % (mean - full_mean) if full_mean is not None else "",
                ])
    batch.write_csv(root / "ablation_summary.csv", rows)


def run_variant(args, root, seed_path, variant):
    variant_root = root / variant
    run_directory = variant_root / "run"
    output_directory = variant_root / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)
    if phase_complete(run_directory):
        print("[%s] already complete; skipping" % variant, flush=True)
        return
    resume = run_directory.joinpath("config.json").is_file()
    if resume and not args.resume:
        raise ValueError(
            "incomplete variant exists; rerun with --resume: %s" % run_directory
        )
    environment = os.environ.copy()
    environment.update({
        "EXECUTABLE": args.executable,
        "PAIRED_SEEDS_FILE": str(seed_path),
        "LOG_DIR": str(run_directory),
        "OUTPUT_DIR": str(output_directory),
        "OVERWRITE_OUTPUT": "1",
        "RUNS": str(args.runs),
        "COPT_TIME": str(args.copt_time),
        "TOTAL_TIME": str(args.total_time),
        "MAX_PARALLEL": str(args.max_parallel),
        "CPUS_PER_JOB": str(args.cpus_per_job),
        "CPU_USAGE_LIMIT": str(args.cpu_usage_limit),
        "CPU_RESERVE": str(args.reserve_cpus),
        "JOB_MEMORY_GB": str(args.job_memory_gb),
        "MEMORY_RESERVE_GB": str(args.reserve_memory_gb),
        "CHECK_INTERVAL": str(args.check_interval),
        "RAMP_UP": str(args.ramp_up),
        "PROCESS_REPORT_INTERVAL": str(args.process_report_interval),
        "SMARTLAB_HLS_ENV": json.dumps(VARIANTS[variant], sort_keys=True),
        "RESUME": "1" if resume else "0",
    })
    command = [
        sys.executable, "-u", str(ROOT / "run_smartlab_hls.py"),
        "--families",
    ] + list(args.families)
    if args.instances:
        command += ["--instances"] + list(args.instances)
    command += [
        "--runs", str(args.runs),
        "--max-parallel", str(args.max_parallel),
        "--cpus-per-job", str(args.cpus_per_job),
        "--cpu-usage-limit", str(args.cpu_usage_limit),
        "--reserve-cpus", str(args.reserve_cpus),
        "--job-memory-gb", str(args.job_memory_gb),
        "--reserve-memory-gb", str(args.reserve_memory_gb),
        "--check-interval", str(args.check_interval),
        "--ramp-up", str(args.ramp_up),
        "--process-report-interval", str(args.process_report_interval),
    ]
    print("[%s] overrides=%s" % (
        variant, json.dumps(VARIANTS[variant], sort_keys=True)
    ), flush=True)
    result = subprocess.run(command, cwd=str(ROOT), env=environment)
    if result.returncode != 0:
        raise RuntimeError("variant %s exited with code %d" % (
            variant, result.returncode
        ))


def main():
    args = parse_args()
    discovered = selected_instances(args.families, args.instances)
    expected = expected_seed_keys(discovered, args.families, args.runs)
    if args.dry_run:
        print("Variants: %s" % " ".join(args.variants))
        print("Families: %s" % " ".join(args.families))
        print("Instances: %d; runs: %d; tasks/variant: %d; total tasks: %d" % (
            sum(len(discovered[family]) for family in args.families),
            args.runs, len(expected), len(expected) * len(args.variants)
        ))
        print("Time/solver: %ds; COPT: %ds; max parallel: %d" % (
            args.total_time, args.copt_time, args.max_parallel
        ))
        print(
            "Resources: CPU limit=%.2f; reserve CPUs=%.1f; CPUs/job=%d; "
            "memory/job=%.1fGiB; memory reserve=%.1fGiB; check=%.1fs; "
            "ramp-up=%d; process report=%.1fs" % (
                args.cpu_usage_limit, args.reserve_cpus, args.cpus_per_job,
                args.job_memory_gb, args.reserve_memory_gb,
                args.check_interval, args.ramp_up,
                args.process_report_interval
            )
        )
        return 0


    stamp = datetime.now(batch.CHINA_TIME).strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir or (
        ROOT / "ablation_runs" / ("core_" + stamp)
    )).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError("output directory is not empty; use --resume: %s" % root)
    root.mkdir(parents=True, exist_ok=True)
    seed_path = root / "paired_seeds.csv"
    prepare_seeds(seed_path, expected, args.seed_file)
    manifest = {
        "schema": 1,
        "created_at": batch.now(),
        "variants": args.variants,
        "variant_overrides": {name: VARIANTS[name] for name in args.variants},
        "families": args.families,
        "instances": {
            family: [label for label, _ in discovered[family]]
            for family in args.families
        },
        "runs": args.runs,
        "total_time": args.total_time,
        "copt_time": args.copt_time,
        "max_parallel": args.max_parallel,
        "cpus_per_job": args.cpus_per_job,
        "cpu_usage_limit": args.cpu_usage_limit,
        "reserve_cpus": args.reserve_cpus,
        "job_memory_gb": args.job_memory_gb,
        "reserve_memory_gb": args.reserve_memory_gb,
        "check_interval": args.check_interval,
        "ramp_up": args.ramp_up,
        "process_report_interval": args.process_report_interval,
        "executable": args.executable,
        "executable_sha256": batch.sha256(args.executable),
        "paired_seeds": str(seed_path),
        "paired_seeds_sha256": batch.sha256(seed_path),
    }
    manifest_path = root / "ablation_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("missing ablation manifest in resume directory")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "schema", "variants", "variant_overrides", "families", "instances",
            "runs", "total_time", "copt_time", "max_parallel", "cpus_per_job",
            "cpu_usage_limit", "reserve_cpus", "job_memory_gb",
            "reserve_memory_gb", "check_interval", "ramp_up",
            "process_report_interval",
            "executable_sha256", "paired_seeds_sha256",
        )
        mismatches = [
            key for key in comparable_keys if existing.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise ValueError(
                "resume configuration differs for: %s" % ", ".join(mismatches)
            )
        manifest = existing
    else:
        batch.atomic_json(manifest_path, manifest)
    print("Ablation directory: %s" % root, flush=True)
    print("Paired seeds: %s" % seed_path, flush=True)
    for variant in args.variants:
        run_variant(args, root, seed_path, variant)
        write_aggregate(root, args.variants, args.families, args.runs)
    print("Ablation summary: %s" % (root / "ablation_summary.csv"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print("Error: %s" % error, file=sys.stderr)
        sys.exit(2)
