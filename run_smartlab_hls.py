#!/usr/bin/env python3
"""Run 10 repetitions for all SmartLab Brandimarte, Barnes and Dauzere instances."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import statistics
import sys
import time
from types import SimpleNamespace

import run_adaptive_brandimarte as batch


DATA_ROOT = batch.ROOT / "datasets/SmartLab_Challenge_2020"
FAMILIES = ("brandimarte", "barnes", "dauzere")
PATTERNS = {
    "brandimarte": re.compile(r"^fjsp\.brandimarte\.(Mk\d+)\..+\.txt$"),
    "barnes": re.compile(r"^fjsp\.barnes\.([^.]+)\..+\.txt$"),
    "dauzere": re.compile(r"^fjsp\.dauzere\.([^.]+)\..+\.txt$"),
}
EXPERIMENT_ENV = {
    "HLS_GA_ACTIVE_DECODE": "0",
    "HLS_GA_LOCAL_RESTART": "0",
    "HLS_GREEDY_COPT_REPAIR": "0",
    "HLS_GREEDY_FLEXIBLE_REPAIR": "1",
    "HLS_GREEDY_REPAIR_RELAYS": "1",
    "HLS_GREEDY_RELEASE_COUNT": "32",
    "HLS_GREEDY_REPAIR_SECONDS": "8",
    "HLS_TABU_CONTINUOUS_TRAJECTORY": "1",
    "HLS_ENABLE_RESTARTS": "1",
    "HLS_RESTART_STAGNATION_CALLS": "16",
    "HLS_GA_WORKLOAD_REDISTRIBUTION": "1",
    "HLS_DISPATCH_INITIAL": "1",
}


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def discover_instances(selected):
    result = {}
    for family in selected:
        entries = {}
        pattern = PATTERNS[family]
        for path in DATA_ROOT.glob("fjsp.%s.*.txt" % family):
            match = pattern.match(path.name)
            if match:
                label = match.group(1)
                if label in entries and entries[label] != path:
                    raise ValueError("Duplicate %s instance label: %s" % (family, label))
                entries[label] = path
        if not entries:
            raise ValueError("No %s instances found in %s" % (family, DATA_ROOT))
        result[family] = [(label, entries[label]) for label in sorted(entries, key=natural_key)]
    return result


def read_reference(family, labels):
    path = batch.ROOT / ("l2s_%s.csv" % family)
    if not path.is_file():
        return {label: [] for label in labels}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row[0]: row[1:] for row in csv.reader(handle) if row}
    return {label: rows.get(label, []) for label in labels}


def parse_env_parameters():
    os.environ.setdefault("SEARCH_MODE", "0.60")
    os.environ.setdefault("RANDOM_STRATEGY", "0.50")
    os.environ.setdefault("MAX_REPEAT", "1000")
    os.environ.setdefault("MAX_ITER", "45")
    os.environ.setdefault("POPULATION_SIZE", "32")
    os.environ.setdefault("TABU_LENGTH", "30")
    os.environ.setdefault("COPT_TIME", "0")
    os.environ.setdefault("TOTAL_TIME", "3600")
    os.environ.setdefault("GUIDED_TOP_K", "96")
    os.environ.setdefault("GREEDY_RL_TOP_K", "256")
    os.environ.setdefault("TABU_RL_TOP_K", "1024")
    os.environ.setdefault("GREEDY_RL_RULE_FRACTION", "0.30")
    os.environ.setdefault("GREEDY_RL_EXPLORATION", "0.20")
    os.environ.setdefault("TABU_RL_RULE_FRACTION", "0.30")
    os.environ.setdefault("TABU_RL_EXPLORATION", "0.25")
    os.environ.setdefault("TABU_RL_MIN_PER_NEIGHBORHOOD", "24")
    os.environ.setdefault("TABU_RL_FALLBACK_ON_NON_IMPROVEMENT", "0")
    return batch.requested_solver()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "10")))
    parser.add_argument("--log-dir", default=os.environ.get("LOG_DIR"))
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RESUME", "0") == "1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "14")))
    parser.add_argument("--cpus-per-job", type=int, default=int(os.environ.get("CPUS_PER_JOB", "1")))
    parser.add_argument("--cpu-usage-limit", type=float, default=float(os.environ.get("CPU_USAGE_LIMIT", "0.95")))
    parser.add_argument("--reserve-cpus", type=float, default=float(os.environ.get("CPU_RESERVE", "0")))
    parser.add_argument("--job-memory-gb", type=float, default=float(os.environ.get("JOB_MEMORY_GB", "2")))
    parser.add_argument("--reserve-memory-gb", type=float, default=float(os.environ.get("MEMORY_RESERVE_GB", "4")))
    parser.add_argument("--check-interval", type=float, default=float(os.environ.get("CHECK_INTERVAL", "5")))
    parser.add_argument("--ramp-up", type=int, default=int(os.environ.get("RAMP_UP", "2")))
    parser.add_argument("--timeout-grace", type=float, default=float(os.environ.get("TIMEOUT_GRACE", "120")))
    args = parser.parse_args()
    if args.runs < 1 or args.max_parallel < 0 or args.cpus_per_job < 1:
        parser.error("runs, cpus-per-job and max-parallel must be valid positive values")
    if not 0.1 <= args.cpu_usage_limit <= 1 or args.job_memory_gb <= 0 or args.reserve_memory_gb < 0:
        parser.error("invalid CPU or memory settings")
    if args.check_interval < 0.1 or args.ramp_up < 1 or args.timeout_grace < 1:
        parser.error("invalid scheduler timing settings")
    if len(set(args.families)) != len(args.families):
        parser.error("duplicate family")
    return args


def solver_path():
    configured = os.environ.get("EXECUTABLE")
    if configured:
        return Path(configured).resolve()
    optimized = batch.ROOT / "build/optimized/production_scheduling"
    return optimized if optimized.is_file() else batch.ROOT / "build/production_scheduling"


def model_path():
    return Path(os.environ.get(
        "MODEL_PATH",
        str(batch.ROOT / "rl_greedy/models/greedy_policy_brandimarte.txt")
    )).resolve()


def create_manifest(directory, settings, discovered, parameters):
    executable = solver_path()
    model = model_path()
    if not executable.is_file() or not os.access(str(executable), os.X_OK):
        raise ValueError("Executable not found or not executable: %s" % executable)
    if not model.is_file():
        raise ValueError("RL model not found: %s" % model)
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    executable_snapshot = inputs / "production_scheduling"
    model_snapshot = inputs / "greedy_policy.txt"
    shutil.copy2(executable, executable_snapshot)
    shutil.copy2(model, model_snapshot)
    artifacts = {
        "executable": {"source": str(executable), "snapshot": str(executable_snapshot.relative_to(directory)),
                        "sha256": batch.sha256(executable_snapshot)},
        "model": {"source": str(model), "snapshot": str(model_snapshot.relative_to(directory)),
                   "sha256": batch.sha256(model_snapshot)},
    }
    families = {}
    instances = []
    references = []
    tasks = []
    seen_seeds = set()
    for family in settings.families:
        family_dir = inputs / family
        family_dir.mkdir(exist_ok=True)
        labels = []
        reference_rows = read_reference(family, [label for label, _ in discovered[family]])
        for label, source in discovered[family]:
            key = "%s::%s" % (family, label)
            safe = "%s__%s" % (family, label)
            snapshot = family_dir / (label + source.suffix)
            shutil.copy2(source, snapshot)
            artifacts[key] = {"source": str(source), "snapshot": str(snapshot.relative_to(directory)),
                              "sha256": batch.sha256(snapshot), "family": family, "label": label}
            instances.append(key)
            labels.append(label)
            references.append([key] + reference_rows[label])
            for run in range(1, settings.runs + 1):
                seed = secrets.randbits(32)
                while seed in seen_seeds:
                    seed = secrets.randbits(32)
                seen_seeds.add(seed)
                tasks.append({"instance": key, "family": family, "label_name": label, "run": run,
                              "seed": seed, "label": "%s_run%02d" % (safe, run),
                              "status": "PENDING", "attempt": 0})
        families[family] = labels
    reference_path = inputs / "l2s_reference.csv"
    batch.write_csv(reference_path, [["instance"] + ["run%d" % n for n in range(1, 11)]] + references)
    artifacts["l2s"] = {"source": "generated from l2s_barnes.csv/l2s_dauzere.csv/l2s_brandimarte.csv",
                         "snapshot": str(reference_path.relative_to(directory)), "sha256": batch.sha256(reference_path)}
    output_root = Path(os.environ.get("OUTPUT_DIR", str(batch.ROOT))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = {"schema": 1, "created_at": batch.now(), "families": families,
              "instances": instances, "runs": settings.runs, "parameters": parameters,
              "artifacts": artifacts, "experimental_environment": dict(EXPERIMENT_ENV),
              "output_files": {family: str(output_root / ("HLS_%s.csv" % family)) for family in settings.families},
              "scheduler_at_creation": vars(settings)}
    state = {"tasks": tasks, "peak_rss": 0, "phase": "READY"}
    batch.write_csv(directory / "seeds.csv", [["family", "instance", "run", "seed"]] + [
        [task["family"], task["label_name"], task["run"], task["seed"]] for task in tasks])
    batch.atomic_json(directory / "config.json", config)
    batch.atomic_json(directory / "state.json", state)
    return config, state


def write_family_outputs(directory, config, state):
    by_key = {(task["family"], task["label_name"], task["run"]): task for task in state["tasks"]}
    for family, labels in config["families"].items():
        rows = [["instance"] + ["run%d" % run for run in range(1, config["runs"] + 1)]]
        for label in labels:
            values = [by_key[(family, label, run)] for run in range(1, config["runs"] + 1)]
            rows.append([label] + [task.get("makespan", task["status"]) for task in values])
        batch.write_csv(Path(config["output_files"][family]), rows)


class SmartLabRunner(batch.Runner):
    def __init__(self, directory, settings, config, state, resources):
        super().__init__(directory, settings, config, state, resources)
        self.next_process_report = time.monotonic() + 600.0
        self.process_report_path = directory / "processes_10min.csv"
        if not self.process_report_path.exists() or self.process_report_path.stat().st_size == 0:
            with self.process_report_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([
                    "timestamp", "pid", "family", "instance", "run", "seed", "elapsed_seconds",
                    "cpu_seconds", "rss_mib", "cpus", "status"
                ])

    def environment(self):
        result = super().environment()
        result.update(self.config["experimental_environment"])
        return result

    def process_report(self):
        process_snapshot = batch.process_stats()
        rows = []
        timestamp = batch.now()
        for pid, item in sorted(self.active.items()):
            task = item["task"]
            process = process_snapshot.get(pid, {})
            rows.append([
                timestamp, pid, task["family"], task["label_name"], task["run"], task["seed"],
                "%.1f" % (time.monotonic() - item["started"]),
                "%.1f" % process.get("cpu", 0.0),
                "%.1f" % (process.get("rss", 0) / batch.GIB * 1024),
                task.get("cpus", ""), task["status"]
            ])
        if not rows:
            rows.append([timestamp, "", "", "", "", "", "", "", "", "", "NO_ACTIVE_PROCESS"])
        with self.process_report_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        descriptions = [
            "pid=%s family=%s instance=%s run=%s elapsed=%.0fs cpu=%.1fs rss=%.1fMiB cpus=%s" % (
                row[1], row[2], row[3], row[4], float(row[6]), float(row[7]), float(row[8]), row[9]
            ) for row in rows if row[10] != "NO_ACTIVE_PROCESS"
        ]
        self.message("10-minute process report: %s" % ("; ".join(descriptions) if descriptions else "no active solver"))

    def report(self, snapshot, job_memory, reserve):
        super().report(snapshot, job_memory, reserve)
        write_family_outputs(self.directory, self.config, self.state)
        if time.monotonic() >= self.next_process_report:
            self.process_report()
            self.next_process_report = time.monotonic() + 600.0


def install_output_hook():
    original = batch.write_results

    def write_all_outputs(directory, config, state):
        original(directory, config, state)
        write_family_outputs(directory, config, state)

    batch.write_results = write_all_outputs


def validate_outputs(config, state):
    expected = len(config["instances"]) * config["runs"]
    completed = sum(task["status"] == "OK" for task in state["tasks"])
    if state["phase"] != "COMPLETE" or completed != expected:
        raise RuntimeError("Batch incomplete: %d/%d successful" % (completed, expected))


def main():
    settings = parse_args()
    selected = tuple(settings.families)
    discovered = discover_instances(selected)
    total_instances = sum(len(values) for values in discovered.values())
    parameters = parse_env_parameters()
    resources = batch.Resources()
    snapshot = resources.sample(set())
    probe_settings = SimpleNamespace(cpu_usage_limit=settings.cpu_usage_limit, reserve_cpus=settings.reserve_cpus,
                                     cpus_per_job=settings.cpus_per_job, job_memory_gb=settings.job_memory_gb,
                                     reserve_memory_gb=settings.reserve_memory_gb, max_parallel=settings.max_parallel)
    ceiling = batch.admission(snapshot, probe_settings, 0)[0]
    print("Families: %s" % ", ".join("%s=%d" % (family, len(discovered[family])) for family in selected), flush=True)
    print("Experiments: %d instances x %d runs = %d" % (total_instances, settings.runs, total_instances * settings.runs), flush=True)
    print("Effective CPU quota: %.2f cores; memory limit: %.2f GiB; available: %.2f GiB; concurrency ceiling now: %d" % (
        snapshot["cores"], snapshot["memory_total"] / batch.GIB, snapshot["memory_available"] / batch.GIB, ceiling), flush=True)
    if settings.dry_run:
        print("Parameters: %s" % json.dumps(parameters, sort_keys=True), flush=True)
        print("Outputs: %s" % ", ".join("HLS_%s.csv" % family for family in selected), flush=True)
        return 0
    stamp = os.environ.get("RUN_TIMESTAMP", time.strftime("%Y%m%d_%H%M%S"))
    directory = Path(settings.log_dir or (batch.ROOT / "run_logs" / ("smartlab_" + stamp))).resolve()
    output_root = Path(os.environ.get("OUTPUT_DIR", str(batch.ROOT))).resolve()
    output_files = [output_root / ("HLS_%s.csv" % family) for family in selected]
    overwrite = os.environ.get("OVERWRITE_OUTPUT", "0") == "1"
    if not settings.resume and not overwrite:
        existing = [str(path) for path in output_files if path.exists()]
        if existing:
            raise ValueError("Output exists; set OVERWRITE_OUTPUT=1 to replace: %s" % ", ".join(existing))
    if settings.resume:
        if not (directory / "config.json").is_file():
            raise ValueError("No manifest to resume in %s" % directory)
        config, state = batch.resume_run(directory)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Run directory is not empty: %s" % directory)
        directory.mkdir(parents=True, exist_ok=True)
        config, state = create_manifest(directory, settings, discovered, parameters)
    install_output_hook()
    write_family_outputs(directory, config, state)
    runner = SmartLabRunner(directory, settings, config, state, resources)
    result = runner.execute()
    validate_outputs(config, state)
    print("Run directory: %s" % directory)
    print("Output files: %s" % ", ".join(config["output_files"].values()))
    return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print("Error: %s" % error, file=sys.stderr)
        sys.exit(2)
