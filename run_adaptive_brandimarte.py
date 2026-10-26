#!/usr/bin/env python3
"""Run repeated HLS experiments with resource-aware admission on Linux."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone


ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
CHINA_TIME = timezone(timedelta(hours=8))
INSTANCES = ["Mk%02d" % n for n in range(1, 11)]
FINAL = re.compile(r"^Final makespan:\s*(\d+)\s*$", re.MULTILINE)

# All instances use this same configuration, matching the fixed uniform version.
SOLVER_PARAMETERS = [
    ("SEARCH_MODE", "0.60", float, 0, 1),
    ("RANDOM_STRATEGY", "0.50", float, 0, 1),
    ("MAX_REPEAT", "1000", int, 1, None),
    ("MAX_ITER", "45", int, 1, None),
    ("POPULATION_SIZE", "32", int, 2, None),
    ("TABU_LENGTH", "30", int, 1, None),
    ("COPT_TIME", "300", int, 0, None),
    ("TOTAL_TIME", "3600", int, 1, None),
]
POLICY_PARAMETERS = [
    ("GUIDED_TOP_K", "96", int, 1, None),
    ("GREEDY_RL_TOP_K", "256", int, 1, None),
    ("TABU_RL_TOP_K", "1024", int, 1, None),
    ("GREEDY_RL_RULE_FRACTION", "0.30", float, 0, 1),
    ("GREEDY_RL_EXPLORATION", "0.20", float, 0, 1),
    ("TABU_RL_RULE_FRACTION", "0.30", float, 0, 1),
    ("TABU_RL_EXPLORATION", "0.25", float, 0, 1),
    ("TABU_RL_MIN_PER_NEIGHBORHOOD", "24", int, 1, None),
    ("TABU_RL_FALLBACK_ON_NON_IMPROVEMENT", "0", int, 0, 1),
]


def now():
    return datetime.now(CHINA_TIME).isoformat(timespec="seconds")


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    atomic_text(path, buffer.getvalue())


def number(name, default, convert, minimum, maximum=None):
    value = convert(os.environ.get(name, str(default)))
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Invalid %s: %s" % (name, value))
    return value


def cgroup_directories(proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    """Find visible memberships, including ancestors which can limit a container."""
    groups = {"v2": [], "cpu": [], "memory": []}
    for line in read_text(proc_root / "self/cgroup").splitlines():
        pieces = line.split(":", 2)
        if len(pieces) != 3:
            continue
        _, controllers, membership = pieces
        kinds = ["v2"] if not controllers else [
            kind for kind in ("cpu", "memory") if kind in controllers.split(",")
        ]
        for kind in kinds:
            roots = [cgroup_root] if kind == "v2" else [
                cgroup_root / controllers, cgroup_root / kind,
                cgroup_root / "cpu,cpuacct" if kind == "cpu" else cgroup_root / kind,
            ]
            for root in roots:
                if not root.is_dir():
                    continue
                leaf = root / membership.lstrip("/")
                # A cgroup namespace may expose its own membership as the mount root.
                root, leaf = root.resolve(), leaf.resolve()
                if not leaf.is_dir() or (leaf != root and root not in leaf.parents):
                    leaf = root
                while True:
                    if leaf not in groups[kind]:
                        groups[kind].append(leaf)
                    if leaf == root:
                        break
                    leaf = leaf.parent
    return groups


def process_stats(proc_root=Path("/proc")):
    """CPU times include all threads; group IDs include any solver subprocesses."""
    result = {}
    ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        text = read_text(path / "stat")
        fields = text[text.rfind(")") + 2:].split()
        if len(fields) < 22:
            continue
        try:
            result[int(path.name)] = {
                "group": int(fields[2]),
                "cpu": (int(fields[11]) + int(fields[12])) / ticks,
                "start_ticks": int(fields[19]),
                "rss": int(fields[21]) * page_size,
            }
        except ValueError:
            continue
    return result


class Resources:
    def __init__(self, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup"), affinity=None):
        self.proc_root = proc_root
        self.groups = cgroup_directories(proc_root, cgroup_root)
        self.affinity = sorted(os.sched_getaffinity(0) if affinity is None else affinity)
        self.previous_time = None
        self.previous_usage = None
        self.previous_scope = None
        self.previous_own = {}

    def limits(self):
        cores = float(len(self.affinity))
        scope = None
        for group in self.groups["v2"]:
            fields = read_text(group / "cpu.max").split()
            if len(fields) == 2 and fields[0] != "max":
                quota = int(fields[0]) / int(fields[1])
                if quota <= cores:
                    cores, scope = quota, (group, "v2")
        for group in self.groups["cpu"]:
            quota = read_text(group / "cpu.cfs_quota_us").strip()
            period = read_text(group / "cpu.cfs_period_us").strip()
            if quota and period and int(quota) > 0 and int(quota) / int(period) <= cores:
                cores, scope = int(quota) / int(period), (group, "v1")
        memory = {}
        for line in read_text(self.proc_root / "meminfo").splitlines():
            key, _, value = line.partition(":")
            if value.split():
                memory[key] = int(value.split()[0]) * 1024
        total = memory.get("MemTotal", 0)
        available = memory.get("MemAvailable", memory.get("MemFree", 0))
        for kind in ("v2", "memory"):
            for group in self.groups[kind]:
                usage_name = "memory.current" if kind == "v2" else "memory.usage_in_bytes"
                usage = read_text(group / usage_name).strip()
                limit_names = ("memory.max", "memory.high") if kind == "v2" else ("memory.limit_in_bytes",)
                for name in limit_names:
                    limit = read_text(group / name).strip()
                    if limit.isdigit() and usage.isdigit():
                        total = min(total, int(limit))
                        available = min(available, max(0, int(limit) - int(usage)))
        if cores <= 0 or total <= 0:
            raise RuntimeError("Cannot determine effective CPU/memory limits on this Linux host.")
        return {"cores": cores, "memory_total": total, "memory_available": available, "scope": scope}

    def cpu_usage(self, scope):
        if scope is not None:
            group, kind = scope
            if kind == "v2":
                for line in read_text(group / "cpu.stat").splitlines():
                    fields = line.split()
                    if fields and fields[0] == "usage_usec":
                        return int(fields[1]) / 1e6
            else:
                usage = read_text(group / "cpuacct.usage").strip()
                if usage:
                    return int(usage) / 1e9
        allowed = {"cpu%d" % cpu for cpu in self.affinity}
        busy = 0
        for line in read_text(self.proc_root / "stat").splitlines():
            fields = line.split()
            if fields and fields[0] in allowed:
                values = [int(value) for value in fields[1:9]]
                # Idle and IO wait do not consume computation capacity.
                busy += sum(values) - values[3] - values[4]
        return busy / os.sysconf("SC_CLK_TCK")

    def sample(self, active_groups):
        snapshot = self.limits()
        stamp = time.monotonic()
        usage = self.cpu_usage(snapshot["scope"])
        processes = process_stats(self.proc_root)
        own = {pid: item["cpu"] for pid, item in processes.items() if item["group"] in active_groups}
        rss = {group: 0 for group in active_groups}
        for item in processes.values():
            if item["group"] in rss:
                rss[item["group"]] += item["rss"]
        elapsed = stamp - self.previous_time if self.previous_time is not None else 0
        busy = 0.0
        own_busy = 0.0
        if elapsed > 0 and snapshot["scope"] == self.previous_scope:
            busy = min(snapshot["cores"], max(0, usage - self.previous_usage) / elapsed)
            own_busy = sum(max(0, value - self.previous_own.get(pid, 0)) for pid, value in own.items()) / elapsed
        snapshot.update({"busy_cores": busy, "external_cores": max(0, busy - own_busy), "rss": rss})
        self.previous_time, self.previous_usage = stamp, usage
        self.previous_scope, self.previous_own = snapshot["scope"], own
        return snapshot


def admission(snapshot, settings, peak_rss):
    """Reserve growth of running jobs before admitting additional ones."""
    running = len(snapshot["rss"])
    cpu_budget = snapshot["cores"] * settings.cpu_usage_limit - min(settings.reserve_cpus, snapshot["cores"] * 0.1)
    cpu_slots = max(0, math.floor((cpu_budget - snapshot["external_cores"]) / settings.cpus_per_job))
    # Fractional and single-core quotas must still be able to run one experiment.
    if cpu_slots == 0 and snapshot["external_cores"] < snapshot["cores"] * 0.1:
        cpu_slots = 1
    if snapshot["busy_cores"] >= snapshot["cores"] * 0.95:
        cpu_slots = min(cpu_slots, running)
    job_memory = max(settings.job_memory_gb * GIB, peak_rss * 1.25)
    reserve = max(settings.reserve_memory_gb * GIB, snapshot["memory_total"] * 0.1)
    unconsumed = sum(max(0, job_memory - rss) for rss in snapshot["rss"].values())
    memory_slots = max(0, running + math.floor((snapshot["memory_available"] - reserve - unconsumed) / job_memory))
    hard_cpu_cap = max(1, math.floor(snapshot["cores"] / settings.cpus_per_job))
    limit = min(cpu_slots, memory_slots, hard_cpu_cap)
    if settings.max_parallel:
        limit = min(limit, settings.max_parallel)
    return max(0, limit), job_memory, reserve


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=number("RUNS", 10, int, 1))
    parser.add_argument("--instances", nargs="+", default=os.environ.get("INSTANCES", " ".join(INSTANCES)).split())
    parser.add_argument("--log-dir", default=os.environ.get("LOG_DIR"))
    parser.add_argument("--resume", action="store_true", default=os.environ.get("RESUME", "0") == "1")
    parser.add_argument("--dry-run", action="store_true", help="Show resource limits and plan without writing or running experiments.")
    parser.add_argument("--max-parallel", type=int, default=number("MAX_PARALLEL", os.environ.get("PARALLEL_JOBS", 0), int, 0))
    parser.add_argument("--cpus-per-job", type=int, default=number("CPUS_PER_JOB", 1, int, 1))
    parser.add_argument("--cpu-usage-limit", type=float, default=number("CPU_USAGE_LIMIT", 0.85, float, 0.1, 1))
    parser.add_argument("--reserve-cpus", type=float, default=number("CPU_RESERVE", 1, float, 0))
    parser.add_argument("--job-memory-gb", type=float, default=number("JOB_MEMORY_GB", 2, float, 0.01))
    parser.add_argument("--reserve-memory-gb", type=float, default=number("MEMORY_RESERVE_GB", 4, float, 0))
    parser.add_argument("--check-interval", type=float, default=number("CHECK_INTERVAL", 5, float, 0.1))
    parser.add_argument("--ramp-up", type=int, default=number("RAMP_UP", 2, int, 1))
    parser.add_argument("--timeout-grace", type=float, default=number("TIMEOUT_GRACE", 120, float, 1))
    args = parser.parse_args()
    if args.runs < 1 or not args.instances or len(set(args.instances)) != len(args.instances):
        parser.error("RUNS must be positive and INSTANCES must be nonempty and unique.")
    if any(instance not in INSTANCES for instance in args.instances):
        parser.error("Only Mk01 through Mk10 are supported.")
    if (args.max_parallel < 0 or args.cpus_per_job < 1 or not 0.1 <= args.cpu_usage_limit <= 1
            or args.reserve_cpus < 0 or args.job_memory_gb <= 0 or args.reserve_memory_gb < 0
            or args.check_interval < 0.1 or args.ramp_up < 1 or args.timeout_grace < 1):
        parser.error("Invalid scheduler resource settings.")
    if any(not math.isfinite(value) for value in (args.cpu_usage_limit, args.reserve_cpus,
           args.job_memory_gb, args.reserve_memory_gb, args.check_interval, args.timeout_grace)):
        parser.error("Resource settings must be finite.")
    if args.resume and not args.log_dir:
        parser.error("--resume requires --log-dir (or LOG_DIR).")
    return args


def requested_solver():
    parameters = {}
    for name, default, convert, minimum, maximum in SOLVER_PARAMETERS + POLICY_PARAMETERS:
        parameters[name] = str(number(name, default, convert, minimum, maximum))
    if int(parameters["COPT_TIME"]) > int(parameters["TOTAL_TIME"]):
        raise ValueError("COPT_TIME must not exceed TOTAL_TIME.")
    return parameters


def solver_environment(parameters, model_path):
    values = {
        "HLS_TARGET_MAKESPAN": "0", "HLS_ENABLE_GUIDED_SEARCH": "force",
        "HLS_GREEDY_SCORE_BONUS": "0", "HLS_GENETIC_SCORE_BONUS": "0",
        "HLS_TABU_SCORE_BONUS": "0", "HLS_GUIDED_SCORE_BONUS": "0",
        "HLS_GA_ACTIVE_DECODE": "0", "HLS_GA_LOCAL_RESTART": "0",
        "HLS_ENABLE_RESTARTS": "0", "HLS_GREEDY_COPT_REPAIR": "0",
        "HLS_GREEDY_ROUTE_REPAIR": "0",
        "HLS_ROUTE_AWARE_METHOD_SELECTION": "0",
    }
    for method in ("GREEDY", "TABU"):
        prefix = "HLS_%s_RL" % method
        values.update({prefix: "1", prefix + "_MODEL": str(model_path),
                       prefix + "_DYNAMIC_TOP_K": "0", prefix + "_FALLBACK_ON_MISS": "1",
                       prefix + "_TEMPERATURE": "1"})
    values.update({"HLS_GREEDY_RL_VERIFY_WITH_BASELINE": "0", "HLS_TABU_RL_VERIFY_FULL_NEIGHBORHOOD": "0"})
    for name, *_ in POLICY_PARAMETERS:
        values["HLS_" + name] = parameters[name]
    return values


def create_run(directory, settings, parameters):
    sources = {"executable": Path(os.environ.get("EXECUTABLE", str(ROOT / "build/production_scheduling"))).resolve(),
               "model": Path(os.environ.get("MODEL_PATH", str(ROOT / "rl_greedy/models/greedy_policy_brandimarte.txt"))).resolve(),
               "l2s": Path(os.environ.get("L2S_FILE", str(ROOT / "l2s_brandimarte.csv"))).resolve()}
    data_dir = Path(os.environ.get("DATA_DIR", str(ROOT / "datasets/Brandimarte_Data/Text"))).resolve()
    for instance in settings.instances:
        sources[instance] = data_dir / (instance + ".fjs")
    for key, path in sources.items():
        if not path.is_file():
            raise ValueError("Missing %s: %s" % (key, path))
    if not os.access(str(sources["executable"]), os.X_OK):
        raise ValueError("EXECUTABLE is not executable.")
    inputs = directory / "inputs"
    inputs.mkdir()
    artifacts = {}
    for key, source in sources.items():
        destination = inputs / {"executable": "production_scheduling", "model": "greedy_policy.txt",
                                "l2s": "l2s_brandimarte.csv"}.get(key, source.name)
        shutil.copy2(str(source), str(destination))
        artifacts[key] = {"source": str(source), "snapshot": str(destination.relative_to(directory)), "sha256": sha256(destination)}
    config = {"schema": 1, "created_at": now(), "instances": settings.instances, "runs": settings.runs,
              "parameters": parameters, "artifacts": artifacts, "scheduler_at_creation": vars(settings)}
    atomic_json(directory / "config.json", config)
    tasks, seen = [], set()
    # Round-robin over instances, then repetitions; seeds are never chosen by score.
    for run in range(1, settings.runs + 1):
        for instance in settings.instances:
            seed = secrets.randbits(32)
            while seed in seen:
                seed = secrets.randbits(32)
            seen.add(seed)
            tasks.append({"instance": instance, "run": run, "seed": seed,
                          "label": "%s_run%02d" % (instance, run), "status": "PENDING", "attempt": 0})
    state = {"tasks": tasks, "peak_rss": 0, "phase": "READY"}
    write_csv(directory / "seeds.csv", [["instance", "run", "seed"]] + [
        [task["instance"], task["run"], task["seed"]] for task in tasks])
    atomic_json(directory / "state.json", state)
    return config, state


def resume_run(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("Unsupported experiment manifest schema.")
    for item in config["artifacts"].values():
        if sha256(directory / item["snapshot"]) != item["sha256"]:
            raise ValueError("Snapshot changed: %s" % item["snapshot"])
    processes = process_stats()
    for task in state["tasks"]:
        if task["status"] == "RUNNING":
            live = processes.get(task.get("pid"))
            if live and live["start_ticks"] == task.get("start_ticks"):
                raise ValueError("Previous solver is still running (PID %s); stop the previous launcher first." % task["pid"])
        if task["status"] == "OK":
            text = read_text(directory / (task["label"] + ".txt"))
            values = FINAL.findall(text)
            if not values or int(values[-1]) != task["makespan"] or task.get("exit_code") != 0:
                raise ValueError("Completed result/log mismatch: %s" % task["label"])
        else:
            task["status"] = "PENDING"
            task.pop("makespan", None)
    return config, state


def reference_means(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in list(csv.reader(handle))[1:]:
            values = [float(value) for value in row[1:] if value.strip()]
            if values:
                result[row[0]] = statistics.mean(values)
    return result


def write_results(directory, config, state):
    by_key = {(task["instance"], task["run"]): task for task in state["tasks"]}
    rows = [["instance"] + ["run%d" % run for run in range(1, config["runs"] + 1)]]
    summary = [["instance", "completed", "expected", "mean", "min", "max", "stddev", "l2s_mean", "hls_minus_l2s", "comparison"]]
    references = reference_means(directory / config["artifacts"]["l2s"]["snapshot"])
    for instance in config["instances"]:
        tasks = [by_key[(instance, run)] for run in range(1, config["runs"] + 1)]
        values = [task["makespan"] for task in tasks if task["status"] == "OK"]
        rows.append([instance] + [task["makespan"] if task["status"] == "OK" else task["status"] for task in tasks])
        mean = statistics.mean(values) if values else None
        l2s = references.get(instance)
        difference = mean - l2s if mean is not None and l2s is not None else None
        comparison = "INCOMPLETE" if len(values) != config["runs"] else (
            "NO_REFERENCE" if difference is None else "BETTER" if difference < -1e-9 else "WORSE" if difference > 1e-9 else "TIE")
        summary.append([instance, len(values), config["runs"], "%.4f" % mean if values else "",
                        min(values) if values else "", max(values) if values else "",
                        "%.4f" % statistics.stdev(values) if len(values) > 1 else "0.0000" if values else "",
                        "%.4f" % l2s if l2s is not None else "", "%.4f" % difference if difference is not None else "", comparison])
    write_csv(directory / "result.csv", rows)
    write_csv(directory / "summary.csv", summary)
    columns = ["instance", "run", "seed", "status", "makespan", "wall_seconds", "exit_code", "attempt", "pid", "cpus", "initial_source", "error"]
    write_csv(directory / "details.csv", [columns] + [[task.get(key, "") for key in columns] for task in state["tasks"]])
    atomic_json(directory / "state.json", state)


class Runner:
    def __init__(self, directory, settings, config, state, resources):
        self.directory, self.settings, self.config, self.state = directory, settings, config, state
        self.resources = resources
        self.active = {}
        self.stop = False
        self.stop_signal = 0
        self.limit = 0
        self.cpu_cursor = 0
        self.log = (directory / "scheduler.log").open("a", buffering=1, encoding="utf-8")
        self.metrics = (directory / "scheduler.csv").open("a", newline="", buffering=1, encoding="utf-8")
        self.metrics_writer = csv.writer(self.metrics)
        if self.metrics.tell() == 0:
            self.metrics_writer.writerow(["time", "running", "pending", "ok", "failed", "limit", "effective_cpus", "cpu_percent", "external_cores", "memory_available_gb", "job_memory_gb", "memory_reserve_gb"])

    def message(self, value):
        line = "[%s] %s" % (now(), value)
        print(line, flush=True)
        self.log.write(line + "\n")

    def request_stop(self, signum, _frame):
        self.stop, self.stop_signal = True, signum

    def environment(self):
        # Ignore inherited experimental HLS flags so all runs use the manifest.
        env = {key: value for key, value in os.environ.items() if not key.startswith("HLS_")}
        model = self.directory / self.config["artifacts"]["model"]["snapshot"]
        env.update(solver_environment(self.config["parameters"], model))
        copt_home = env.get("COPT_HOME", str(Path.home() / "copt80"))
        env["COPT_HOME"] = copt_home
        env.setdefault("COPT_LICENSE_DIR", copt_home)
        env["LD_LIBRARY_PATH"] = str(Path(copt_home) / "lib") + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(self.settings.cpus_per_job)
        return env

    def launch(self, task):
        occupied = {cpu for item in self.active.values() for cpu in item["cpus"]}
        pool = self.resources.affinity
        rotated = pool[self.cpu_cursor:] + pool[:self.cpu_cursor]
        cpus = [cpu for cpu in rotated if cpu not in occupied][:self.settings.cpus_per_job]
        if len(cpus) != self.settings.cpus_per_job:
            return False
        self.cpu_cursor = (pool.index(cpus[-1]) + 1) % len(pool)
        path = self.directory / (task["label"] + ".txt")
        if path.exists():
            archive = self.directory / "attempts"
            archive.mkdir(exist_ok=True)
            path.rename(archive / ("%s_attempt%02d_%d.txt" % (task["label"], task["attempt"], time.time_ns())))
        artifact = self.config["artifacts"]
        parameters = self.config["parameters"]
        command = [str(self.directory / artifact["executable"]["snapshot"]),
                   str(self.directory / artifact[task["instance"]]["snapshot"])]
        command += [parameters[name] for name, *_ in SOLVER_PARAMETERS] + [str(task["seed"])]
        task.update({"status": "RUNNING", "attempt": task["attempt"] + 1, "started_at": now(), "cpus": " ".join(map(str, cpus))})
        handle = path.open("w", encoding="utf-8")
        try:
            # The single-threaded parent sets affinity before exec; COPT threads
            # inherit it, so a solver cannot use every visible host CPU.
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       env=self.environment(), cwd=str(ROOT), start_new_session=True,
                                       preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        except (OSError, subprocess.SubprocessError) as error:
            handle.write("Launch failed: %s\n" % error)
            handle.close()
            task.update({"status": "FAILED", "error": str(error), "exit_code": 127, "wall_seconds": 0})
            self.message("failed %s: %s" % (task["label"], error))
            return True
        task["pid"] = process.pid
        task["start_ticks"] = process_stats().get(process.pid, {}).get("start_ticks")
        self.active[process.pid] = {"process": process, "task": task, "handle": handle,
                                    "started": time.monotonic(), "cpus": cpus, "ending": None}
        self.message("start %s seed=%d pid=%d CPUs=%s" % (task["label"], task["seed"], process.pid, task["cpus"]))
        return True

    @staticmethod
    def terminate(item, reason):
        if item["ending"] is None:
            item["ending"] = (time.monotonic(), reason)
            try:
                os.killpg(item["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def reap(self):
        changed = False
        timeout = int(self.config["parameters"]["TOTAL_TIME"]) + self.settings.timeout_grace
        for pid, item in list(self.active.items()):
            process, task = item["process"], item["task"]
            elapsed = time.monotonic() - item["started"]
            if elapsed > timeout:
                self.terminate(item, "TIMEOUT")
            if item["ending"] and time.monotonic() - item["ending"][0] > 3 and process.poll() is None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = process.poll()
            if code is None:
                continue
            item["handle"].close()
            path = self.directory / (task["label"] + ".txt")
            text = read_text(path)
            finals = FINAL.findall(text)
            experiment_environment = self.config.get("experimental_environment", {})
            expected_greedy_rl = experiment_environment.get("HLS_GREEDY_RL", "1") != "0"
            expected_tabu_rl = experiment_environment.get("HLS_TABU_RL", "1") != "0"
            greedy_rl_ok = (
                ("greedy_rl_enabled: true" in text) if expected_greedy_rl
                else ("greedy_rl_enabled: false" in text)
            )
            tabu_rl_ok = (
                ("tabu_rl_enabled: true" in text) if expected_tabu_rl
                else ("tabu_rl_enabled: false" in text)
            )
            rl_ok = greedy_rl_ok and tabu_rl_ok
            status = item["ending"][1] if item["ending"] else "OK" if code == 0 and finals and rl_ok else "FAILED"
            task.update({"status": status, "exit_code": code, "wall_seconds": round(elapsed, 3), "finished_at": now()})
            if status == "OK":
                task["makespan"] = int(finals[-1])
            elif not item["ending"]:
                task["error"] = (
                    "exit=%s, final_makespan=%s, expected_greedy_rl=%s, "
                    "greedy_rl_ok=%s, expected_tabu_rl=%s, tabu_rl_ok=%s" % (
                        code, bool(finals), expected_greedy_rl, greedy_rl_ok,
                        expected_tabu_rl, tabu_rl_ok
                    )
                )
            task["initial_source"] = "COPT" if "COPT initial makespan:" in text else "HLS_FALLBACK" if "HLS fallback initial makespan:" in text else "UNKNOWN"
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\nwall_seconds=%s\nexit_code=%s\nbatch_status=%s\n" % (task["wall_seconds"], code, status))
            self.message("%s %s makespan=%s wall=%.1fs" % (status.lower(), task["label"], task.get("makespan", "-"), elapsed))
            del self.active[pid]
            changed = True
        return changed

    def report(self, snapshot, job_memory, reserve):
        tasks = self.state["tasks"]
        pending = sum(task["status"] == "PENDING" for task in tasks)
        ok = sum(task["status"] == "OK" for task in tasks)
        failed = sum(task["status"] in ("FAILED", "TIMEOUT", "INTERRUPTED") for task in tasks)
        cpu_percent = 100 * snapshot["busy_cores"] / snapshot["cores"]
        self.metrics_writer.writerow([now(), len(self.active), pending, ok, failed, self.limit, "%.2f" % snapshot["cores"],
                                      "%.2f" % cpu_percent, "%.2f" % snapshot["external_cores"],
                                      "%.2f" % (snapshot["memory_available"] / GIB), "%.2f" % (job_memory / GIB), "%.2f" % (reserve / GIB)])
        line = "progress OK=%d/%d failed=%d running=%d limit=%d pending=%d CPU=%.1f%% memory_available=%.1fGiB" % (
            ok, len(tasks), failed, len(self.active), self.limit, pending, cpu_percent, snapshot["memory_available"] / GIB)
        self.message(line)
        atomic_text(self.directory / "status.md", "# HLS Brandimarte batch\n\nUpdated: %s\nPhase: %s\n%s\n\nResults: result.csv\nPer-instance statistics and L2S comparison: summary.csv\n" % (now(), self.state["phase"], line))

    def execute(self):
        previous_handlers = {sig: signal.signal(sig, self.request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        self.state["phase"] = "RUNNING"
        self.message("Run directory: %s" % self.directory)
        self.message("Uniform solver parameters: %s" % json.dumps(self.config["parameters"], sort_keys=True))
        experiment_environment = self.config.get("experimental_environment", {})
        greedy_rl = experiment_environment.get("HLS_GREEDY_RL", "1") != "0"
        tabu_rl = experiment_environment.get("HLS_TABU_RL", "1") != "0"
        self.message(
            "Greedy RL=%s; Tabu RL=%s; %d tasks; random seeds retained in seeds.csv" % (
                "ON" if greedy_rl else "OFF",
                "ON" if tabu_rl else "OFF",
                len(self.state["tasks"])
            )
        )
        self.message("Executable SHA256: %s" % self.config["artifacts"]["executable"]["sha256"])
        self.resources.sample(set())
        next_check = time.monotonic() + min(1, self.settings.check_interval)
        write_results(self.directory, self.config, self.state)
        try:
            while True:
                changed = self.reap()
                if self.stop:
                    self.message("Stopping owned solver process groups; completed results and seeds are preserved.")
                    self.state["phase"] = "STOPPING"
                    break
                pending = [task for task in self.state["tasks"] if task["status"] == "PENDING"]
                if not pending and not self.active:
                    break
                if time.monotonic() >= next_check:
                    snapshot = self.resources.sample(set(self.active))
                    self.state["peak_rss"] = max([self.state["peak_rss"]] + list(snapshot["rss"].values()))
                    target, job_memory, reserve = admission(snapshot, self.settings, self.state["peak_rss"])
                    # Shrink admission immediately; grow gradually to measure model memory first.
                    self.limit = min(target, len(self.active) + self.settings.ramp_up)
                    for task in pending[:max(0, self.limit - len(self.active))]:
                        if self.stop:
                            break
                        if self.launch(task):
                            changed = True
                    self.report(snapshot, job_memory, reserve)
                    next_check = time.monotonic() + self.settings.check_interval
                if changed:
                    write_results(self.directory, self.config, self.state)
                time.sleep(min(0.25, self.settings.check_interval))
        finally:
            for item in self.active.values():
                self.terminate(item, "INTERRUPTED")
            while self.active:
                self.reap()
                if self.active:
                    time.sleep(0.1)
            ok = sum(task["status"] == "OK" for task in self.state["tasks"])
            self.state["phase"] = "COMPLETE" if ok == len(self.state["tasks"]) else "INTERRUPTED" if self.stop else "INCOMPLETE"
            write_results(self.directory, self.config, self.state)
            snapshot = self.resources.sample(set())
            _, job_memory, reserve = admission(snapshot, self.settings, self.state["peak_rss"])
            self.report(snapshot, job_memory, reserve)
            self.message("Result file: %s" % (self.directory / "result.csv"))
            self.message("Statistics and L2S comparison: %s" % (self.directory / "summary.csv"))
            self.log.close()
            self.metrics.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        return 128 + self.stop_signal if self.stop else 0 if self.state["phase"] == "COMPLETE" else 1


def main():
    settings = arguments()
    resources = Resources()
    snapshot = resources.sample(set())
    limit, job_memory, reserve = admission(snapshot, settings, 0)
    print("Effective CPU quota: %.2f cores; memory limit: %.2f GiB; available: %.2f GiB" % (
        snapshot["cores"], snapshot["memory_total"] / GIB, snapshot["memory_available"] / GIB), flush=True)
    print("Resource-based concurrency ceiling now: %d; CPUs/job: %d; memory estimate/job: %.2f GiB; memory reserve: %.2f GiB" % (
        limit, settings.cpus_per_job, job_memory / GIB, reserve / GIB), flush=True)
    if settings.cpus_per_job > len(resources.affinity):
        raise ValueError("CPUS_PER_JOB exceeds the available CPU affinity set.")
    if settings.dry_run:
        config = json.loads((Path(settings.log_dir) / "config.json").read_text(encoding="utf-8")) if settings.resume else None
        instances = config["instances"] if config else settings.instances
        runs = config["runs"] if config else settings.runs
        parameters = config["parameters"] if config else requested_solver()
        print("Plan: %s; %d runs each; %d experiments; COPT=%ss; total=%ss; both RL policies enabled." % (
            " ".join(instances), runs, len(instances) * runs, parameters["COPT_TIME"], parameters["TOTAL_TIME"]))
        return 0
    stamp = os.environ.get("RUN_TIMESTAMP", datetime.now(CHINA_TIME).strftime("%Y%m%d_%H%M%S_%f"))
    directory = Path(settings.log_dir or str(ROOT / "run_logs" / stamp)).resolve()
    if settings.resume:
        if not (directory / "config.json").is_file():
            raise ValueError("No experiment manifest to resume in %s" % directory)
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory is not empty; use RESUME=1 to continue this batch.")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another launcher holds the lock for %s" % directory)
        config, state = resume_run(directory) if settings.resume else create_run(directory, settings, requested_solver())
        runner = Runner(directory, settings, config, state, resources)
        return runner.execute()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print("Error: %s" % error, file=sys.stderr)
        sys.exit(2)
