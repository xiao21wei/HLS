"""Resource limits and subprocess lifecycle tests; no optimization runs here."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adaptive_runner", ROOT / "run_adaptive_brandimarte.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ResourceTests(unittest.TestCase):
    def test_cgroup_quota_and_memory_override_virtual_proc(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            proc, cgroup = base / "proc", base / "cgroup"
            (proc / "self").mkdir(parents=True)
            cgroup.mkdir()
            (proc / "self/cgroup").write_text("0::/\n")
            (proc / "meminfo").write_text("MemTotal: 67108864 kB\nMemAvailable: 67108864 kB\n")
            (cgroup / "cpu.max").write_text("1601000 100000\n")
            (cgroup / "cpu.stat").write_text("usage_usec 1000000\n")
            (cgroup / "memory.max").write_text(str(64 * runner.GIB))
            (cgroup / "memory.current").write_text(str(14 * runner.GIB))
            resources = runner.Resources(proc, cgroup, affinity=list(range(320)))
            snapshot = resources.limits()
            self.assertAlmostEqual(snapshot["cores"], 16.01)
            self.assertEqual(snapshot["memory_available"], 50 * runner.GIB)
            (cgroup / "limited").mkdir()
            (proc / "self/cgroup").write_text("0::/limited\n")
            (cgroup / "limited/cpu.max").write_text("max 100000\n")
            self.assertAlmostEqual(runner.Resources(proc, cgroup, affinity=list(range(320))).limits()["cores"], 16.01)

    def test_external_load_and_reserved_memory_stop_admission(self):
        settings = SimpleNamespace(cpu_usage_limit=0.85, reserve_cpus=1, cpus_per_job=1,
                                   job_memory_gb=2, reserve_memory_gb=4, max_parallel=0)
        snapshot = {"cores": 16.01, "memory_total": 64 * runner.GIB, "memory_available": 50 * runner.GIB,
                    "busy_cores": 0, "external_cores": 0, "rss": {}}
        idle = runner.admission(snapshot, settings, 0)[0]
        self.assertGreater(idle, 5)
        self.assertLessEqual(idle, 16)
        snapshot["external_cores"] = 9
        self.assertLess(runner.admission(snapshot, settings, 0)[0], idle)
        snapshot["external_cores"] = 0
        snapshot["memory_available"] = 5 * runner.GIB
        self.assertEqual(runner.admission(snapshot, settings, 0)[0], 0)
        snapshot["cores"] = 1
        snapshot["memory_available"] = 50 * runner.GIB
        self.assertEqual(runner.admission(snapshot, settings, 0)[0], 1)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "batch"
        self.env = dict(os.environ, EXECUTABLE=str(ROOT / "tests/fixtures/adaptive_solver.py"),
                        LOG_DIR=str(self.directory), INSTANCES="Mk01 Mk02", RUNS="2",
                        MAX_PARALLEL="2", CPUS_PER_JOB="1", CHECK_INTERVAL="0.1",
                        JOB_MEMORY_GB="0.01", MEMORY_RESERVE_GB="0", FAKE_SLEEP="0.4",
                        HLS_GA_ACTIVE_DECODE="1")
        self.env.pop("RESUME", None)
        self.env.pop("FAKE_FAIL_INSTANCE", None)
        self.env.pop("FAKE_CHILD", None)
        self.command = [sys.executable, str(ROOT / "run_adaptive_brandimarte.py")]

    def tearDown(self):
        self.temporary.cleanup()

    def launch(self, **changes):
        return subprocess.run(self.command, env=dict(self.env, **changes), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=20)

    def test_four_runs_unique_seeds_affinity_and_latest_reference(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout)
        state = json.loads((self.directory / "state.json").read_text())
        self.assertEqual(len({task["seed"] for task in state["tasks"]}), 4)
        self.assertTrue(all(task["status"] == "OK" for task in state["tasks"]))
        for task in state["tasks"]:
            log = (self.directory / (task["label"] + ".txt")).read_text()
            self.assertIn("fixture_cpus: 1", log)
            self.assertIn("fixture_active_decode: 0", log)
        with (self.directory / "result.csv").open() as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["instance", "run1", "run2"])
        self.assertEqual(len(rows), 3)
        with (self.directory / "summary.csv").open() as handle:
            summaries = {row["instance"]: row for row in csv.DictReader(handle)}
        actual_reference = runner.reference_means(ROOT / "l2s_brandimarte.csv")
        self.assertAlmostEqual(float(summaries["Mk02"]["l2s_mean"]), actual_reference["Mk02"])
        with (self.directory / "scheduler.csv").open() as handle:
            self.assertTrue(any(int(row["running"]) == 2 for row in csv.DictReader(handle)))

    def test_failure_excluded_and_resume_preserves_success_and_seeds(self):
        failed = self.launch(FAKE_FAIL_INSTANCE="Mk02")
        self.assertEqual(failed.returncode, 1, failed.stdout)
        before = json.loads((self.directory / "state.json").read_text())
        seed_hash = runner.sha256(self.directory / "seeds.csv")
        good_hash = runner.sha256(self.directory / "Mk01_run01.txt")
        with (self.directory / "summary.csv").open() as handle:
            summaries = {row["instance"]: row for row in csv.DictReader(handle)}
        self.assertEqual(summaries["Mk02"]["completed"], "0")
        self.assertEqual(summaries["Mk02"]["comparison"], "INCOMPLETE")
        resumed = self.launch(RESUME="1", FAKE_FAIL_INSTANCE="")
        self.assertEqual(resumed.returncode, 0, resumed.stdout)
        after = json.loads((self.directory / "state.json").read_text())
        self.assertEqual([task["seed"] for task in before["tasks"]], [task["seed"] for task in after["tasks"]])
        self.assertEqual(seed_hash, runner.sha256(self.directory / "seeds.csv"))
        self.assertEqual(good_hash, runner.sha256(self.directory / "Mk01_run01.txt"))
        self.assertEqual(len(list((self.directory / "attempts").glob("*.txt"))), 2)

    def test_interrupt_kills_owned_group_and_resume_uses_same_seed(self):
        process = subprocess.Popen(self.command, env=dict(self.env, FAKE_SLEEP="30", FAKE_CHILD="1"),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        child_pid = None
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                log = runner.read_text(self.directory / "Mk01_run01.txt")
                match = re.search(r"fixture_child_pid: (\d+)", log)
                if match:
                    child_pid = int(match.group(1))
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            self.assertIsNotNone(child_pid)
            process.terminate()
            output = process.communicate(timeout=10)[0]
            self.assertEqual(process.returncode, 143, output)
            child_stat = runner.read_text(Path("/proc") / str(child_pid) / "stat")
            self.assertTrue(not child_stat or child_stat[child_stat.rfind(")") + 2:].split()[0] == "Z")
            state = json.loads((self.directory / "state.json").read_text())
            self.assertTrue(any(task["status"] == "INTERRUPTED" for task in state["tasks"]))
            resumed = self.launch(RESUME="1", FAKE_CHILD="0", FAKE_SLEEP="0.1")
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            after = json.loads((self.directory / "state.json").read_text())
            self.assertEqual([task["seed"] for task in state["tasks"]], [task["seed"] for task in after["tasks"]])
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
