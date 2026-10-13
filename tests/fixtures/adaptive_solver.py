#!/usr/bin/env python3
"""Short process fixture for admission, failure, and signal integration checks."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

instance = Path(sys.argv[1]).stem
print("greedy_rl_enabled: true", flush=True)
print("tabu_rl_enabled: true", flush=True)
print("fixture_cpus: " + str(len(os.sched_getaffinity(0))), flush=True)
print("fixture_active_decode: " + os.environ["HLS_GA_ACTIVE_DECODE"], flush=True)
if os.environ.get("FAKE_CHILD") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(300)"])
    print("fixture_child_pid: %d" % child.pid, flush=True)
time.sleep(float(os.environ.get("FAKE_SLEEP", "0.4")))
if instance == os.environ.get("FAKE_FAIL_INSTANCE"):
    print("Final makespan: 1", flush=True)  # A final line alone must not count as success.
    sys.exit(7)
print("Final makespan: %d" % (173 + int(sys.argv[-1]) % 3), flush=True)
