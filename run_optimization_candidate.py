#!/usr/bin/env python3
"""Use the audited batch runner for one uniform experimental HLS candidate."""

import json
import os
from pathlib import Path
import sys

import run_adaptive_brandimarte as batch


class CandidateRunner(batch.Runner):
    def environment(self):
        result = super().environment()
        result.update(self.config.get("experimental_environment", {}))
        return result

    def launch(self, task):
        complete_affinity = self.resources.affinity
        width = (self.settings.max_parallel or max(1, int(self.resources.limits()["cores"]))) * self.settings.cpus_per_job
        self.resources.affinity = complete_affinity[:width]
        try:
            return super().launch(task)
        finally:
            self.resources.affinity = complete_affinity


def run_candidate(settings, directory):
    if settings.resume:
        config, state = batch.resume_run(directory)
    else:
        if directory.exists() and any(path.name != ".runner.lock" for path in directory.iterdir()):
            raise ValueError("Candidate directory already contains artifacts.")
        directory.mkdir(parents=True, exist_ok=True)
        config, state = batch.create_run(directory, settings, batch.requested_solver())
        extras = json.loads(os.environ.get("CANDIDATE_HLS_ENV", "{}"))
        if not isinstance(extras, dict) or any(not key.startswith("HLS_") or not isinstance(value, str) for key, value in extras.items()):
            raise ValueError("CANDIDATE_HLS_ENV must map HLS_* names to string values.")
        config["experimental_environment"] = extras
        seeds_file = os.environ.get("PAIRED_SEEDS_FILE")
        if seeds_file:
            with Path(seeds_file).open(newline="") as handle:
                seeds = {(row["instance"], int(row["run"])): int(row["seed"]) for row in batch.csv.DictReader(handle)}
            for task in state["tasks"]:
                seed = seeds[(task["instance"], task["run"])]
                if not 0 <= seed <= 4294967295:
                    raise ValueError("Invalid paired seed.")
                task["seed"] = seed
            batch.write_csv(directory / "seeds.csv", [["instance", "run", "seed"]] + [
                [task["instance"], task["run"], task["seed"]] for task in state["tasks"]])
        batch.atomic_json(directory / "config.json", config)
        batch.atomic_json(directory / "state.json", state)
    resources = batch.Resources()
    offset = int(os.environ.get("CPU_AFFINITY_OFFSET", "0")) % len(resources.affinity)
    resources.affinity = resources.affinity[offset:] + resources.affinity[:offset]
    return CandidateRunner(directory, settings, config, state, resources).execute()


def main():
    settings = batch.arguments()
    if settings.dry_run:
        return batch.main()
    if not settings.log_dir:
        raise ValueError("Set LOG_DIR for a named, auditable candidate.")
    directory = Path(settings.log_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".runner.lock").open("a") as lock:
        try:
            batch.fcntl.flock(lock, batch.fcntl.LOCK_EX | batch.fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another launcher holds this experiment directory.")
        return run_candidate(settings, directory)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print("Error: %s" % error, file=sys.stderr)
        sys.exit(2)
