from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FEATURE_COUNT = 16
MOVE_SEQUENCE = 0
MOVE_MACHINE = 1
MOVE_SWAP = 2

OperationId = tuple[int, int]


@dataclass(frozen=True)
class Operation:
    job_id: int
    process_id: int
    options: tuple[tuple[int, int], ...]

    def processing_time(self, machine_id: int) -> int:
        for option_machine, process_time in self.options:
            if option_machine == machine_id:
                return process_time
        raise ValueError(
            f"operation {(self.job_id, self.process_id)} cannot use machine "
            f"{machine_id}"
        )


@dataclass(frozen=True)
class FjspInstance:
    jobs: tuple[tuple[Operation, ...], ...]
    machine_count: int
    name: str = "instance"

    @property
    def operation_count(self) -> int:
        return sum(len(job) for job in self.jobs)

    @property
    def operations(self) -> dict[OperationId, Operation]:
        return {
            (operation.job_id, operation.process_id): operation
            for job in self.jobs
            for operation in job
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "FjspInstance":
        input_path = Path(path)
        lines = [
            line.strip()
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"empty FJSP instance: {input_path}")
        header = lines[0].split()
        job_count = int(header[0])
        machine_count = int(header[1])
        if len(lines) - 1 < job_count:
            raise ValueError(f"missing job rows in {input_path}")

        raw_jobs: list[list[tuple[list[tuple[int, int]], int]]] = []
        machine_ids: list[int] = []
        for job_id in range(job_count):
            tokens = [int(value) for value in lines[job_id + 1].split()]
            cursor = 0
            process_count = tokens[cursor]
            cursor += 1
            raw_job: list[tuple[list[tuple[int, int]], int]] = []
            for process_id in range(process_count):
                option_count = tokens[cursor]
                cursor += 1
                options: list[tuple[int, int]] = []
                for _ in range(option_count):
                    machine_id = tokens[cursor]
                    process_time = tokens[cursor + 1]
                    cursor += 2
                    options.append((machine_id, process_time))
                    machine_ids.append(machine_id)
                raw_job.append((options, process_id))
            if cursor != len(tokens):
                raise ValueError(f"unexpected tokens in job {job_id} of {input_path}")
            raw_jobs.append(raw_job)

        one_based = bool(machine_ids) and min(machine_ids) >= 1 and max(machine_ids) == machine_count
        jobs: list[tuple[Operation, ...]] = []
        for job_id, raw_job in enumerate(raw_jobs):
            operations = []
            for options, process_id in raw_job:
                normalized = tuple(
                    (machine_id - 1 if one_based else machine_id, process_time)
                    for machine_id, process_time in options
                )
                operations.append(Operation(job_id, process_id, normalized))
            jobs.append(tuple(operations))
        return cls(tuple(jobs), machine_count, input_path.stem)

    @classmethod
    def synthetic(
        cls,
        rng: np.random.Generator,
        job_count: int,
        machine_count: int,
        min_operations: int = 3,
        max_operations: int = 15,
        max_machine_options: int = 6,
        min_processing_time: int = 1,
        max_processing_time: int = 20,
    ) -> "FjspInstance":
        if min_operations <= 0 or max_operations < min_operations:
            raise ValueError("invalid synthetic operation range")
        if max_machine_options <= 0:
            raise ValueError("max_machine_options must be positive")
        if min_processing_time <= 0 or max_processing_time < min_processing_time:
            raise ValueError("invalid synthetic processing-time range")
        jobs: list[tuple[Operation, ...]] = []
        for job_id in range(job_count):
            process_count = int(rng.integers(min_operations, max_operations + 1))
            operations = []
            for process_id in range(process_count):
                option_count = int(
                    rng.integers(1, min(machine_count, max_machine_options) + 1)
                )
                machines = rng.choice(
                    machine_count,
                    size=option_count,
                    replace=False,
                )
                options = tuple(
                    (
                        int(machine),
                        int(rng.integers(
                            min_processing_time,
                            max_processing_time + 1,
                        )),
                    )
                    for machine in machines
                )
                operations.append(Operation(job_id, process_id, options))
            jobs.append(tuple(operations))
        return cls(
            tuple(jobs),
            machine_count,
            f"synthetic-{job_count}x{machine_count}",
        )


@dataclass
class Schedule:
    machine_sequences: list[list[OperationId]]
    assignment: dict[OperationId, int]
    makespan: int
    start_times: dict[OperationId, int]
    slack: dict[OperationId, int]
    valid: bool = True

    def copy_sequences(self) -> list[list[OperationId]]:
        return [list(sequence) for sequence in self.machine_sequences]


@dataclass(frozen=True)
class GreedyAction:
    move_type: int
    operation: OperationId
    source_machine: int
    target_machine: int
    source_position: int
    target_position: int
    features: tuple[float, ...]


def evaluate_schedule(
    instance: FjspInstance,
    machine_sequences: Sequence[Sequence[OperationId]],
) -> Schedule:
    if len(machine_sequences) != instance.machine_count:
        return Schedule([], {}, 0, {}, {}, False)
    operations = instance.operations
    assignment: dict[OperationId, int] = {}
    for machine_id, sequence in enumerate(machine_sequences):
        for operation_id in sequence:
            if operation_id in assignment or operation_id not in operations:
                return Schedule([], {}, 0, {}, {}, False)
            try:
                operations[operation_id].processing_time(machine_id)
            except ValueError:
                return Schedule([], {}, 0, {}, {}, False)
            assignment[operation_id] = machine_id
    if len(assignment) != instance.operation_count:
        return Schedule([], {}, 0, {}, {}, False)

    successors: dict[OperationId, set[OperationId]] = {
        operation_id: set() for operation_id in operations
    }
    predecessors: dict[OperationId, set[OperationId]] = {
        operation_id: set() for operation_id in operations
    }

    def add_edge(source: OperationId, target: OperationId) -> None:
        if source == target or target in successors[source]:
            return
        successors[source].add(target)
        predecessors[target].add(source)

    for job in instance.jobs:
        for process_id in range(1, len(job)):
            add_edge((job[process_id - 1].job_id, process_id - 1),
                     (job[process_id].job_id, process_id))
    for sequence in machine_sequences:
        for position in range(1, len(sequence)):
            add_edge(sequence[position - 1], sequence[position])

    in_degree = {
        operation_id: len(operation_predecessors)
        for operation_id, operation_predecessors in predecessors.items()
    }
    ready = deque(sorted(
        operation_id for operation_id, degree in in_degree.items() if degree == 0
    ))
    topological_order: list[OperationId] = []
    start_times = {operation_id: 0 for operation_id in operations}
    completion_times: dict[OperationId, int] = {}
    while ready:
        operation_id = ready.popleft()
        topological_order.append(operation_id)
        duration = operations[operation_id].processing_time(assignment[operation_id])
        completion = start_times[operation_id] + duration
        completion_times[operation_id] = completion
        for successor in sorted(successors[operation_id]):
            start_times[successor] = max(start_times[successor], completion)
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
    if len(topological_order) != instance.operation_count:
        return Schedule([], {}, 0, {}, {}, False)

    makespan = max(completion_times.values(), default=0)
    latest_start: dict[OperationId, int] = {}
    for operation_id in reversed(topological_order):
        duration = operations[operation_id].processing_time(assignment[operation_id])
        if successors[operation_id]:
            latest_finish = min(
                latest_start[successor]
                for successor in successors[operation_id]
            )
        else:
            latest_finish = makespan
        latest_start[operation_id] = latest_finish - duration
    slack = {
        operation_id: latest_start[operation_id] - start_times[operation_id]
        for operation_id in operations
    }
    return Schedule(
        [list(sequence) for sequence in machine_sequences],
        assignment,
        makespan,
        start_times,
        slack,
        True,
    )


def build_initial_schedule(
    instance: FjspInstance,
    rng: np.random.Generator,
) -> Schedule:
    machine_sequences: list[list[OperationId]] = [
        [] for _ in range(instance.machine_count)
    ]
    machine_ready = [0] * instance.machine_count
    job_order = list(range(len(instance.jobs)))
    rng.shuffle(job_order)
    for job_id in job_order:
        job_ready = 0
        for operation in instance.jobs[job_id]:
            ranked_options = sorted(
                operation.options,
                key=lambda option: (
                    max(job_ready, machine_ready[option[0]]) + option[1],
                    option[1],
                    machine_ready[option[0]],
                    option[0],
                ),
            )
            machine_id, process_time = ranked_options[0]
            finish = max(job_ready, machine_ready[machine_id]) + process_time
            machine_sequences[machine_id].append((job_id, operation.process_id))
            machine_ready[machine_id] = finish
            job_ready = finish
    schedule = evaluate_schedule(instance, machine_sequences)
    if not schedule.valid:
        raise RuntimeError("failed to build a feasible initial schedule")
    return schedule


def _locations(schedule: Schedule) -> dict[OperationId, tuple[int, int]]:
    return {
        operation_id: (machine_id, position)
        for machine_id, sequence in enumerate(schedule.machine_sequences)
        for position, operation_id in enumerate(sequence)
    }


def _workloads(instance: FjspInstance, schedule: Schedule) -> list[int]:
    operations = instance.operations
    return [
        sum(operations[operation_id].processing_time(machine_id)
            for operation_id in sequence)
        for machine_id, sequence in enumerate(schedule.machine_sequences)
    ]


def _bounded_ratio(numerator: float, denominator: float) -> float:
    return float(np.clip(numerator / max(1.0, denominator), -2.0, 2.0))


def build_action_features(
    move_type: int,
    slack: int,
    current_processing_time: int,
    target_processing_time: int,
    source_workload: int,
    target_workload: int,
    source_position: int,
    target_position: int,
    source_length: int,
    target_length: int,
    operation_count: int,
    makespan: int,
) -> tuple[float, ...]:
    features = [0.0] * FEATURE_COUNT
    features[0] = 1.0 if move_type == MOVE_SEQUENCE else 0.0
    features[1] = 1.0 if move_type == MOVE_MACHINE else 0.0
    features[2] = 1.0 if move_type == MOVE_SWAP else 0.0
    features[3] = 1.0 if slack == 0 else 0.0
    features[4] = _bounded_ratio(slack, makespan)
    features[5] = _bounded_ratio(current_processing_time, makespan)
    features[6] = _bounded_ratio(target_processing_time, makespan)
    features[7] = _bounded_ratio(
        current_processing_time - target_processing_time, makespan
    )
    features[8] = _bounded_ratio(source_workload, makespan)
    features[9] = _bounded_ratio(target_workload, makespan)
    features[10] = _bounded_ratio(source_workload - target_workload, makespan)
    features[11] = _bounded_ratio(source_position, source_length - 1)
    features[12] = _bounded_ratio(target_position, target_length)
    features[13] = _bounded_ratio(
        abs(target_position - source_position),
        max(source_length, target_length),
    )
    features[14] = _bounded_ratio(source_length, operation_count)
    features[15] = _bounded_ratio(target_length, operation_count)
    return tuple(features)


def _rank_processes(
    instance: FjspInstance,
    schedule: Schedule,
    flexible_only: bool,
    near_critical_only: bool,
    candidate_window: int,
) -> list[OperationId]:
    operations = instance.operations
    locations = _locations(schedule)
    workloads = _workloads(instance, schedule)
    near_limit = max(6, min(20, schedule.makespan // 40))
    ranked: list[tuple[tuple[int, int, int, int, int], OperationId]] = []
    for operation_id, operation in operations.items():
        slack = schedule.slack[operation_id]
        if not near_critical_only and slack != 0:
            continue
        if near_critical_only and (slack == 0 or slack > near_limit):
            continue
        if flexible_only and len(operation.options) <= 1:
            continue
        machine_id, _ = locations[operation_id]
        current_time = operation.processing_time(machine_id)
        minimum_time = min(process_time for _, process_time in operation.options)
        alternative_gain = max(0, current_time - minimum_time)
        key = (
            slack,
            -alternative_gain if flexible_only else 0,
            -workloads[machine_id],
            -current_time,
            -schedule.start_times[operation_id],
        )
        ranked.append((key, operation_id))
    ranked.sort(key=lambda item: item[0])
    limit = 12 if near_critical_only else 28
    if len(ranked) <= limit:
        return [operation_id for _, operation_id in ranked]
    fixed = min(3 if near_critical_only else 8, limit)
    selected = ranked[:fixed]
    rotating_count = limit - fixed
    tail = ranked[fixed:]
    offset = (candidate_window * max(1, rotating_count)) % len(tail)
    selected.extend(
        tail[(offset + index) % len(tail)]
        for index in range(rotating_count)
    )
    return [operation_id for _, operation_id in selected]


def generate_actions(
    instance: FjspInstance,
    schedule: Schedule,
    near_critical: bool = False,
    include_sequence: bool = True,
    include_machine: bool = True,
    include_swap: bool = False,
    candidate_window: int = 0,
) -> list[GreedyAction]:
    operations = instance.operations
    locations = _locations(schedule)
    workloads = _workloads(instance, schedule)
    sequence_processes = _rank_processes(
        instance, schedule, False, near_critical, candidate_window
    )
    machine_processes = _rank_processes(
        instance, schedule, True, near_critical, candidate_window
    )
    actions: list[GreedyAction] = []

    if include_sequence:
        for operation_id in sequence_processes:
            machine_id, source_position = locations[operation_id]
            source_length = len(schedule.machine_sequences[machine_id])
            if source_length <= 1:
                continue
            process_time = operations[operation_id].processing_time(machine_id)
            for target_position in range(source_length):
                if target_position == source_position:
                    continue
                features = build_action_features(
                    MOVE_SEQUENCE,
                    schedule.slack[operation_id],
                    process_time,
                    process_time,
                    workloads[machine_id],
                    workloads[machine_id],
                    source_position,
                    target_position,
                    source_length,
                    source_length - 1,
                    instance.operation_count,
                    schedule.makespan,
                )
                actions.append(GreedyAction(
                    MOVE_SEQUENCE,
                    operation_id,
                    machine_id,
                    machine_id,
                    source_position,
                    target_position,
                    features,
                ))

    if include_machine:
        for operation_id in machine_processes:
            operation = operations[operation_id]
            source_machine, source_position = locations[operation_id]
            source_length = len(schedule.machine_sequences[source_machine])
            current_time = operation.processing_time(source_machine)
            options = sorted(
                operation.options,
                key=lambda option: (
                    workloads[option[0]] + option[1],
                    option[1],
                ),
            )
            seen_machines: set[int] = set()
            for target_machine, target_time in options:
                if target_machine == source_machine or target_machine in seen_machines:
                    continue
                seen_machines.add(target_machine)
                target_length = len(schedule.machine_sequences[target_machine])
                for target_position in range(target_length + 1):
                    features = build_action_features(
                        MOVE_MACHINE,
                        schedule.slack[operation_id],
                        current_time,
                        target_time,
                        workloads[source_machine],
                        workloads[target_machine],
                        source_position,
                        target_position,
                        source_length,
                        target_length,
                        instance.operation_count,
                        schedule.makespan,
                    )
                    actions.append(GreedyAction(
                        MOVE_MACHINE,
                        operation_id,
                        source_machine,
                        target_machine,
                        source_position,
                        target_position,
                        features,
                    ))

    if include_swap:
        for left in range(len(sequence_processes)):
            left_id = sequence_processes[left]
            left_machine, left_position = locations[left_id]
            for right in range(left + 1, len(sequence_processes)):
                right_id = sequence_processes[right]
                right_machine, right_position = locations[right_id]
                if left_machine != right_machine or left_position == right_position:
                    continue
                process_count = len(schedule.machine_sequences[left_machine])
                features = build_action_features(
                    MOVE_SWAP,
                    schedule.slack[left_id],
                    operations[left_id].processing_time(left_machine),
                    operations[right_id].processing_time(right_machine),
                    workloads[left_machine],
                    workloads[right_machine],
                    left_position,
                    right_position,
                    process_count,
                    process_count,
                    instance.operation_count,
                    schedule.makespan,
                )
                actions.append(GreedyAction(
                    MOVE_SWAP,
                    left_id,
                    left_machine,
                    right_machine,
                    left_position,
                    right_position,
                    features,
                ))
    return actions


def apply_action(
    instance: FjspInstance,
    schedule: Schedule,
    action: GreedyAction,
) -> Schedule:
    sequences = schedule.copy_sequences()
    if action.move_type == MOVE_SWAP:
        sequence = sequences[action.source_machine]
        sequence[action.source_position], sequence[action.target_position] = (
            sequence[action.target_position],
            sequence[action.source_position],
        )
    else:
        operation_id = sequences[action.source_machine].pop(action.source_position)
        if operation_id != action.operation:
            return Schedule([], {}, 0, {}, {}, False)
        sequences[action.target_machine].insert(action.target_position, operation_id)
    return evaluate_schedule(instance, sequences)


def evaluate_actions(
    instance: FjspInstance,
    schedule: Schedule,
    actions: Iterable[GreedyAction],
) -> list[tuple[GreedyAction, Schedule]]:
    evaluated = []
    for action in actions:
        candidate = apply_action(instance, schedule, action)
        if candidate.valid:
            evaluated.append((action, candidate))
    return evaluated


class GreedyRlEnv:
    def __init__(
        self,
        instance: FjspInstance,
        rng: np.random.Generator,
        horizon: int = 30,
    ) -> None:
        self.instance = instance
        self.rng = rng
        self.horizon = horizon
        self.current = build_initial_schedule(instance, rng)
        self.best = self.current
        self.step_count = 0
        self.stagnation = 0

    def reset(self) -> Schedule:
        self.current = build_initial_schedule(self.instance, self.rng)
        self.best = self.current
        self.step_count = 0
        self.stagnation = 0
        return self.current

    def available_actions(self) -> list[tuple[GreedyAction, Schedule]]:
        actions = generate_actions(
            self.instance,
            self.current,
            near_critical=self.stagnation >= 4,
            include_sequence=True,
            include_machine=True,
            include_swap=self.stagnation >= 2,
            candidate_window=self.step_count,
        )
        return [
            (action, apply_action(self.instance, self.current, action))
            for action in actions
        ]

    def step(
        self,
        evaluated_action: tuple[GreedyAction, Schedule],
    ) -> tuple[float, bool, dict[str, int]]:
        _, candidate = evaluated_action
        previous_makespan = self.current.makespan
        previous_best = self.best.makespan
        if not candidate.valid:
            self.stagnation += 1
            self.step_count += 1
            return -0.20, self.step_count >= self.horizon, {
                "current_makespan": self.current.makespan,
                "best_makespan": self.best.makespan,
            }
        self.current = candidate
        if candidate.makespan < self.best.makespan:
            self.best = candidate
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.step_count += 1

        local_gain = (previous_makespan - candidate.makespan) / max(
            1, previous_makespan
        )
        best_gain = max(0, previous_best - self.best.makespan) / max(
            1, previous_best
        )
        reward = local_gain + 2.0 * best_gain
        if candidate.makespan >= previous_makespan:
            reward -= 0.01
        done = self.step_count >= self.horizon
        return reward, done, {
            "current_makespan": self.current.makespan,
            "best_makespan": self.best.makespan,
        }
