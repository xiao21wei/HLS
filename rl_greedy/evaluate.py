from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .fjsp_env import (
    FjspInstance,
    Schedule,
    apply_action,
    build_initial_schedule,
    generate_actions,
)
from .policy import (
    load_checkpoint,
    load_exported_actor,
    select_hybrid_indices,
)


@dataclass
class SearchResult:
    initial_makespan: int
    final_makespan: int
    exact_evaluations: int
    steps: int
    ranking_states: int = 0
    hybrid_best_recall: float = 0.0
    improvement_recall: float = 0.0
    mean_regret: float = 0.0
    selected_valid_rate: float = 0.0
    median_model_best_rank: float = 0.0


@dataclass
class PolicyDiagnostics:
    states: int = 0
    states_with_improvement: int = 0
    hybrid_best_hits: int = 0
    improvement_hits: int = 0
    regret_sum: float = 0.0
    valid_rate_sum: float = 0.0
    model_best_ranks: list[int] = field(default_factory=list)

    def result_values(self) -> tuple[int, float, float, float, float, float]:
        if self.states == 0:
            return 0, 0.0, 1.0, 0.0, 0.0, 0.0
        return (
            self.states,
            self.hybrid_best_hits / self.states,
            (
                self.improvement_hits / self.states_with_improvement
                if self.states_with_improvement else 1.0
            ),
            self.regret_sum / self.states,
            self.valid_rate_sum / self.states,
            float(np.median(self.model_best_ranks)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and RL-filtered HLS Greedy neighborhoods."
    )
    parser.add_argument("instances", nargs="+", help="FJSP .fjs files")
    parser.add_argument(
        "--model",
        default="rl_greedy/models/greedy_policy.txt",
    )
    parser.add_argument("--top-k", type=int, default=96)
    parser.add_argument(
        "--dynamic-top-k",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rule-fraction", type=float, default=0.25)
    parser.add_argument("--exploration", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(path: str, device: torch.device):
    if path.endswith(".txt"):
        return load_exported_actor(path, device)
    return load_checkpoint(path, device)


def select_best_candidate(
    instance: FjspInstance,
    schedule: Schedule,
    actions,
    model,
    top_k: int | None,
    device: torch.device,
    rng: np.random.Generator | None = None,
    dynamic_top_k: bool = True,
    rule_fraction: float = 0.25,
    exploration_fraction: float = 0.15,
    temperature: float = 1.0,
    diagnostics: PolicyDiagnostics | None = None,
) -> tuple[Schedule, int]:
    if not actions:
        return schedule, 0
    selected_actions = actions
    evaluated_all = None
    if model is not None and top_k is not None:
        features = torch.tensor(
            [action.features for action in actions],
            dtype=torch.float32,
            device=device,
        )
        indices = select_hybrid_indices(
            model,
            features,
            top_k,
            instance.operation_count,
            rng if rng is not None else np.random.default_rng(0),
            dynamic_top_k=dynamic_top_k,
            rule_fraction=rule_fraction,
            exploration_fraction=exploration_fraction,
            temperature=temperature,
        )
        selected_actions = [actions[index] for index in indices]
        if diagnostics is not None:
            evaluated_all = [
                apply_action(instance, schedule, action) for action in actions
            ]
            gains = np.asarray([
                (
                    (schedule.makespan - candidate.makespan) /
                    max(1, schedule.makespan)
                    if candidate.valid else -0.20
                )
                for candidate in evaluated_all
            ], dtype=np.float32)
            valid = np.asarray(
                [candidate.valid for candidate in evaluated_all],
                dtype=np.bool_,
            )
            scores = model.action_scores(features).detach().cpu().numpy()
            model_order = np.argsort(-scores, kind="stable")
            best_gain = float(np.max(gains))
            best_indices = set(
                np.flatnonzero(np.isclose(gains, best_gain)).tolist()
            )
            diagnostics.states += 1
            diagnostics.hybrid_best_hits += int(
                any(index in best_indices for index in indices)
            )
            has_improvement = bool(np.any(gains > 0))
            if has_improvement:
                diagnostics.states_with_improvement += 1
                diagnostics.improvement_hits += int(
                    np.any(gains[np.asarray(indices, dtype=np.int64)] > 0)
                )
            selected_best_gain = float(np.max(
                gains[np.asarray(indices, dtype=np.int64)]
            ))
            diagnostics.regret_sum += max(0.0, best_gain - selected_best_gain)
            diagnostics.valid_rate_sum += float(np.mean(
                valid[np.asarray(indices, dtype=np.int64)]
            ))
            diagnostics.model_best_ranks.append(next(
                rank
                for rank, index in enumerate(model_order, start=1)
                if int(index) in best_indices
            ))
    best = schedule
    evaluation_count = 0
    for selected_position, action in enumerate(selected_actions):
        evaluation_count += 1
        candidate = (
            evaluated_all[indices[selected_position]]
            if evaluated_all is not None else
            apply_action(instance, schedule, action)
        )
        if candidate.valid and candidate.makespan < best.makespan:
            best = candidate
    return best, evaluation_count


def run_search(
    instance: FjspInstance,
    initial: Schedule,
    model,
    top_k: int | None,
    steps: int,
    device: torch.device,
    rng: np.random.Generator,
    dynamic_top_k: bool,
    rule_fraction: float,
    exploration_fraction: float,
    temperature: float,
) -> SearchResult:
    current = initial
    total_evaluations = 0
    completed_steps = 0
    diagnostics = PolicyDiagnostics()
    for step in range(steps):
        actions = generate_actions(
            instance,
            current,
            include_sequence=True,
            include_machine=True,
            include_swap=False,
            candidate_window=step,
        )
        best, evaluations = select_best_candidate(
            instance, current, actions, model, top_k, device,
            rng, dynamic_top_k, rule_fraction, exploration_fraction,
            temperature, diagnostics,
        )
        total_evaluations += evaluations
        if best.makespan >= current.makespan:
            near_actions = generate_actions(
                instance,
                current,
                near_critical=True,
                include_sequence=True,
                include_machine=True,
                include_swap=False,
                candidate_window=step,
            )
            best, evaluations = select_best_candidate(
                instance, current, near_actions, model, top_k, device,
                rng, dynamic_top_k, rule_fraction, exploration_fraction,
                temperature, diagnostics,
            )
            total_evaluations += evaluations
        if best.makespan >= current.makespan:
            swap_actions = generate_actions(
                instance,
                current,
                include_sequence=False,
                include_machine=False,
                include_swap=True,
                candidate_window=step,
            )
            best, evaluations = select_best_candidate(
                instance, current, swap_actions, model, top_k, device,
                rng, dynamic_top_k, rule_fraction, exploration_fraction,
                temperature, diagnostics,
            )
            total_evaluations += evaluations
        if best.makespan >= current.makespan:
            break
        current = best
        completed_steps += 1
    diagnostics_values = diagnostics.result_values()
    return SearchResult(
        initial.makespan,
        current.makespan,
        total_evaluations,
        completed_steps,
        *diagnostics_values,
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.steps <= 0:
        raise ValueError("top-k and steps must be positive")
    if not 0.0 <= args.rule_fraction <= 0.5:
        raise ValueError("rule-fraction must be between 0 and 0.5")
    if not 0.0 <= args.exploration <= 0.5:
        raise ValueError("exploration must be between 0 and 0.5")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    device = resolve_device(args.device)
    model = load_model(args.model, device).eval()
    model_path = Path(args.model).resolve()
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    rows = []
    for offset, path in enumerate(args.instances):
        instance = FjspInstance.from_file(path)
        initial = build_initial_schedule(
            instance,
            np.random.default_rng(args.seed + offset),
        )
        baseline = run_search(
            instance,
            initial,
            model=None,
            top_k=None,
            steps=args.steps,
            device=device,
            rng=np.random.default_rng(args.seed + offset + 1_000_000),
            dynamic_top_k=False,
            rule_fraction=0.0,
            exploration_fraction=0.0,
            temperature=args.temperature,
        )
        rl_result = run_search(
            instance,
            initial,
            model=model,
            top_k=args.top_k,
            steps=args.steps,
            device=device,
            rng=np.random.default_rng(args.seed + offset + 2_000_000),
            dynamic_top_k=args.dynamic_top_k,
            rule_fraction=args.rule_fraction,
            exploration_fraction=args.exploration,
            temperature=args.temperature,
        )
        row = {
            "instance": instance.name,
            "run": 1,
            "seed": args.seed + offset,
            "model": str(model_path),
            "model_sha256": model_sha256,
            "top_k": args.top_k,
            "dynamic_top_k": args.dynamic_top_k,
            "rule_fraction": args.rule_fraction,
            "exploration_fraction": args.exploration,
            "temperature": args.temperature,
            "max_steps": args.steps,
            "initial": initial.makespan,
            "baseline": baseline.final_makespan,
            "rl": rl_result.final_makespan,
            "rl_minus_baseline": rl_result.final_makespan - baseline.final_makespan,
            "baseline_evaluations": baseline.exact_evaluations,
            "rl_evaluations": rl_result.exact_evaluations,
            "baseline_steps": baseline.steps,
            "rl_steps": rl_result.steps,
            "ranking_states": rl_result.ranking_states,
            "hybrid_best_recall": rl_result.hybrid_best_recall,
            "improvement_recall": rl_result.improvement_recall,
            "mean_regret": rl_result.mean_regret,
            "selected_valid_rate": rl_result.selected_valid_rate,
            "median_model_best_rank": rl_result.median_model_best_rank,
        }
        rows.append(row)
        print(
            f"{instance.name}: initial={row['initial']} "
            f"baseline={row['baseline']} rl={row['rl']} "
            f"difference={row['rl_minus_baseline']:+d} "
            f"evaluations={row['baseline_evaluations']}/"
            f"{row['rl_evaluations']} "
            f"best_recall={row['hybrid_best_recall']:.3f} "
            f"improvement_recall={row['improvement_recall']:.3f} "
            f"regret={row['mean_regret']:.6f}"
        )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"result_file={output_path.resolve()}")


if __name__ == "__main__":
    main()
