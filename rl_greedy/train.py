from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from .fjsp_env import FjspInstance, GreedyRlEnv
from .policy import (
    GreedyActorCritic,
    RankingSample,
    Transition,
    export_actor,
    ppo_update,
    ranking_metrics,
    save_checkpoint,
    supervised_ranking_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the self-contained HLS Greedy neighborhood policy."
    )
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="Optional .fjs training instance; may be supplied multiple times.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ranking-epochs", type=int, default=2)
    parser.add_argument("--ranking-temperature", type=float, default=0.05)
    parser.add_argument("--ranking-state-batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--min-jobs", type=int, default=6)
    parser.add_argument("--max-jobs", type=int, default=12)
    parser.add_argument("--min-machines", type=int, default=4)
    parser.add_argument("--max-machines", type=int, default=8)
    parser.add_argument("--min-operations", type=int, default=3)
    parser.add_argument("--max-operations", type=int, default=15)
    parser.add_argument("--max-machine-options", type=int, default=6)
    parser.add_argument("--min-processing-time", type=int, default=1)
    parser.add_argument("--max-processing-time", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        default="rl_greedy/models/greedy_policy.pt",
    )
    parser.add_argument(
        "--export",
        default="rl_greedy/models/greedy_policy.txt",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.episodes_per_epoch <= 0 or args.horizon <= 0:
        raise ValueError("epochs, episodes-per-epoch and horizon must be positive")
    if args.ranking_epochs < 0 or args.ppo_epochs < 0:
        raise ValueError("ranking-epochs and ppo-epochs cannot be negative")
    if args.min_jobs <= 0 or args.max_jobs < args.min_jobs:
        raise ValueError("invalid job-count range")
    if args.min_machines <= 0 or args.max_machines < args.min_machines:
        raise ValueError("invalid machine-count range")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    fixed_instances = [FjspInstance.from_file(path) for path in args.instance]
    model = GreedyActorCritic(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_epoch_makespan = float("inf")
    best_ranking_score = -float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    print(
        f"device={device} torch={torch.__version__} "
        f"fixed_instances={len(fixed_instances)}"
    )
    for epoch in range(1, args.epochs + 1):
        transitions: list[Transition] = []
        ranking_samples: list[RankingSample] = []
        episode_rewards = []
        episode_best = []
        action_counts = []
        for episode in range(args.episodes_per_epoch):
            if fixed_instances:
                instance = fixed_instances[
                    (epoch * args.episodes_per_epoch + episode) %
                    len(fixed_instances)
                ]
            else:
                instance = FjspInstance.synthetic(
                    rng,
                    job_count=int(rng.integers(args.min_jobs, args.max_jobs + 1)),
                    machine_count=int(
                        rng.integers(args.min_machines, args.max_machines + 1)
                    ),
                    min_operations=args.min_operations,
                    max_operations=args.max_operations,
                    max_machine_options=args.max_machine_options,
                    min_processing_time=args.min_processing_time,
                    max_processing_time=args.max_processing_time,
                )
            env = GreedyRlEnv(instance, rng, args.horizon)
            episode_start = len(transitions)
            total_reward = 0.0
            for _ in range(args.horizon):
                candidates = env.available_actions()
                if not candidates:
                    break
                feature_array = np.asarray(
                    [candidate[0].features for candidate in candidates],
                    dtype=np.float32,
                )
                gains = np.asarray([
                    (
                        (env.current.makespan - candidate[1].makespan) /
                        max(1, env.current.makespan)
                        if candidate[1].valid else -0.20
                    )
                    for candidate in candidates
                ], dtype=np.float32)
                valid = np.asarray(
                    [candidate[1].valid for candidate in candidates],
                    dtype=np.bool_,
                )
                ranking_samples.append(RankingSample(
                    feature_array,
                    gains,
                    valid,
                ))
                feature_tensor = torch.as_tensor(feature_array, device=device)
                action_index, log_probability, value = model.select_action(
                    feature_tensor
                )
                reward, done, _ = env.step(candidates[action_index])
                transitions.append(Transition(
                    feature_array,
                    action_index,
                    log_probability,
                    reward,
                    value,
                    done,
                ))
                total_reward += reward
                action_counts.append(len(candidates))
                if done:
                    break
            if len(transitions) > episode_start:
                transitions[-1].done = True
            episode_rewards.append(total_reward)
            episode_best.append(env.best.makespan)

        metrics = ppo_update(
            model,
            optimizer,
            transitions,
            device,
            epochs=args.ppo_epochs,
        )
        if len(ranking_samples) >= 5:
            split_index = max(1, int(len(ranking_samples) * 0.80))
            ranking_train_samples = ranking_samples[:split_index]
            ranking_validation_samples = ranking_samples[split_index:]
        else:
            ranking_train_samples = ranking_samples
            ranking_validation_samples = ranking_samples
        ranking_update_metrics = supervised_ranking_update(
            model,
            optimizer,
            ranking_train_samples,
            device,
            epochs=args.ranking_epochs,
            target_temperature=args.ranking_temperature,
            state_batch_size=args.ranking_state_batch_size,
        )
        recall_metrics = ranking_metrics(
            model,
            ranking_validation_samples,
            device,
        )
        mean_best = float(np.mean(episode_best)) if episode_best else 0.0
        best_epoch_makespan = min(best_epoch_makespan, mean_best)
        ranking_score = (
            recall_metrics.get("improvement_recall@96", 0.0)
            - recall_metrics.get("regret@96", 0.0)
        )
        if ranking_score > best_ranking_score:
            best_ranking_score = ranking_score
            best_model_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:04d} transitions={len(transitions)} "
            f"reward={np.mean(episode_rewards):.6f} "
            f"mean_best={mean_best:.2f} "
            f"mean_actions={np.mean(action_counts) if action_counts else 0:.1f} "
            f"ranking_train={len(ranking_train_samples)} "
            f"ranking_validation={len(ranking_validation_samples)} "
            f"ranking_loss={ranking_update_metrics['ranking_loss']:.6f} "
            f"ppo_loss={metrics['loss']:.6f} "
            f"recall@96={recall_metrics.get('recall@96', 0.0):.3f} "
            f"improvement_recall@96="
            f"{recall_metrics.get('improvement_recall@96', 0.0):.3f} "
            f"regret@96={recall_metrics.get('regret@96', 0.0):.6f} "
            f"median_best_rank="
            f"{recall_metrics.get('median_best_rank', 0.0):.1f}"
        )

    model.load_state_dict(best_model_state)
    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "horizon": args.horizon,
        "ranking_epochs": args.ranking_epochs,
        "ranking_temperature": args.ranking_temperature,
        "min_operations": args.min_operations,
        "max_operations": args.max_operations,
        "max_machine_options": args.max_machine_options,
        "min_processing_time": args.min_processing_time,
        "max_processing_time": args.max_processing_time,
        "training_instances": [str(Path(path)) for path in args.instance],
        "synthetic_training": not bool(args.instance),
        "best_epoch_mean_makespan": best_epoch_makespan,
        "best_ranking_score": best_ranking_score,
    }
    save_checkpoint(model, args.checkpoint, metadata)
    export_actor(model, args.export)
    print(f"checkpoint={args.checkpoint}")
    print(f"export={args.export}")


if __name__ == "__main__":
    main()
