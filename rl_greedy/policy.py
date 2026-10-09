from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .fjsp_env import FEATURE_COUNT


MODEL_MAGIC = "HLS_GREEDY_RL_V1"


class GreedyActorCritic(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.actor_hidden = nn.Linear(FEATURE_COUNT, hidden_size)
        self.actor_output = nn.Linear(hidden_size, 1)
        self.critic = nn.Sequential(
            nn.Linear(FEATURE_COUNT, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def action_scores(self, action_features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.actor_hidden(action_features))
        return self.actor_output(hidden).squeeze(-1)

    def state_value(self, action_features: torch.Tensor) -> torch.Tensor:
        pooled = action_features.mean(dim=0)
        return self.critic(pooled).squeeze(-1)

    @torch.no_grad()
    def select_action(
        self,
        action_features: torch.Tensor,
    ) -> tuple[int, float, float]:
        scores = self.action_scores(action_features)
        distribution = Categorical(logits=scores)
        action_index = distribution.sample()
        return (
            int(action_index.item()),
            float(distribution.log_prob(action_index).item()),
            float(self.state_value(action_features).item()),
        )

    @torch.no_grad()
    def top_k(
        self,
        action_features: torch.Tensor,
        count: int,
    ) -> list[int]:
        scores = self.action_scores(action_features)
        count = min(max(1, count), scores.numel())
        return torch.topk(scores, count).indices.cpu().tolist()


def effective_top_k(base_top_k: int, operation_count: int, dynamic: bool) -> int:
    if not dynamic:
        return base_top_k
    if operation_count >= 200:
        return max(base_top_k, 384)
    if operation_count >= 160:
        return max(base_top_k, 256)
    if operation_count >= 80:
        return max(base_top_k, 192)
    return base_top_k


@torch.no_grad()
def select_hybrid_indices(
    model: GreedyActorCritic,
    action_features: torch.Tensor,
    base_top_k: int,
    operation_count: int,
    rng: np.random.Generator,
    dynamic_top_k: bool = True,
    rule_fraction: float = 0.25,
    exploration_fraction: float = 0.15,
    temperature: float = 1.0,
) -> list[int]:
    candidate_count = int(action_features.shape[0])
    selection_limit = min(
        candidate_count,
        effective_top_k(base_top_k, operation_count, dynamic_top_k),
    )
    if candidate_count <= selection_limit:
        return list(range(candidate_count))
    scores = model.action_scores(action_features).cpu().numpy()
    features = action_features.detach().cpu().numpy()
    rule_scores = (
        10.0 * features[:, 3]
        + 8.0 * features[:, 7]
        + 0.50 * features[:, 10]
        + 0.25 * features[:, 8]
        + 0.10 * features[:, 5]
        - 1.00 * features[:, 13]
    )
    exploration_count = min(
        selection_limit,
        int(round(selection_limit * exploration_fraction)),
    )
    rule_count = min(
        selection_limit - exploration_count,
        int(round(selection_limit * rule_fraction)),
    )
    model_count = selection_limit - exploration_count - rule_count
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(index: int) -> bool:
        if index in selected_set:
            return False
        selected_set.add(index)
        selected.append(index)
        return True

    for index in np.argsort(-scores, kind="stable"):
        if len(selected) >= model_count:
            break
        add(int(index))
    for index in np.argsort(-rule_scores, kind="stable"):
        if len(selected) >= model_count + rule_count:
            break
        add(int(index))

    remaining = np.asarray(
        [index for index in range(candidate_count) if index not in selected_set],
        dtype=np.int64,
    )
    while len(selected) < selection_limit and remaining.size:
        remaining_scores = scores[remaining]
        logits = np.clip(
            (remaining_scores - np.max(remaining_scores)) / temperature,
            -50.0,
            0.0,
        )
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        chosen_position = int(rng.choice(remaining.size, p=probabilities))
        add(int(remaining[chosen_position]))
        remaining = np.delete(remaining, chosen_position)
    return selected


@dataclass
class Transition:
    features: np.ndarray
    action_index: int
    old_log_probability: float
    reward: float
    value: float
    done: bool


@dataclass
class RankingSample:
    features: np.ndarray
    gains: np.ndarray
    valid: np.ndarray


def supervised_ranking_update(
    model: GreedyActorCritic,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[RankingSample],
    device: torch.device,
    epochs: int = 2,
    target_temperature: float = 0.05,
    state_batch_size: int = 8,
    pairwise_weight: float = 0.25,
    pairwise_margin: float = 0.20,
) -> dict[str, float]:
    if not samples or epochs <= 0:
        return {"ranking_loss": 0.0}
    if target_temperature <= 0:
        raise ValueError("target_temperature must be positive")
    model.train()
    losses: list[float] = []
    order = np.arange(len(samples))
    for _ in range(epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), state_batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for sample_index in order[start:start + state_batch_size]:
                sample = samples[int(sample_index)]
                features = torch.as_tensor(
                    sample.features,
                    dtype=torch.float32,
                    device=device,
                )
                gains = torch.as_tensor(
                    sample.gains,
                    dtype=torch.float32,
                    device=device,
                )
                scores = model.action_scores(features)
                targets = torch.softmax(gains / target_temperature, dim=0)
                listwise_loss = -(targets * torch.log_softmax(scores, dim=0)).sum()

                best_index = int(torch.argmax(gains).item())
                negative_mask = gains < gains[best_index] - 1e-8
                if bool(torch.any(negative_mask).item()):
                    masked_scores = scores.clone()
                    masked_scores[~negative_mask] = -torch.inf
                    hard_index = int(torch.argmax(masked_scores).item())
                    pairwise_loss = torch.relu(
                        pairwise_margin
                        - scores[best_index]
                        + scores[hard_index]
                    )
                else:
                    pairwise_loss = torch.zeros((), device=device)
                batch_losses.append(
                    listwise_loss + pairwise_weight * pairwise_loss
                )
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"ranking_loss": float(np.mean(losses)) if losses else 0.0}


@torch.no_grad()
def ranking_metrics(
    model: GreedyActorCritic,
    samples: Sequence[RankingSample],
    device: torch.device,
    top_ks: Sequence[int] = (16, 32, 64, 96),
) -> dict[str, float]:
    if not samples:
        return {}
    model.eval()
    recall = {int(k): 0 for k in top_ks}
    improvement_recall = {int(k): 0 for k in top_ks}
    regret = {int(k): 0.0 for k in top_ks}
    valid_rate = {int(k): 0.0 for k in top_ks}
    best_ranks = []
    states_with_improvement = 0
    for sample in samples:
        features = torch.as_tensor(
            sample.features,
            dtype=torch.float32,
            device=device,
        )
        scores = model.action_scores(features).cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        gains = sample.gains
        best_gain = float(np.max(gains))
        best_indices = set(np.flatnonzero(np.isclose(gains, best_gain)).tolist())
        best_rank = next(
            rank for rank, index in enumerate(order, start=1)
            if int(index) in best_indices
        )
        best_ranks.append(best_rank)
        has_improvement = bool(np.any(gains > 0))
        if has_improvement:
            states_with_improvement += 1
        for raw_k in top_ks:
            k = int(raw_k)
            selected = order[:min(k, len(order))]
            recall[k] += int(any(int(index) in best_indices for index in selected))
            improvement_recall[k] += int(
                has_improvement and np.any(gains[selected] > 0)
            )
            regret[k] += max(0.0, best_gain - float(np.max(gains[selected])))
            valid_rate[k] += float(np.mean(sample.valid[selected]))

    result = {
        "median_best_rank": float(np.median(best_ranks)),
        "mean_best_rank": float(np.mean(best_ranks)),
        "states": float(len(samples)),
        "states_with_improvement": float(states_with_improvement),
    }
    for raw_k in top_ks:
        k = int(raw_k)
        result[f"recall@{k}"] = recall[k] / len(samples)
        result[f"improvement_recall@{k}"] = (
            improvement_recall[k] / states_with_improvement
            if states_with_improvement else 1.0
        )
        result[f"regret@{k}"] = regret[k] / len(samples)
        result[f"valid_rate@{k}"] = valid_rate[k] / len(samples)
    return result


def _gae(
    transitions: Sequence[Transition],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = np.zeros(len(transitions), dtype=np.float32)
    returns = np.zeros(len(transitions), dtype=np.float32)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        continuation = 0.0 if transition.done else 1.0
        delta = (
            transition.reward
            + gamma * next_value * continuation
            - transition.value
        )
        next_advantage = (
            delta
            + gamma * gae_lambda * continuation * next_advantage
        )
        advantages[index] = next_advantage
        returns[index] = next_advantage + transition.value
        next_value = transition.value
    advantages_tensor = torch.from_numpy(advantages)
    if advantages_tensor.numel() > 1:
        advantages_tensor = (
            advantages_tensor - advantages_tensor.mean()
        ) / (advantages_tensor.std(unbiased=False) + 1e-8)
    return advantages_tensor, torch.from_numpy(returns)


def ppo_update(
    model: GreedyActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    device: torch.device,
    epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    if not transitions:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    advantages, returns = _gae(transitions, gamma, gae_lambda)
    advantages = advantages.to(device)
    returns = returns.to(device)
    old_log_probabilities = torch.tensor(
        [transition.old_log_probability for transition in transitions],
        dtype=torch.float32,
        device=device,
    )

    totals = np.zeros(3, dtype=np.float64)
    for _ in range(epochs):
        policy_losses = []
        value_losses = []
        entropies = []
        for index, transition in enumerate(transitions):
            features = torch.as_tensor(
                transition.features,
                dtype=torch.float32,
                device=device,
            )
            distribution = Categorical(logits=model.action_scores(features))
            action = torch.tensor(
                transition.action_index,
                dtype=torch.long,
                device=device,
            )
            new_log_probability = distribution.log_prob(action)
            ratio = torch.exp(
                new_log_probability - old_log_probabilities[index]
            )
            unclipped = ratio * advantages[index]
            clipped = torch.clamp(
                ratio,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            ) * advantages[index]
            policy_losses.append(-torch.minimum(unclipped, clipped))
            value_losses.append(
                (model.state_value(features) - returns[index]).square()
            )
            entropies.append(distribution.entropy())

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = (
            policy_loss
            + value_coefficient * value_loss
            - entropy_coefficient * entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        totals += np.array([
            float(loss.detach().cpu()),
            float(policy_loss.detach().cpu()),
            float(value_loss.detach().cpu()),
        ])
    totals /= max(1, epochs)
    return {
        "loss": float(totals[0]),
        "policy_loss": float(totals[1]),
        "value_loss": float(totals[2]),
    }


def save_checkpoint(
    model: GreedyActorCritic,
    path: str | Path,
    metadata: dict[str, object] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "hidden_size": model.hidden_size,
        "feature_count": FEATURE_COUNT,
        "metadata": metadata or {},
    }, output_path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> GreedyActorCritic:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint["feature_count"]) != FEATURE_COUNT:
        raise ValueError("checkpoint feature count does not match HLS")
    model = GreedyActorCritic(int(checkpoint["hidden_size"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


def export_actor(model: GreedyActorCritic, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_weights = model.actor_hidden.weight.detach().cpu().numpy().reshape(-1)
    hidden_bias = model.actor_hidden.bias.detach().cpu().numpy().reshape(-1)
    output_weights = model.actor_output.weight.detach().cpu().numpy().reshape(-1)
    output_bias = model.actor_output.bias.detach().cpu().numpy().reshape(-1)
    with output_path.open("w", encoding="ascii") as output:
        output.write(f"{MODEL_MAGIC}\n")
        output.write(f"{FEATURE_COUNT} {model.hidden_size}\n")
        for values in (input_weights, hidden_bias, output_weights, output_bias):
            output.write(" ".join(f"{float(value):.17g}" for value in values))
            output.write("\n")


def load_exported_actor(
    path: str | Path,
    device: torch.device,
) -> GreedyActorCritic:
    tokens = Path(path).read_text(encoding="ascii").split()
    if len(tokens) < 3 or tokens[0] != MODEL_MAGIC:
        raise ValueError("invalid exported Greedy RL model")
    input_size = int(tokens[1])
    hidden_size = int(tokens[2])
    if input_size != FEATURE_COUNT:
        raise ValueError("exported model feature count does not match HLS")
    cursor = 3

    def take(count: int) -> torch.Tensor:
        nonlocal cursor
        values = [float(value) for value in tokens[cursor:cursor + count]]
        if len(values) != count:
            raise ValueError("exported model is incomplete")
        cursor += count
        return torch.tensor(values, dtype=torch.float32)

    model = GreedyActorCritic(hidden_size)
    with torch.no_grad():
        model.actor_hidden.weight.copy_(
            take(hidden_size * FEATURE_COUNT).view(hidden_size, FEATURE_COUNT)
        )
        model.actor_hidden.bias.copy_(take(hidden_size))
        model.actor_output.weight.copy_(take(hidden_size).view(1, hidden_size))
        model.actor_output.bias.copy_(take(1))
    return model.to(device)
