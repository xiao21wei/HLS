from __future__ import annotations

import numpy as np
import torch

from rl_greedy.fjsp_env import (
    FEATURE_COUNT,
    FjspInstance,
    apply_action,
    build_initial_schedule,
    generate_actions,
)
from rl_greedy.policy import (
    GreedyActorCritic,
    export_actor,
    load_exported_actor,
    select_hybrid_indices,
)


def test_brandimarte_parse_and_action_generation() -> None:
    instance = FjspInstance.from_file(
        "datasets/Brandimarte_Data/Text/Mk01.fjs"
    )
    assert len(instance.jobs) == 10
    assert instance.machine_count == 6
    assert instance.operation_count == 55

    schedule = build_initial_schedule(instance, np.random.default_rng(7))
    assert schedule.valid
    assert schedule.makespan > 0
    actions = generate_actions(instance, schedule)
    assert actions
    assert all(len(action.features) == FEATURE_COUNT for action in actions)
    assert np.isfinite(np.asarray([action.features for action in actions])).all()
    assert any(apply_action(instance, schedule, action).valid for action in actions)


def test_exported_actor_matches_python_model(tmp_path) -> None:
    torch.manual_seed(11)
    model = GreedyActorCritic(hidden_size=16).eval()
    features = torch.randn(13, FEATURE_COUNT)
    path = tmp_path / "policy.txt"
    export_actor(model, path)
    restored = load_exported_actor(path, torch.device("cpu")).eval()
    torch.testing.assert_close(
        model.action_scores(features),
        restored.action_scores(features),
        rtol=1e-6,
        atol=1e-6,
    )


def test_brandimarte_scale_synthetic_and_dynamic_hybrid_selection() -> None:
    rng = np.random.default_rng(19)
    instance = FjspInstance.synthetic(
        rng,
        job_count=20,
        machine_count=15,
        min_operations=10,
        max_operations=15,
        max_machine_options=6,
        min_processing_time=1,
        max_processing_time=20,
    )
    assert 200 <= instance.operation_count <= 300
    assert all(
        1 <= len(operation.options) <= 6
        for job in instance.jobs
        for operation in job
    )

    torch.manual_seed(23)
    model = GreedyActorCritic(hidden_size=16).eval()
    features = torch.randn(500, FEATURE_COUNT)
    selected = select_hybrid_indices(
        model,
        features,
        base_top_k=96,
        operation_count=240,
        rng=np.random.default_rng(29),
        dynamic_top_k=True,
        rule_fraction=0.25,
        exploration_fraction=0.15,
        temperature=1.0,
    )
    assert len(selected) == 384
    assert len(set(selected)) == len(selected)
