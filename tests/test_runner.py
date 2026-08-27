from __future__ import annotations

import json

import pandas as pd
import pytest

from Data.arm_data import default_arm_configuration, load_arm_configuration
from Environments.synthetic_environment_bandits import ArmBanditEnvironment
from evaluate import compare_policies, save_results
from main import (
    build_arm_policy_factories,
    combine_aggregate_results,
    parse_args,
)
from models.environment_config import EnvironmentConfig


def test_default_arm_standard_deviations_are_ten_percent() -> None:
    arms, _ = default_arm_configuration()
    assert all(arm.cost_std == pytest.approx(arm.cost_mean * 0.10) for arm in arms)


def test_json_arm_configuration_and_validation(tmp_path) -> None:
    path = tmp_path / "arms.json"
    path.write_text(
        json.dumps(
            {
                "arms": [
                    {
                        "name": "A",
                        "group": "A",
                        "repayment_probability": 0.8,
                        "cost_mean": 10,
                        "cost_std": 1,
                    }
                ],
                "target_shares": {"A": 0.5},
            }
        ),
        encoding="utf-8",
    )
    arms, targets = load_arm_configuration(path)
    assert arms[0].cost_std == 1
    assert targets == {"A": 0.5}

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_shares"] = {"missing": 0.5}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown groups"):
        load_arm_configuration(path)


def test_shared_runner_saves_mode_neutral_results(tmp_path) -> None:
    arms, targets = default_arm_configuration()

    def factory(seed: int) -> ArmBanditEnvironment:
        return ArmBanditEnvironment(
            arms, EnvironmentConfig(total_budget=80, max_rounds=2, seed=seed)
        )

    policies = {"random": build_arm_policy_factories(targets)["random"]}
    summaries, trajectories = compare_policies(factory, policies, targets, [0])
    aggregate = save_results(summaries, trajectories, tmp_path, "arm")
    assert aggregate.loc["random", ("cumulative_reward", "std")] == 0.0
    for filename in (
        "episode_summaries.csv",
        "trajectories.csv",
        "aggregate_results.csv",
        "cumulative_reward.png",
    ):
        assert (tmp_path / filename).exists()
    assert isinstance(pd.read_csv(tmp_path / "trajectories.csv"), pd.DataFrame)


def test_runner_accepts_four_numeric_and_named_modes() -> None:
    assert parse_args(["--mode", "1"]).mode == "1"
    assert parse_args(["--mode", "contextual"]).mode == "contextual"
    assert parse_args(["--mode", "3"]).mode == "3"
    assert parse_args(["--mode", "all"]).mode == "all"


def test_combined_results_preserve_mode_and_rank() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("cumulative_reward", "mean"), ("cumulative_reward", "std")]
    )
    arm = pd.DataFrame([[2.0, 0.0]], index=["arm_policy"], columns=columns)
    cmdp = pd.DataFrame([[3.0, 0.0]], index=["cmdp_policy"], columns=columns)
    arm.index.name = "policy"
    cmdp.index.name = "policy"
    combined = combine_aggregate_results({"arm": arm, "cmdp": cmdp})
    assert set(combined["mode"]) == {"arm", "cmdp"}
    assert set(combined["policy"]) == {"arm_policy", "cmdp_policy"}
    assert (combined["rank_within_mode"] == 1.0).all()
