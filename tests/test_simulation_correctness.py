from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from Algorithms.bandit import RandomPolicy
from Algorithms.cmdp import RandomCMDPPolicy
from Algorithms.contextual import RandomContextualPolicy
from Environments.base_environment import BanditEnvironment
from Environments.cmdp_environment import StatefulMicrofinanceCMDPEnv
from evaluate import run_cmdp_episode, run_episode
from models.arm_spec import ArmSpec
from models.environment_config import (
    CMDPEnvironmentConfig,
    ContextualEnvironmentConfig,
    EnvironmentConfig,
)
from simulation import SyntheticBorrowerGenerator


def set_pool_schedule(monkeypatch, rounds):
    def generate_pool(generator, pool_size):
        costs_and_probabilities = rounds[generator.applicant_counter // pool_size]
        assert len(costs_and_probabilities) == pool_size
        pool = []
        for cost, probability in costs_and_probabilities:
            candidate = generator.generate_candidate()
            application = replace(
                candidate.borrower.application_context, requested_loan_amount=cost
            )
            borrower = type(candidate.borrower)(application)
            pool.append(replace(
                candidate,
                borrower=borrower,
                context=generator.encoder.transform(borrower),
                true_repayment_probability=probability,
            ))
        return pool

    monkeypatch.setattr(SyntheticBorrowerGenerator, "generate_pool", generate_pool)


def test_arm_opportunities_and_outcomes_are_paired_across_policies():
    class ScriptedPolicy(RandomPolicy):
        def __init__(self, choices, extra_draws):
            self.choices = choices
            self.extra_draws = extra_draws
            super().__init__(2)

        def reset(self):
            super().reset()
            self.costs_seen = []

        def select_action(self, env):
            assert not hasattr(env, "true_probabilities")
            assert not hasattr(env, "arms")
            self.rng.random(self.extra_draws)
            self.costs_seen.append(env.current_costs().copy())
            return self.choices[env.round_index]

    env = BanditEnvironment.for_arms(
        [ArmSpec("A", "A", 0.5, 10, 2), ArmSpec("B", "B", 0.5, 6, 1)],
        EnvironmentConfig(100, 3),
    )
    first = ScriptedPolicy([0, 1, 0], 0)
    second = ScriptedPolicy([1, 1, 1], 100)
    first_run = run_episode(env, first)
    second_run = run_episode(env, second)
    np.testing.assert_array_equal(first.costs_seen, second.costs_seen)
    assert list(first_run.trajectory["reward"]) == [0, 1, 0]
    assert list(second_run.trajectory["reward"]) == [1, 1, 0]
    assert first.total_updates == second.total_updates == 3

    replay = run_episode(env, first)
    assert replay.summary == first_run.summary
    pd.testing.assert_frame_equal(replay.trajectory, first_run.trajectory)


def test_stochastic_arm_cost_can_be_affordable_after_an_empty_round(monkeypatch):
    monkeypatch.setattr(
        BanditEnvironment, "_sample_arm_costs",
        lambda env: np.array([[6.0, 4.0, 4.0][env.round_index]]),
    )
    env = BanditEnvironment.for_arms(
        [ArmSpec("A", "A", 1.0, 6, 2)], EnvironmentConfig(4, 3, minimum_cost=1)
    )
    assert not env.done
    with pytest.raises(ValueError, match="affordable"):
        env.step(0)
    assert env.remaining_budget == 4
    assert env.total_allocations == 0
    policy = RandomPolicy(1)
    report = run_episode(env, policy)
    assert list(report.trajectory["cost"]) == [0, 4]
    assert list(report.trajectory["reward"]) == [0, 1]
    assert list(report.trajectory["budget_remaining"]) == [4, 0]
    assert report.summary["decision_rounds"] == 2
    assert report.summary["rounds"] == policy.total_updates == 1


def pool_signature(env):
    return [
        (candidate.borrower.application_context, candidate.fairness_group,
         candidate.true_repayment_probability, candidate.context.tolist())
        for candidate in env.current_pool
    ]


def test_contextual_pool_is_unchanged_after_different_funding_counts():
    config = ContextualEnvironmentConfig(150_000, 3, seed=3, pool_size=4)
    first = BanditEnvironment.for_contextual(config)
    second = BanditEnvironment.for_contextual(config)
    initial_pool = pool_signature(first)
    for round_index in range(config.max_rounds):
        assert pool_signature(first) == pool_signature(second)
        expected_id = f"APP-{round_index * config.pool_size + 1:07d}"
        assert first.current_pool[0].borrower.id == expected_id
        funded = first.step(0)
        if round_index == 0:
            second.advance_round()
        else:
            paired = second.step(0)
            assert (funded.identifier, funded.reward) == (paired.identifier, paired.reward)
    assert first.total_allocations == second.total_allocations + 1
    first.reset()
    assert pool_signature(first) == initial_pool


@pytest.mark.parametrize(
    "delay, feedback_counts, updates_at_decision, observed, pending",
    [
        (0, [1, 0, 1, 0], [0, 1], 2, 0),
        (1, [0, 1, 0, 1], [0, 1], 2, 0),
        (2, [0, 0, 1, 0], [0, 1], 2, 0),
        (3, [0, 0, 0, 1], [0, 0], 1, 1),
    ],
)
def test_contextual_skips_and_feedback_follow_the_decision_clock(
    monkeypatch, delay, feedback_counts, updates_at_decision, observed, pending
):
    set_pool_schedule(monkeypatch, [[(cost, 1.0)] for cost in (4, 12, 4, 12)])
    env = BanditEnvironment.for_contextual(
        ContextualEnvironmentConfig(8, 4, pool_size=1, feedback_delay_rounds=delay)
    )

    class RecordingPolicy(RandomContextualPolicy):
        def reset(self):
            super().reset()
            self.decisions = []
            self.feedback_ids = []

        def select_action(self, observation):
            assert not hasattr(observation, "true_probabilities")
            assert not hasattr(observation, "generator")
            assert not hasattr(observation, "pending_feedback")
            self.decisions.append((observation.round_index, self.total_updates))
            return super().select_action(observation)

        def update(self, feedback):
            self.feedback_ids.append(feedback.applicant_id)
            super().update(feedback)

    policy = RecordingPolicy(env.n_features)
    report = run_episode(env, policy)
    rows = report.trajectory
    assert policy.decisions == list(zip([0, 2], updates_at_decision))
    assert list(rows["round"]) == [0, 1, 2, 3]
    assert list(rows["cost"]) == [4, 0, 4, 0]
    assert list(rows["budget_remaining"]) == [4, 4, 0, 0]
    assert list(rows["cumulative_cost"]) == [4, 4, 8, 8]
    assert list(rows["reward"]) == [1, 0, 1, 0]
    assert list(rows["repayment_rate"]) == [1, 1, 1, 1]
    assert list(rows["observed_feedback_count"]) == feedback_counts
    assert policy.feedback_ids == ["APP-0000001", "APP-0000003"][:sum(feedback_counts)]
    assert report.summary["observed_reward_by_horizon"] == observed
    assert report.summary["pending_feedback"] == pending
    assert report.summary["cumulative_reward"] == report.summary["rounds"] == 2
    assert report.summary["reward_per_cost"] == 0.25
    assert sum(env.group_population_opportunity_counts.values()) == 4
    assert sum(env.group_opportunity_counts.values()) == 2
    assert env.advance_round() == ()


def test_cmdp_outcomes_stay_with_applicants_after_reordering_and_skips(monkeypatch):
    set_pool_schedule(monkeypatch, [[(10, 0.5), (10, 0.5)]] * 3)
    config = CMDPEnvironmentConfig(
        initial_capital=100, max_rounds=3, pool_size=2, max_loans_per_round=2
    )
    first, second = StatefulMicrofinanceCMDPEnv(config), StatefulMicrofinanceCMDPEnv(config)
    for round_index in range(config.max_rounds):
        assert pool_signature(first) == pool_signature(second)
        first_outcomes = {}
        for _ in range(2):
            loan = first.fund(0, first.current_pool[0].context)
            first_outcomes[loan.applicant_id] = loan.hidden_reward
        second_outcomes = {}
        for _ in range(1 if round_index == 0 else 2):
            loan = second.fund(len(second.current_pool) - 1, second.current_pool[-1].context)
            second_outcomes[loan.applicant_id] = loan.hidden_reward
        assert all(first_outcomes[key] == reward for key, reward in second_outcomes.items())
        if round_index == 1:
            assert list(first_outcomes.values()) == [0, 1]
            assert list(second_outcomes.values()) == [1, 0]
        first.advance_round()
        second.advance_round()
    assert first.total_allocations == second.total_allocations + 1


class RecordingCMDPPolicy(RandomCMDPPolicy):
    def reset(self):
        super().reset()
        self.decisions = []

    def select_action(self, observation):
        assert not hasattr(observation, "true_probabilities")
        assert not hasattr(observation, "eventual_pending_capital_return")
        assert not hasattr(observation, "funding_history")
        assert isinstance(observation.pending_loans, int)
        assert all(not hasattr(c, "true_repayment_probability") for c in observation.current_pool)
        self.decisions.append((observation.round_index, self.total_updates))
        return observation.feasible_actions()[0]


@pytest.mark.parametrize("probability, recycle, recovery, funded, cash, loss, unrecycled", [
    (1.0, 1.0, 0.0, 3, 2, 0, 0),
    (0.0, 1.0, 0.0, 1, 2, 10, 0),
    (1.0, 0.5, 0.0, 1, 7, 0, 5),
    (0.0, 1.0, 0.25, 1, 4.5, 7.5, 0),
])
def test_cmdp_capital_conservation_with_reserve_and_delayed_returns(
    monkeypatch, probability, recycle, recovery, funded, cash, loss, unrecycled
):
    set_pool_schedule(monkeypatch, [[(10, probability)]] * 5)
    env = StatefulMicrofinanceCMDPEnv(CMDPEnvironmentConfig(
        initial_capital=12, minimum_liquidity_reserve=2, max_rounds=5,
        pool_size=1, repayment_delay_rounds=2, recycle_fraction=recycle,
        default_recovery_fraction=recovery,
    ))
    policy = RecordingCMDPPolicy(env.encoder.n_features)
    report = run_cmdp_episode(env, policy, {}, 1.0, seed=0)
    rows, summary = report.trajectory, report.summary
    assert rows["round"].tolist() == [0, 1, 2, 3, 4]
    assert rows["funded_this_round"].tolist() == ([1, 0, 1, 0, 1] if funded == 3 else [1, 0, 0, 0, 0])
    assert policy.decisions == ([(0, 0), (2, 1), (4, 2)] if funded == 3 else [(0, 0)])
    assert (rows["available_capital"] >= 2).all()
    np.testing.assert_allclose(
        rows["available_capital"] + rows["capital_at_risk"]
        + rows["cumulative_default_loss"] + rows["cumulative_unrecycled_successful_principal"],
        12,
    )
    np.testing.assert_allclose(
        rows["available_capital"], 12 - rows["cumulative_disbursed"] + rows["cumulative_recycled"]
    )
    assert summary["final_available_capital"] == cash
    assert summary["default_loss_by_horizon"] == loss
    assert summary["unrecycled_successful_principal_by_horizon"] == unrecycled
    assert cash + summary["final_capital_at_risk"] + loss + unrecycled == 12
    assert cash == 12 - summary["cumulative_disbursed"] + summary["cumulative_recycled"]
    assert summary["eventual_terminal_capital"] == 12 - loss - unrecycled
    assert summary["rounds"] == sum(env.group_population_opportunity_counts.values()) == 5
    assert summary["loans_funded"] == funded
    assert summary["cumulative_reward"] == report.funding_log["hidden_reward"].sum() == probability * funded
    assert summary["observed_reward_by_horizon"] == report.maturity_log["reward"].sum() == probability * min(funded, 2)
    assert report.funding_log["applicant_id"].is_unique
    assert report.maturity_log["applicant_id"].is_unique
    assert env.advance_round().matured_feedback == ()


def test_cmdp_zero_delay_recycles_and_updates_before_the_next_slot(monkeypatch):
    set_pool_schedule(monkeypatch, [[(10, 1.0), (10, 0.0)]])
    env = StatefulMicrofinanceCMDPEnv(CMDPEnvironmentConfig(
        initial_capital=12, minimum_liquidity_reserve=2, max_rounds=1,
        pool_size=2, max_loans_per_round=2, repayment_delay_rounds=0,
        default_recovery_fraction=0.25,
    ))
    policy = RecordingCMDPPolicy(env.encoder.n_features)
    report = run_cmdp_episode(env, policy, {}, 1.0, seed=0)
    assert policy.decisions == [(0, 0), (0, 1)]
    assert policy.total_updates == 2
    assert report.trajectory["matured_this_round"].tolist() == [2]
    assert report.maturity_log["capital_returned"].tolist() == [10, 2.5]
    assert report.summary["final_available_capital"] == 4.5
    assert report.summary["default_loss_by_horizon"] == 7.5
    assert report.summary["observed_reward_by_horizon"] == report.summary["cumulative_reward"] == 1
    assert report.summary["repayment_rate"] == 0.5
    assert report.summary["reward_per_cost"] == 0.05
    assert report.summary["pending_loans"] == 0
