from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np

from Data.feature_encoder import BorrowerFeatureEncoder
from metrics import (
    allocation_shares,
    demographic_parity_gap,
    group_funding_rates,
    quota_deficits,
)
from models.environment_config import CMDPEnvironmentConfig
from models.events import (
    FundingResult,
    MaturedLoanFeedback,
    PendingLoan,
    RoundAdvanceResult,
)
from simulation import SyntheticBorrowerGenerator, environment_rngs


class StatefulMicrofinanceCMDPEnv:
    """Phase 3 contextual CMDP with persistent and recyclable capital.

    Each round supplies a borrower pool and allows a bounded batch of loans.
    Principal moves from available cash to capital-at-risk immediately; outcome
    feedback and recoverable principal arrive only when each loan matures.
    """

    def __init__(
        self,
        config: CMDPEnvironmentConfig,
        encoder: BorrowerFeatureEncoder | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder or BorrowerFeatureEncoder(
            include_protected_attributes=False
        )
        self.reset()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def total_allocations(self) -> int:
        return len(self.funding_history)

    @property
    def capital_at_risk(self) -> float:
        return float(sum(loan.principal for loan in self.pending_loans))

    @property
    def observed_outcomes(self) -> int:
        return self.observed_repayments + self.observed_defaults

    @property
    def observed_repayment_rate(self) -> float:
        if self.observed_outcomes == 0:
            return 0.5
        return self.observed_repayments / self.observed_outcomes

    @property
    def recent_repayment_rate(self) -> float:
        if not self.recent_outcomes:
            return 0.5
        return float(np.mean(self.recent_outcomes))

    @property
    def matured_at_round_start(self) -> tuple[MaturedLoanFeedback, ...]:
        """Feedback released while advancing into the current round."""

        return self._matured_at_round_start

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        actual_seed = self.config.seed if seed is None else seed
        self.rng, self._outcome_rng = environment_rngs(actual_seed)
        self.generator = SyntheticBorrowerGenerator(
            config=self.config.borrower_generation_config(actual_seed),
            encoder=self.encoder,
            rng=self.rng,
        )
        self.round_index = 0
        self.available_capital = float(self.config.initial_capital)
        self.pending_loans: list[PendingLoan] = []
        self.funding_history: list[dict[str, Any]] = []
        self.maturity_history: list[dict[str, Any]] = []
        self.group_funding_counts: dict[str, int] = defaultdict(int)
        self.group_opportunity_counts: dict[str, int] = defaultdict(int)
        self.group_population_opportunity_counts: dict[str, int] = defaultdict(int)
        self.recent_outcomes: deque[float] = deque(maxlen=30)
        self.observed_repayments = 0
        self.observed_defaults = 0
        self.cumulative_disbursed = 0.0
        self.cumulative_recycled = 0.0
        self.cumulative_default_loss = 0.0
        self.cumulative_unrecycled_successful_principal = 0.0
        self._prepare_round()
        self._matured_at_round_start: tuple[MaturedLoanFeedback, ...] = tuple()
        self._done = False
        return self.observation()

    def _prepare_round(self) -> None:
        self.current_pool = self.generator.generate_pool(self.config.pool_size)
        self._outcome_draws = {
            candidate.borrower.id: draw
            for candidate, draw in zip(
                self.current_pool, self._outcome_rng.random(len(self.current_pool))
            )
        }
        self._record_pool_opportunities()

    def _record_pool_opportunities(self) -> None:
        """Count each arrival once and measure affordability before the batch."""
        groups = self.current_groups()
        for group in groups:
            self.group_population_opportunity_counts[group] += 1
        for index in self.feasible_actions():
            group = groups[index]
            self.group_funding_counts.setdefault(group, 0)
            self.group_opportunity_counts[group] += 1

    def base_context_matrix(self) -> np.ndarray:
        if not self.current_pool:
            return np.empty((0, self.encoder.n_features), dtype=float)
        return np.vstack([candidate.context for candidate in self.current_pool])

    def current_costs(self) -> np.ndarray:
        return np.asarray(
            [candidate.cost for candidate in self.current_pool], dtype=float
        )

    def current_groups(self) -> tuple[str, ...]:
        return tuple(candidate.fairness_group for candidate in self.current_pool)

    def true_probabilities(self) -> np.ndarray:
        """Evaluator-only probabilities used by the informed portfolio heuristic."""

        return np.asarray(
            [
                candidate.true_repayment_probability
                for candidate in self.current_pool
            ],
            dtype=float,
        )

    def feasible_actions(self) -> list[int]:
        spendable = self.available_capital - self.config.minimum_liquidity_reserve
        return [
            index
            for index, candidate in enumerate(self.current_pool)
            if 0 < candidate.cost <= spendable
        ]

    def due_soon_capital(self, lookahead_rounds: int = 2) -> float:
        horizon = self.round_index + max(0, lookahead_rounds)
        return float(
            sum(
                loan.principal
                for loan in self.pending_loans
                if loan.due_round <= horizon
            )
        )

    def next_due_in(self) -> int | None:
        if not self.pending_loans:
            return None
        return max(
            0,
            min(loan.due_round for loan in self.pending_loans) - self.round_index,
        )

    def fairness_snapshot(
        self, target_shares: Mapping[str, float]
    ) -> dict[str, Any]:
        shares = allocation_shares(self.group_funding_counts)
        rates = group_funding_rates(
            self.group_funding_counts,
            self.group_opportunity_counts,
        )
        deficits = quota_deficits(self.group_funding_counts, target_shares)
        return {
            "allocation_shares": shares,
            "funding_rates": rates,
            "quota_deficits": deficits,
            "total_quota_deficit": float(sum(deficits.values())),
            "demographic_parity_gap": demographic_parity_gap(
                self.group_funding_counts,
                self.group_opportunity_counts,
            ),
        }

    def projected_fairness(
        self,
        action: int,
        target_shares: Mapping[str, float],
        parity_tolerance: float,
    ) -> dict[str, float]:
        funded = dict(self.group_funding_counts)
        group = self.current_pool[action].fairness_group
        funded[group] = funded.get(group, 0) + 1
        deficits = quota_deficits(funded, target_shares)
        gap = demographic_parity_gap(funded, self.group_opportunity_counts)
        return {
            "quota_violation": float(sum(deficits.values())),
            "parity_gap": gap,
            "parity_violation": gap - parity_tolerance if gap > parity_tolerance else 0.0,
        }

    def state_bucket(
        self,
        target_shares: Mapping[str, float] | None = None,
        parity_tolerance: float = 0.05,
    ) -> str:
        targets = dict(target_shares or {})
        liquidity = self.available_capital / self.config.initial_capital
        liquidity_bucket = 0 if liquidity < 0.20 else (1 if liquidity < 0.55 else 2)

        gap = demographic_parity_gap(
            self.group_funding_counts,
            self.group_opportunity_counts,
        )
        fairness_bucket = "NA" if np.isnan(gap) else (
            0
            if gap <= parity_tolerance
            else (1 if gap <= 2 * parity_tolerance else 2)
        )

        repayment = self.recent_repayment_rate
        repayment_bucket = 0 if repayment < 0.55 else (1 if repayment < 0.75 else 2)

        quota = sum(quota_deficits(self.group_funding_counts, targets).values())
        quota_bucket = 0 if quota <= 0.01 else (1 if quota <= 0.10 else 2)
        return (
            f"B{liquidity_bucket}-F{fairness_bucket}-"
            f"R{repayment_bucket}-Q{quota_bucket}"
        )

    def observation(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "available_capital": self.available_capital,
            "capital_at_risk": self.capital_at_risk,
            "pending_loans": len(self.pending_loans),
            "recent_repayment_rate": self.recent_repayment_rate,
            "observed_repayment_rate": self.observed_repayment_rate,
            "cumulative_disbursed": self.cumulative_disbursed,
            "cumulative_recycled": self.cumulative_recycled,
            "cumulative_default_loss": self.cumulative_default_loss,
            "contexts": self.base_context_matrix().copy(),
            "costs": self.current_costs().copy(),
            "groups": self.current_groups(),
            "feasible_actions": self.feasible_actions(),
            "group_funding_counts": dict(self.group_funding_counts),
            "group_opportunity_counts": dict(self.group_opportunity_counts),
            "group_population_opportunity_counts": dict(
                self.group_population_opportunity_counts
            ),
        }

    def policy_observation(self, *, informed: bool = False) -> SimpleNamespace:
        """Snapshot the current candidates and observable portfolio state."""
        observed = self.observation()
        feasible = observed.pop("feasible_actions")
        current_pool = tuple(
            SimpleNamespace(context=context, cost=cost, fairness_group=group)
            for context, cost, group in zip(
                observed["contexts"], observed["costs"], observed["groups"]
            )
        )
        fairness_state = SimpleNamespace(
            current_pool=current_pool,
            group_funding_counts=observed["group_funding_counts"],
            group_opportunity_counts=observed["group_opportunity_counts"],
        )
        repayment_schedule = SimpleNamespace(
            round_index=self.round_index,
            pending_loans=tuple(
                SimpleNamespace(due_round=loan.due_round, principal=loan.principal)
                for loan in self.pending_loans
            ),
        )
        view = SimpleNamespace(
            **observed,
            config=self.config,
            total_allocations=self.total_allocations,
            current_pool=current_pool,
            base_context_matrix=lambda: observed["contexts"],
            current_costs=lambda: observed["costs"],
            current_groups=lambda: observed["groups"],
            feasible_actions=lambda: feasible,
            due_soon_capital=partial(
                StatefulMicrofinanceCMDPEnv.due_soon_capital, repayment_schedule
            ),
            projected_fairness=partial(
                StatefulMicrofinanceCMDPEnv.projected_fairness, fairness_state
            ),
        )
        if informed:
            probabilities = self.true_probabilities()
            view.true_probabilities = lambda: probabilities
        return view

    def fund(self, action: int, learning_context: np.ndarray) -> FundingResult:
        if self.done:
            raise RuntimeError("Cannot fund in a completed environment")
        if action < 0 or action >= len(self.current_pool):
            raise IndexError(f"Invalid action {action}")
        if action not in self.feasible_actions():
            raise ValueError("Selected borrower is not currently affordable")

        candidate = self.current_pool[action]
        principal = float(candidate.cost)
        reward = float(
            self._outcome_draws[candidate.borrower.id]
            < candidate.true_repayment_probability
        )
        due_round = self.round_index + self.config.repayment_delay_rounds

        loan = PendingLoan(
            applicant_id=candidate.borrower.id,
            fairness_group=candidate.fairness_group,
            principal=principal,
            funded_round=self.round_index,
            due_round=due_round,
            reward=reward,
            true_repayment_probability=candidate.true_repayment_probability,
            learning_context=np.asarray(learning_context, dtype=float).copy(),
        )
        self.current_pool.pop(action)
        self.available_capital -= principal
        self.cumulative_disbursed += principal
        self.group_funding_counts[candidate.fairness_group] += 1
        self.pending_loans.append(loan)
        self.funding_history.append(
            {
                "round": self.round_index,
                "applicant_id": candidate.borrower.id,
                "fairness_group": candidate.fairness_group,
                "principal": principal,
                "hidden_reward": reward,
                "true_repayment_probability": candidate.true_repayment_probability,
                "due_round": due_round,
                "available_capital_after": self.available_capital,
                "poverty_relief_score": candidate.borrower.poverty_relief_score,
                "vulnerability_score": candidate.borrower.vulnerability_score,
            }
        )
        observed = self._release_due_loans() if due_round == self.round_index else ()
        return FundingResult(
            round_index=self.round_index,
            applicant_id=candidate.borrower.id,
            fairness_group=candidate.fairness_group,
            principal=principal,
            hidden_reward=reward,
            true_repayment_probability=candidate.true_repayment_probability,
            due_round=due_round,
            available_capital=self.available_capital,
            capital_at_risk=self.capital_at_risk,
            observed_feedback=observed,
        )

    def _release_due_loans(self) -> tuple[MaturedLoanFeedback, ...]:
        matured = [
            loan
            for loan in self.pending_loans
            if loan.due_round <= self.round_index
        ]
        if not matured:
            return tuple()
        self.pending_loans = [
            loan
            for loan in self.pending_loans
            if loan.due_round > self.round_index
        ]
        feedback: list[MaturedLoanFeedback] = []
        for loan in matured:
            if loan.repaid:
                returned = loan.principal * self.config.recycle_fraction
                self.observed_repayments += 1
                self.cumulative_unrecycled_successful_principal += (
                    loan.principal - returned
                )
            else:
                returned = (
                    loan.principal * self.config.default_recovery_fraction
                )
                self.observed_defaults += 1
                self.cumulative_default_loss += loan.principal - returned
            self.available_capital += returned
            self.cumulative_recycled += returned
            self.recent_outcomes.append(loan.reward)
            self.maturity_history.append(
                {
                    "round": self.round_index,
                    "applicant_id": loan.applicant_id,
                    "fairness_group": loan.fairness_group,
                    "reward": loan.reward,
                    "principal": loan.principal,
                    "capital_returned": returned,
                    "available_capital_after": self.available_capital,
                }
            )
            feedback.append(
                MaturedLoanFeedback(
                    applicant_id=loan.applicant_id,
                    fairness_group=loan.fairness_group,
                    principal=loan.principal,
                    reward=loan.reward,
                    learning_context=loan.learning_context.copy(),
                    capital_returned=returned,
                    due_round=loan.due_round,
                )
            )
        return tuple(feedback)

    def advance_round(self) -> RoundAdvanceResult:
        """Move one lending cycle forward and release repayments due there."""

        if self.done:
            return RoundAdvanceResult(
                new_round_index=self.round_index,
                matured_feedback=tuple(),
                available_capital=self.available_capital,
                capital_at_risk=self.capital_at_risk,
                done=True,
            )

        self.round_index += 1
        matured = self._release_due_loans()
        if self.round_index < self.config.max_rounds:
            self._prepare_round()
        else:
            self.current_pool = []
        self._matured_at_round_start = matured
        self._done = self.round_index >= self.config.max_rounds
        return RoundAdvanceResult(
            new_round_index=self.round_index,
            matured_feedback=matured,
            available_capital=self.available_capital,
            capital_at_risk=self.capital_at_risk,
            done=self.done,
        )

    def eventual_pending_capital_return(self) -> float:
        """Evaluator-only value of capital returning after the horizon."""

        return float(
            sum(
                loan.principal
                * (
                    self.config.recycle_fraction
                    if loan.repaid
                    else self.config.default_recovery_fraction
                )
                for loan in self.pending_loans
            )
        )

    def eventual_reward(self) -> float:
        """Evaluator-only eventual reward of all loans funded so far."""

        return float(
            sum(row["hidden_reward"] for row in self.funding_history)
        )


# Short compatibility alias for callers that prefer environment-style naming.
CMDPEnvironment = StatefulMicrofinanceCMDPEnv
