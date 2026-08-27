from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping

import numpy as np

from phase2.feature_encoder import BorrowerFeatureEncoder
from phase2.metrics import (
    allocation_shares,
    demographic_parity_gap,
    group_funding_rates,
    quota_deficits,
)
from phase2.models import ContextualEnvironmentConfig
from phase2.simulator import SyntheticBorrowerGenerator
from phase3.models import (
    CMDPEnvironmentConfig,
    FundingResult,
    MaturedLoanFeedback,
    PendingLoan,
    RoundAdvanceResult,
)


class StatefulMicrofinanceCMDPEnv:
    """Minimal contextual CMDP for delayed microfinance lending.

    Phase-2 contextual bandits only tracked a one-way remaining budget. This
    Phase-3 environment adds persistent portfolio state:

    * available capital;
    * outstanding principal / capital-at-risk;
    * delayed repayment outcomes;
    * capital recycling after repayment;
    * default losses;
    * running fairness state and repayment history.

    A round represents a lending cycle. A new applicant pool arrives once per
    round and the policy may fund up to `max_loans_per_round` applicants from
    that pool. Outcomes are hidden until their maturity round.
    """

    def __init__(
        self,
        config: CMDPEnvironmentConfig,
        encoder: BorrowerFeatureEncoder | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder or BorrowerFeatureEncoder(include_protected_attributes=False)
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

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        actual_seed = self.config.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)
        phase2_config = ContextualEnvironmentConfig(
            total_budget=self.config.initial_capital,
            max_rounds=self.config.max_rounds,
            pool_size=self.config.pool_size,
            seed=actual_seed,
            feedback_delay_rounds=0,
            minimum_cost=self.config.minimum_cost,
            female_probability=self.config.female_probability,
            marginalized_probability=self.config.marginalized_probability,
            reserved_probability=self.config.reserved_probability,
            disability_probability=self.config.disability_probability,
            rural_probability=self.config.rural_probability,
        )
        self.generator = SyntheticBorrowerGenerator(
            config=phase2_config,
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
        self.recent_outcomes: deque[float] = deque(maxlen=30)
        self.observed_repayments = 0
        self.observed_defaults = 0
        self.cumulative_disbursed = 0.0
        self.cumulative_recycled = 0.0
        self.cumulative_default_loss = 0.0
        self.current_pool = self.generator.generate_pool(self.config.pool_size)
        self._record_pool_opportunities(self.current_pool)
        self._matured_at_round_start: tuple[MaturedLoanFeedback, ...] = tuple()
        self._done = False
        self._refresh_done_flag()
        return self.observation()

    def _record_pool_opportunities(self, pool) -> None:
        for candidate in pool:
            group = candidate.fairness_group
            self.group_opportunity_counts[group] += 1
            self.group_funding_counts.setdefault(group, 0)

    def base_context_matrix(self) -> np.ndarray:
        if not self.current_pool:
            return np.empty((0, self.encoder.n_features), dtype=float)
        return np.vstack([candidate.context for candidate in self.current_pool])

    def current_costs(self) -> np.ndarray:
        return np.asarray([candidate.cost for candidate in self.current_pool], dtype=float)

    def current_groups(self) -> tuple[str, ...]:
        return tuple(candidate.fairness_group for candidate in self.current_pool)

    def true_probabilities(self) -> np.ndarray:
        return np.asarray(
            [candidate.true_repayment_probability for candidate in self.current_pool],
            dtype=float,
        )

    def feasible_actions(self) -> list[int]:
        spendable = self.available_capital - self.config.minimum_liquidity_reserve
        if spendable < self.config.minimum_cost - 1e-12:
            return []
        return [
            index
            for index, candidate in enumerate(self.current_pool)
            if candidate.cost <= spendable + 1e-12
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
        return max(0, min(loan.due_round for loan in self.pending_loans) - self.round_index)

    def fairness_snapshot(self, target_shares: Mapping[str, float]) -> dict[str, Any]:
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
            "parity_violation": max(0.0, gap - parity_tolerance),
        }

    def state_bucket(
        self,
        target_shares: Mapping[str, float] | None = None,
        parity_tolerance: float = 0.05,
    ) -> str:
        targets = dict(target_shares or {})
        liquidity = self.available_capital / self.config.initial_capital
        if liquidity < 0.20:
            b = 0
        elif liquidity < 0.55:
            b = 1
        else:
            b = 2

        gap = demographic_parity_gap(
            self.group_funding_counts,
            self.group_opportunity_counts,
        )
        if gap <= parity_tolerance:
            f = 0
        elif gap <= 2 * parity_tolerance:
            f = 1
        else:
            f = 2

        rr = self.recent_repayment_rate
        if rr < 0.55:
            r = 0
        elif rr < 0.75:
            r = 1
        else:
            r = 2

        quota = sum(quota_deficits(self.group_funding_counts, targets).values())
        q = 0 if quota <= 0.01 else (1 if quota <= 0.10 else 2)
        return f"B{b}-F{f}-R{r}-Q{q}"

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
        }

    def fund(self, action: int, learning_context: np.ndarray) -> FundingResult:
        if self.done:
            raise RuntimeError("Cannot fund in a completed environment")
        if action < 0 or action >= len(self.current_pool):
            raise IndexError(f"Invalid action {action}")
        if action not in self.feasible_actions():
            raise ValueError("Selected borrower is not currently affordable")

        candidate = self.current_pool.pop(action)
        principal = float(candidate.cost)
        reward = float(self.rng.binomial(1, candidate.true_repayment_probability))
        due_round = self.round_index + self.config.repayment_delay_rounds

        self.available_capital -= principal
        self.cumulative_disbursed += principal
        self.group_funding_counts[candidate.fairness_group] += 1
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
        )

    def _release_due_loans(self) -> tuple[MaturedLoanFeedback, ...]:
        matured = [loan for loan in self.pending_loans if loan.due_round <= self.round_index]
        if not matured:
            return tuple()
        self.pending_loans = [
            loan for loan in self.pending_loans if loan.due_round > self.round_index
        ]
        feedback: list[MaturedLoanFeedback] = []
        for loan in matured:
            if loan.repaid:
                returned = loan.principal * self.config.recycle_fraction
                self.observed_repayments += 1
            else:
                returned = loan.principal * self.config.default_recovery_fraction
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
        """Move to the next lending cycle and release repayments due there."""

        if self.done:
            return RoundAdvanceResult(
                new_round_index=self.round_index,
                matured_feedback=tuple(),
                available_capital=self.available_capital,
                capital_at_risk=self.capital_at_risk,
                done=True,
            )

        self.round_index += 1
        # Repayments due exactly at the end of the decision horizon still
        # mature and are counted as observed-by-horizon outcomes.
        matured = self._release_due_loans()
        if self.round_index < self.config.max_rounds:
            self.current_pool = self.generator.generate_pool(self.config.pool_size)
            self._record_pool_opportunities(self.current_pool)
        else:
            self.current_pool = []
        self._matured_at_round_start = matured
        self._refresh_done_flag()
        return RoundAdvanceResult(
            new_round_index=self.round_index,
            matured_feedback=matured,
            available_capital=self.available_capital,
            capital_at_risk=self.capital_at_risk,
            done=self.done,
        )

    def eventual_pending_capital_return(self) -> float:
        """Evaluator-only value of capital that will eventually return after horizon."""

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
        """Evaluator-only eventual binary reward of all loans funded so far."""

        return float(sum(row["hidden_reward"] for row in self.funding_history))

    def _refresh_done_flag(self) -> None:
        horizon_done = self.round_index >= self.config.max_rounds
        permanently_illiquid = (
            not self.pending_loans
            and self.available_capital - self.config.minimum_liquidity_reserve
            < self.config.minimum_cost
        )
        self._done = horizon_done or permanently_illiquid
