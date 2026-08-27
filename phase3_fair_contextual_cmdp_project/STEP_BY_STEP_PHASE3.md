# Phase 3 — Contextual Constrained MDP

Phase 3 extends the borrower-level contextual bandit from Phase 2 with persistent portfolio state, delayed repayment, capital recycling, and adaptive fairness constraints.

## 1. State

At round `t`, the environment retains:

- available capital;
- capital at risk in outstanding loans;
- pending repayment count and time to maturity;
- recent and cumulative repayment history;
- group funding and opportunity counts;
- quota deficits and demographic-parity gap;
- cumulative disbursement, recycled capital, and default loss.

The state is also logged as a compact bucket such as `B1-F2-R1-Q1` for liquidity, fairness, repayment, and quota status.

## 2. Action

A round represents a lending cycle. A new borrower pool arrives and the policy may fund up to `max_loans_per_round` applicants. Each funded applicant is removed from the current pool.

## 3. Transition

Funding immediately moves principal from available capital to capital-at-risk. After `repayment_delay_rounds`, the hidden binary outcome is revealed:

- repayment: principal is recycled back into available capital;
- default: only the configured recovery fraction returns; the rest is recorded as loss.

The learner only updates after the outcome matures.

## 4. Reward

The core reward remains the binary repayment indicator used in Phases 1 and 2. `cumulative_reward` reports the eventual repayment outcome of every funded loan, while `observed_reward_by_horizon` reports only repayments whose feedback arrived before the decision horizon ended.

## 5. Portfolio-aware contextual policy

`ContextOnlyBudgetedLinUCBPolicy` is the Phase-2-style control. It uses borrower context and cost but ignores portfolio state.

`PortfolioAwareLinUCBPolicy` keeps the borrower repayment model but adds a state-dependent shadow price for principal. The shadow price rises when liquid capital is scarce or defaults accumulate and falls when substantial capital is due to return soon.

This makes state change the ranking of borrowers rather than merely appending identical state features to every candidate.

## 6. Hard constrained CMDP policy

`HardFairCMDPPolicy` applies the priority:

1. repair quota deficits;
2. repair demographic-parity violations;
3. otherwise follow portfolio-aware LinUCB.

## 7. Lagrangian CMDP policy

The soft constrained policy scores applicant `i` using:

`adjusted utility = normalized utility - lambda_quota * projected quota violation - lambda_parity * projected parity violation`

The dual variables are updated every round. They rise when violations exceed their tolerances and fall when constraints are comfortably satisfied.

## 8. Phase-3 capital metrics

Because capital is recycled, Phase-2 `budget_utilization = cumulative spend / initial budget` is no longer appropriate: cumulative disbursement can exceed initial capital. Phase 3 therefore adds:

- `capital_turnover = cumulative_disbursed / initial_capital`;
- `cumulative_recycled`;
- `capital_at_risk`;
- `final_available_capital`;
- `capital_survival_ratio`;
- `eventual_terminal_capital`.

## 9. Main experiment

```bash
python experiments/run_phase3.py \
  --seeds 10 \
  --rounds 200 \
  --pool-size 20 \
  --batch-size 3 \
  --capital 600000 \
  --delay 8
```

## 10. Delay sweep

```bash
python experiments/run_phase3_delay_sweep.py \
  --seeds 10 \
  --rounds 200 \
  --delays 2 5 8 12 20
```

The important scientific comparison is `context_only_budgeted_linucb` versus `portfolio_linucb` and the two constrained CMDP policies under increasing repayment delay. If state adds value, the benefit should become clearer when capital remains tied up for longer.
