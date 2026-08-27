# Fair Budgeted Microfinance Loan Allocation — Phases 1–3

This repository now contains all three implementation phases.

- **Phase 1:** fixed borrower-group MAB/BwK baselines.
- **Phase 2:** borrower-level contextual bandits with opportunity-based fairness metrics.
- **Phase 3:** minimal contextual constrained MDP with persistent portfolio state, delayed repayment, capital recycling, and hard/Lagrangian fairness constraints.

## Phase 3 design

The Phase-3 model follows the stateful plan in the project presentation: budget/history/fairness become persistent state, borrower context still drives repayment learning, the contextual policy proposes actions, and a CMDP wrapper handles long-run constraints. Repayment feedback is delayed and successful principal returns to the lending pool.

### Policies

- `random_cmdp`
- `context_only_budgeted_linucb` — Phase-2-style control inside the stateful environment
- `portfolio_linucb` — state-dependent capital shadow price
- `portfolio_thompson`
- `cmdp_fair_hard`
- `cmdp_fair_lagrangian`
- `oracle_portfolio` — simulator-only benchmark

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Test

```bash
pytest -q
```

### Run Phase 3

```bash
python experiments/run_phase3.py --seeds 10 --rounds 200 --delay 8
```

### Run the delay ablation

```bash
python experiments/run_phase3_delay_sweep.py --seeds 10 --rounds 200 --delays 2 5 8 12 20
```

See `STEP_BY_STEP_PHASE3.md` for the detailed implementation explanation.

## Important interpretation notes

1. **Capital is a stock in Phase 3.** Cumulative loan disbursement may exceed initial capital because successful principal is recycled.
2. **Reward remains binary repayment.** This preserves comparability with earlier phases; it is not a profit or poverty-impact measure.
3. **Protected attributes remain excluded from repayment prediction by default.** They are used for fairness monitoring and constraint enforcement.
4. **The Lagrangian policy is a minimal research prototype, not a claim of a solved optimal CMDP.** Its purpose is to test whether persistent state and adaptive constraint prices improve long-horizon behaviour under delay.
