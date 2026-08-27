# Fair Budgeted Microfinance — Phases 1–3

This project provides one runner and shared domain layer for three experiments:

- **Phase 1 / arm mode** learns which borrower-group distribution to fund when loan costs
  are sampled from configured means and standard deviations;
- **Phase 2 / contextual mode** generates a fresh pool of individual applications
  and learns which affordable borrower to fund from observable application features;
- **Phase 3 / CMDP mode** adds persistent portfolio state, delayed repayment,
  capital-at-risk, principal recycling, and hard or Lagrangian fairness constraints.

All phases reuse the same borrower/application models, feature encoding, synthetic
population, fairness accounting, core metrics, ridge learner, plotting, and result
display. Phase 3 keeps its stateful environment and policy contract in dedicated
files because recyclable capital and batch lending are different from the one-shot
budget lifecycle used by Phases 1 and 2.

## Unified environment API

`BanditEnvironment` is the canonical environment for both experiment modes. It
uses named constructors so mode-specific inputs remain explicit:

```python
from Environments.base_environment import BanditEnvironment

arm_env = BanditEnvironment.for_arms(arms, environment_config)
contextual_env = BanditEnvironment.for_contextual(
    contextual_environment_config,
    encoder,
)
```

Both forms expose the same budget, action, fairness, feedback, and episode
lifecycle methods. Existing imports named `ArmBanditEnvironment`,
`SyntheticEnvironment`, `ContextualBanditEnvironment`,
`ContextualMicrofinanceEnv`, and `BaseBudgetEnvironment` remain available as
compatibility wrappers or aliases.

## Install

```bash
python -m pip install -r requirements.txt
```

## Four runner options

```bash
python main.py --mode 1  # Phase 1: arm bandits
python main.py --mode 2  # Phase 2: contextual bandits
python main.py --mode 3  # Phase 3: contextual CMDP
python main.py --mode 4  # all phases plus a combined metrics table
```

The named forms `arm`, `contextual`, `cmdp`, and `all` are equivalent. Running
`python main.py` without `--mode` opens the same four-option menu.

Shared options are `--seeds`, `--rounds`, `--budget`, and `--output`. Contextual
mode also accepts `--pool-size` and `--feedback-delay`. CMDP mode accepts
`--pool-size`, `--batch-size`, `--repayment-delay`, `--recycle-fraction`,
`--default-recovery`, `--liquidity-reserve`, and `--parity-tolerance`. Arm mode
can load distributions from `--arms-config path/to/arms.json`:

```json
{
  "arms": [
    {
      "name": "Group A",
      "group": "A",
      "repayment_probability": 0.8,
      "cost_mean": 12.0,
      "cost_std": 1.2
    }
  ],
  "target_shares": {"A": 0.2}
}
```

If no arm file is supplied, three built-in groups are used and every cost
standard deviation is 10% of its mean.

Phase 1/2 results are printed and written to `outputs/arm/` or
`outputs/contextual/`:

- `episode_summaries.csv`
- `trajectories.csv`
- `aggregate_results.csv`
- cumulative reward, repayment rate, parity gap, and quota satisfaction plots

Contextual mode evaluates budget-aware epsilon-greedy, LinUCB, and contextual
Thompson sampling under unconstrained, hard-fair, and soft-fair selection. The
existing non-budgeted, random, and oracle policies remain as baselines.

Phase 3 writes the same summary/trajectory/aggregate files under `outputs/cmdp/`,
plus `funding_log.csv`, `maturity_log.csv`, and portfolio-state plots. All mode
writes each phase beneath `outputs/all/` and adds
`combined_aggregate_results.csv`. Rankings in that table are within each mode;
raw reward and cost scales are not assumed to be directly comparable across
different phase lifecycles.

Contextual and CMDP runs also write `ranked_results.csv`, `winner_summary.txt`,
`run_manifest.json`, and a bordered `report.html`. A policy qualifies as
fairness-feasible when its mean parity gap and quota deficit meet their configured
thresholds; feasible policies are ranked by reward per cost and cumulative reward.
When none qualify, the report labels the minimum-violation result as the best
compromise. CMDP mode applies the same unconstrained/hard/Lagrangian matrix to
portfolio-aware epsilon-greedy, LinUCB, and Thompson sampling.

## Fairness and prediction

Allocation shares measure the distribution of funded loans. Funding rates use
only affordable choices that were actually available at a decision. Protected
attributes are retained for fairness auditing but are excluded from contextual
repayment prediction by default. Hidden repayment probabilities are visible only
to the simulator, exported evaluation data, and explicitly labelled true reward
benchmarks.

## Test

```bash
python -m pytest -q
```

The imported `phase3_fair_contextual_cmdp_project/` directory is retained as a
reference snapshot. The merged parent implementation does not import from it.
See `MERGE_PLAN.md` for the consolidation analysis and reusable component map.
