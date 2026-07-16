# Fair Budgeted Microfinance Bandits

This project provides one implementation for two related experiments:

- **arm mode** learns which borrower-group distribution to fund when loan costs
  are sampled from configured means and standard deviations;
- **contextual mode** generates a fresh pool of individual applications and
  learns which affordable borrower to fund from observable application features.

Both modes use the same budget lifecycle, feedback schema, fairness accounting,
evaluation pipeline, metrics, result files, and fairness wrappers.

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

## Run

```bash
python main.py --mode arm
python main.py --mode contextual
```

Shared options are `--seeds`, `--rounds`, `--budget`, and `--output`.
Contextual mode also accepts `--pool-size` and `--feedback-delay`. Arm mode can
load distributions from `--arms-config path/to/arms.json`:

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

Results are printed and written to `outputs/arm/` or `outputs/contextual/`:

- `episode_summaries.csv`
- `trajectories.csv`
- `aggregate_results.csv`
- cumulative reward, repayment rate, parity gap, and quota satisfaction plots

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
