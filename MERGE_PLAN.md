# Phase 1–3 merge analysis and reusable component plan

## Outcome

The parent project is the canonical implementation. Phase 3 has been integrated
without copying its duplicate Phase 1/2 packages. The original
`phase3_fair_contextual_cmdp_project/` directory remains only as a reference;
runtime code and canonical tests use parent modules exclusively.

The runner exposes four choices:

1. arm bandits over borrower-group reward/cost distributions;
2. contextual bandits over generated `ApplicationContext` borrower pools;
3. a stateful contextual CMDP with delayed repayment and recyclable capital;
4. all three experiments, their individual metrics, and one combined table.

## Logic analysis

Phases 1 and 2 share a one-shot budget lifecycle. Every decision spends budget,
selects one action, records immediate or configured delayed feedback, and moves
to a new decision round. They can therefore share `BanditEnvironment`,
`BasePolicy`, `FeedbackEvent`, `StepResult`, and the same episode runner.

Phase 3 shares the borrower and learning problem but not the budget lifecycle.
It may fund several borrowers in a lending cycle, moves principal into
capital-at-risk, waits for maturity, recycles repaid principal, loses defaulted
principal, and changes constraint prices after a round. Consequently, its
environment, action loop, feedback event types, and policy lifecycle stay in
segregated CMDP files. Forcing these into `BanditEnvironment.step()` would make
the common interface misleading and would couple Phases 1/2 to portfolio state.

Protected attributes remain available for fairness auditing but are excluded
from repayment prediction by default in all contextual modes. The hidden true
repayment probability remains simulator/evaluator-only and is exposed to oracle
benchmarks only.

## Reusable classes consolidated in parent files

| Parent file | Reusable class | Used by |
|---|---|---|
| `models/application_context.py` | `ApplicationContext` | Phases 2 and 3 |
| `Data/borrower.py` | `Borrower` and its protected-group/need derivations | Phases 2 and 3 |
| `models/borrower_candidate.py` | `BorrowerCandidate` | Phases 2 and 3 |
| `Data/feature_encoder.py` | `BorrowerFeatureEncoder` | Phases 2 and 3 |
| `simulation.py` | `SyntheticBorrowerGenerator` | Phases 2 and 3 |
| `models/arm_spec.py` | `ArmSpec` | Phase 1 |
| `models/environment_config.py` | `EnvironmentConfig`, `ContextualEnvironmentConfig`, `CMDPEnvironmentConfig` | Mode-specific configuration in all phases |
| `models/fairness_config.py` | `FairnessConfig`, `LagrangianConfig` | Static fairness in Phases 1/2 and adaptive fairness in Phase 3 |
| `models/events.py` | `FeedbackEvent`, `StepResult`, `PendingLoan`, `MaturedLoanFeedback`, `FundingResult`, `RoundAdvanceResult` | Shared event home with phase-appropriate event types |
| `Algorithms/contextual.py` | `LinearRidgeModel` | Phase 2 contextual policies and Phase 3 CMDP policies |
| `Environments/base_environment.py` | `BanditEnvironment` | Phases 1 and 2 |
| `Environments/cmdp_environment.py` | `StatefulMicrofinanceCMDPEnv` / `CMDPEnvironment` | Phase 3 only |
| `Algorithms/bandit.py` | `BasePolicy`, `ArmPolicy` | Phases 1 and 2 |
| `Algorithms/cmdp.py` | `CMDPPolicy` and CMDP policy family | Phase 3 only |
| `evaluate.py` | `EpisodeResult`, `CMDPEpisodeResult` | Common reporting home with separate lifecycle results |

`CMDPPolicy` intentionally does not inherit `BasePolicy`: CMDP action selection
may return no action while capital is temporarily illiquid, updates receive
matured loan contexts rather than `FeedbackEvent`, and policies need an
`on_round_end()` hook for dual updates.

## Reusable functions consolidated in parent files

### Population and representation

- `SyntheticBorrowerGenerator._generate_application()` creates the common
  observable application.
- `SyntheticBorrowerGenerator.true_repayment_probability()` is the common
  hidden simulator outcome rule.
- `generate_candidate()` and `generate_pool()` supply both contextual phases.
- `BorrowerFeatureEncoder.transform()` and `transform_many()` supply the common
  learning representation.
- `_validate_population_probabilities()` validates Phase 2/3 population knobs.
- `CMDPEnvironmentConfig.borrower_generation_config()` adapts Phase 3 settings
  into the shared generator without duplicating simulator code.

### Fairness and performance metrics

The following functions now live only in parent `metrics.py` and are reusable
by every applicable phase:

- `allocation_shares()`;
- `group_funding_rates()`;
- `demographic_parity_gap()`;
- `quota_deficits()`;
- `quota_satisfaction_rate()`;
- `total_quota_deficit()`;
- `parity_violation()`;
- `repayment_rate()`;
- `reward_per_unit_cost()` and its Phase 3 semantic name
  `reward_per_disbursed_capital()`;
- `budget_utilization()` for one-shot budgets;
- `capital_turnover()`, `liquidity_utilization()`, and
  `capital_survival_ratio()` for recyclable capital.

Budget utilization is deliberately not applied to Phase 3. Cumulative
disbursement may exceed initial capital when successful principal is recycled,
so capital turnover and survival are the correct portfolio measures.

### Evaluation and runner utilities

- `plot_metric_over_time()` renders both bandit and CMDP trajectories.
- `display_results()` prints every mode's aggregate table.
- `compare_policies()` and `run_episode()` cover the shared Phase 1/2 lifecycle.
- `compare_cmdp_policies()` and `run_cmdp_episode()` cover the stateful Phase 3
  lifecycle while using the same domain and metrics functions.
- `save_results()` and `save_cmdp_results()` use consistent neutral filenames;
  the CMDP saver adds funding and maturity audit logs.
- `build_arm_policy_factories()`, `build_contextual_policy_factories()`, and
  `build_cmdp_policy_factories()` provide one construction point per family.
- `combine_aggregate_results()` flattens mode aggregates, preserves mode/policy,
  and computes `rank_within_mode`.
- `run_all()` writes the complete cross-phase report.

## Segregated Phase 3 components

The following logic is necessarily Phase 3-specific and remains visibly
segregated:

- portfolio transitions and maturity processing in
  `Environments/cmdp_environment.py`;
- context-only control, portfolio-aware LinUCB/Thompson, oracle, hard fairness,
  and Lagrangian policies in `Algorithms/cmdp.py`;
- Phase 3 event dataclasses in the shared event file;
- CMDP episode/audit-log functions in `evaluate.py`;
- capital stock metrics in `metrics.py`.

## Verification plan and status

- [x] Inventory both repositories and trace all entry points.
- [x] Select the parent implementation for duplicate Phase 1/2 logic.
- [x] Merge Phase 3 configs/events into parent model files.
- [x] Repoint Phase 3 to parent borrower, encoder, simulator, metrics, and ridge model.
- [x] Add the dedicated CMDP environment and policy layer.
- [x] Add four runner options and the all-phases aggregate report.
- [x] Port Phase 3 lifecycle tests into the canonical parent test suite.
- [x] Isolate reference-project test discovery to prevent import shadowing.
- [x] Smoke-test individual modes and the combined mode through `main.py`.
