# Unified Model Guide

The runtime is organized by responsibility rather than by project stage.

## Shared experiment flow

```text
configuration -> available affordable choices -> policy score -> selection
              -> simulated repayment -> feedback -> fairness and utility metrics
```

`EnvironmentConfig` defines the budget and horizon. `FairnessConfig` defines
minimum group shares and parity settings. `AlgorithmConfig` holds learning
parameters. `FeedbackEvent`, `StepResult`, and `EpisodeResult` give both modes a
common update and reporting contract.

`BanditEnvironment` is the canonical concrete environment. Create it with
`BanditEnvironment.for_arms(...)` or
`BanditEnvironment.for_contextual(...)`. Its shared interface provides the
current action count, costs, groups, feasible actions, hidden evaluation
probabilities, observations, reset, and step behavior. Contextual mode
additionally exposes feature contexts, while the legacy mode-specific class
names remain thin compatibility entry points.

## Arm simulation

`ArmSpec` represents a borrower-group distribution with a repayment probability,
loan-cost mean, and loan-cost standard deviation. In arm mode the unified
environment samples an affordable cost for each arm every round. Arm policies
learn one reward/cost history per distribution.

## Contextual simulation

`ApplicationContext` is the observable application. `Borrower` derives protected
memberships, income band, poverty-relief score, and vulnerability score.
`BorrowerFeatureEncoder` converts non-protected application data into a numeric
vector. In contextual mode, `SyntheticBorrowerGenerator` calculates a hidden
repayment probability used only to sample outcomes. Contextual policies learn a
shared feature-to-repayment model and choose one affordable borrower from each
pool.

## Fairness

Both environments track funded counts and affordable opportunities. Quotas use
allocation shares, while demographic parity uses funded/opportunity rates.
`HardFairPolicy` restricts selection when quotas or parity are violated;
`SoftFairPolicy` adds a configurable score bonus for underrepresented groups.
