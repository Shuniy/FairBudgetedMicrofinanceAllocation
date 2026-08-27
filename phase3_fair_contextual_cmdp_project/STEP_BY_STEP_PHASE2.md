# Step-by-Step Phase 2 Build Guide

## Step 1 — Keep Phase 1 as the benchmark

The existing `phase1/` package remains unchanged. Phase 2 adds borrower-level context while keeping the same fundamental budget and binary repayment objective.

## Step 2 — Define `ApplicationContext`

`models/application_context.py` contains the observable application fields. Four financial fields were added because a contextual lending simulator cannot operate on demographic fields alone:

```python
requested_loan_amount
existing_debt
previous_repayment_rate
dependants
```

## Step 3 — Initialize derived borrower features

`Borrower.__post_init__()` calls `calculate_scores_features()` automatically. This prevents `is_socially_marginalized()` from accessing `protected_groups` before it exists.

## Step 4 — Encode borrowers

`BorrowerFeatureEncoder` transforms each borrower into a 28-dimensional vector. The vector is shared by all contextual policies.

```python
context = encoder.transform(borrower)
```

## Step 5 — Generate a borrower pool

`SyntheticBorrowerGenerator` creates a new list of candidates for each round. Every candidate stores:

```python
borrower
context
true_repayment_probability  # hidden from learning policies
```

## Step 6 — Define the hidden repayment process

The simulator calculates a hidden probability from repayment history, affordability, debt, urgency, location, sector and purpose. Repayment is then sampled:

```python
reward = rng.binomial(1, true_repayment_probability)
```

## Step 7 — Implement the contextual environment

At each round, the environment exposes:

```python
contexts
costs
groups
remaining_budget
feasible_actions
```

The chosen borrower leaves the pool and a new pool is generated.

## Step 8 — Implement contextual epsilon-greedy

The policy estimates:

```text
estimated repayment = theta^T context
```

It chooses randomly with probability epsilon, otherwise the highest estimate. The budget-aware version divides the estimate by loan cost.

## Step 9 — Implement LinUCB

LinUCB scores borrower `x` using:

```text
theta^T x + alpha * sqrt(x^T A^-1 x)
```

Budgeted LinUCB divides this optimistic repayment score by the requested loan amount.

## Step 10 — Implement contextual Thompson Sampling

The policy samples one plausible parameter vector from its posterior and scores all borrowers with that sampled model. The budget-aware version divides scores by cost.

## Step 11 — Add fairness wrappers

Hard fairness first checks quota deficits, then funding-rate parity. If an underrepresented group is present in the current pool, selection is restricted to that group.

Soft fairness adds a quota-deficit bonus to each candidate's utility score.

## Step 12 — Calculate Phase-2 fairness correctly

The environment counts both:

```python
group_funding_counts
group_opportunity_counts
```

Therefore:

```text
funding rate = funded / opportunities
```

and demographic parity is the largest difference between group funding rates.

## Step 13 — Run repeated-seed experiments

`experiments/run_phase2.py` runs all policies on the same seeded applicant streams and reports means and standard deviations.

## Step 14 — Compare Phase 1 and Phase 2

Compare policies at similar fairness levels using:

- cumulative reward;
- reward per cost;
- repayment rate;
- budget utilization;
- quota satisfaction;
- demographic parity gap.

The main research question is whether borrower context improves repayment or cost efficiency without increasing unfair group funding-rate gaps.
