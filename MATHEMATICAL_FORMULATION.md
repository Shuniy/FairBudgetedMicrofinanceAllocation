# Mathematical formulation

This reference describes the three simulator modes and the policies registered in [main.py](main.py). Round indices are zero-based. Defaults below are the experiment factory settings where those differ from class defaults.

## Shared notation, reward and feasibility

At decision $t$, action $i$ has currently observed cost $c_{t,i}>0$, group $g_{t,i}$ and simulator repayment probability $p_{t,i}$. A funded loan draws $r_{t,i}\sim\operatorname{Bernoulli}(p_{t,i})$. Phase 1 uses an arm's fixed $p_i$ and a fresh cost each round, $c_{t,i}=\max(c_{\min},Z_{t,i})$, where $c_{\min}$ is the configured cost floor and $Z_{t,i}$ is drawn from the arm's configured normal distribution or set to its mean when the standard deviation is zero. Phases 2–3 use requested loan principal and the synthetic borrower's probability, a logistic function of borrower and loan attributes clipped to $[0.05,0.95]$ by `SyntheticBorrowerGenerator.true_repayment_probability` in [simulation.py](simulation.py). Only the informed policies use $p_{t,i}$ to select actions. Repayment counts and repayments per unit of principal measure simulated lending outcomes. The simulator supplies no monetary profit or causal poverty-alleviation measure.

Let $T_{\max}$ be the configured horizon (`max_rounds`) and $T\le T_{\max}$ the realized number of rounds. In Phases 1–2, each round funds one action $a_t$. A Phase 1 action identifies a borrower-group arm, and a Phase 2 action identifies an individual borrower candidate in the current pool. Let $B_0$ be the initial budget and $B_t$ the remaining budget before round $t$. The feasible set $\mathcal F_t$ contains the affordable current offers for $t<T_{\max}$ and is empty at the horizon. The episode ends at the horizon or at the first offer set with no feasible action, so $T$ can depend on the policy and can be zero. Repayment leaves the remaining budget unchanged. See `BanditEnvironment.feasible_actions`, `step` and `_refresh_done_flag` in [base_environment.py](Environments/base_environment.py).

Phase 1 learns immediately. With Phase 2 feedback delay $d_{\mathrm{fb}}$, funding at round $s$ releases feedback during the step at $s+d_{\mathrm{fb}}$ if that step occurs, after its decision. The next decision can use it if the episode continues. The runner records outcomes at funding for evaluation and delivers only released events to learners. See `run_episode` in [evaluate.py](evaluate.py).

`BasePolicy.select_action` in [bandit.py](Algorithms/bandit.py) and `CMDPPolicy.select_action` in [cmdp.py](Algorithms/cmdp.py) intersect any supplied candidate subset with feasible indices and sample uniformly among sorted options whose scores equal the maximum within NumPy's `isclose` tolerance. Same-sign infinities tie. Phase 1–2 raise when the set is empty. Phase 3 returns `None`. KUBE and hard-fairness wrappers have the additional selection stages below.

### Finite-horizon optimisation in Phases 1–2

For each non-recycling phase, let $\pi$ be a policy that selects $a_t$ using the information available at decision $t$. The expectation averages over simulator and policy randomness. The research problem is

$$
\begin{aligned}
\max_{\pi}\quad &\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}r_{t,a_t}\right] \\
\text{subject to}\quad &a_t\in\mathcal F_t\qquad (0\le t<T),\\
&\sum_{t=0}^{T-1}c_{t,a_t}\le B_0\qquad\text{almost surely}.
\end{aligned}
$$

Action feasibility enforces the pathwise budget constraint through $\mathcal F_t=\{i:c_{t,i}\le B_t\}$ in the exact formulation. The simulator checks $c_{t,i}\le B_t+10^{-12}$ and updates $B_{t+1}=\max(0,B_t-c_{t,a_t})$. Recorded principal can therefore differ from the change in remaining budget near exhaustion.

The terminal fairness metrics are total quota deficit $Q_T$ and the phase-specific gap $G_T$ defined [below](#fairness-metrics-and-phase-12-wrappers). Expected-value targets $\mathbb E_\pi[Q_T]\le\tau_q$ and $\mathbb E_\pi[G_T]\le\tau_p$ allow individual episodes to exceed either threshold. Evaluation reports the empirical means and the fraction of individual episodes meeting the thresholds. The policies implement heuristics without a guarantee of terminal or anytime fairness.

### Primary evaluation metrics

In Phases 1–2, cumulative reward is the number of successful repayments and cumulative cost is the total principal disbursed. For an episode,

$$
R_T=\sum_{t=0}^{T-1}r_{t,a_t},\qquad
C_T=\sum_{t=0}^{T-1}c_{t,a_t},\qquad
RPC_T=\frac{R_T}{C_T},\qquad
RR_T=\frac{R_T}{N_{\mathrm{funded}}},\qquad
BU_T=\frac{B_0-B_T}{B_0}.
$$

Here $N_{\mathrm{funded}}=T$ in Phases 1–2 and $B_T$ is the terminal remaining budget. Repayment rate $RR_T$ is the fraction of funded loans that repay. Reward per unit cost $RPC_T$ counts successful repayments per unit of principal. Monetary profit would require cash flows such as interest and lending expenses. Reported rewards use all sampled outcomes at funding, including outcomes still awaiting learner feedback in Phase 2.

Empty sums are zero. The summary uses zero for $RPC_T$ when $C_T\le0$ and for $RR_T$ when no loans were funded. Budget utilisation $BU_T$ measures the fraction of the one-use budget spent. The environment requires $B_0>0$, and the `budget_utilization` helper divides by $B_0$ directly with no zero guard. See `reward_per_unit_cost`, `repayment_rate` and `budget_utilization` in [metrics.py](metrics.py), and `run_episode` in [evaluate.py](evaluate.py).

In Phase 3, use $R_T=R_{\mathrm{eventual}}$ and the batch cost $C_t^{\mathrm{batch}}$ defined [below](#phase-3-cmdp), so $C_T=\sum_{t=0}^{T-1}C_t^{\mathrm{batch}}$ and $N_{\mathrm{funded}}=\sum_{t=0}^{T-1}|\mathcal A_t|$, where $\mathcal A_t$ is the funded batch. The same reward-per-cost and repayment-rate formulas and zero conventions apply. The summary reports $R_{\mathrm{observed}}$ separately as `observed_reward_by_horizon`. The trajectory's `observed_repayment_rate` divides observed successes by the number of matured loans and uses $0.5$ before any loan matures.

Phase 3 capital turnover is $C_T/B_{\mathrm{init}}$, where $B_{\mathrm{init}}$ is initial capital. It can exceed one because returned principal can fund more loans. The Phase 3 sweeps also label the net decline in available cash relative to initial capital as `budget_utilization`. See `_add_sweep_metrics` in [robustness.py](robustness.py).

Liquidity utilisation is $P_{\mathrm{now}}/(B_{\mathrm{now}}+P_{\mathrm{now}})$, and capital survival ratio is $(B_{\mathrm{now}}+P_{\mathrm{now}})/B_{\mathrm{init}}$, where $B_{\mathrm{now}}$ is current available cash and $P_{\mathrm{now}}$ is outstanding principal at face value. Liquidity utilisation is zero when the sum in its denominator is nonpositive. The turnover and survival helpers return zero for nonpositive initial capital, which the configuration disallows. The runner logs all three using post-funding totals and balances before each round advance. The summary uses total disbursements for turnover and terminal balances for survival. See the capital metric helpers in [metrics.py](metrics.py), and `run_cmdp_episode` in [evaluate.py](evaluate.py).

## Phase 1 group-level bandits

`ArmPolicy.reset/update` in [bandit.py](Algorithms/bandit.py) initialize and update the count $n_i$, reward sum $S_i$, and cost sum $D_i$ from zero. Each released event for $i$ adds $(1,r,c)$ to $(n_i,S_i,D_i)$ and increments $N=\sum_i n_i$. For $n_i>0$, $\hat r_i=S_i/n_i$ and $\hat c_i=D_i/n_i$. At zero observations the code exposes $\hat r_i=0$ and $\hat c_i=+\infty$. The registered cost-aware policies use the current $c_{t,i}$ in selection.

| Policy and implementation | Score or selection rule | Verification in [test_policies.py](tests/test_policies.py) |
| --- | --- | --- |
| Random, `RandomPolicy.scores` in [bandit.py](Algorithms/bandit.py) | Independent $\operatorname{Uniform}[0,1)$ scores give uniform feasible selection. Baseline. | `test_policies_run_directly_against_canonical_environment` |
| Epsilon-greedy, `EpsilonGreedyPolicy` in [epsilon_greedy.py](Algorithms/epsilon_greedy.py) | If initial exploration is enabled and an unobserved arm is eligible, give unobserved arms $+\infty$ and observed arms $-\infty$. Otherwise, with probability $\epsilon=0.10$ use random scores and with probability $1-\epsilon$ use $\hat r_i$. Standard rule with a feasible initialization stage. | `test_epsilon_greedy_forces_only_available_unobserved_arms`, `test_score_ties_distinguish_positive_and_negative_infinity` |
| Decaying epsilon-greedy, `DecayingEpsilonGreedyPolicy` in [epsilon_greedy.py](Algorithms/epsilon_greedy.py) | Same feasible initialization and scoring, with $\epsilon(N)=\max(0.05,1/(1+0.02N))$. Feedback-count schedule. | `test_epsilon_greedy_forces_only_available_unobserved_arms`, `test_decaying_epsilon_uses_feedback_count_and_floor` |
| UCB, `UCBPolicy.scores` in [ucb.py](Algorithms/ucb.py) | $U_i=\hat r_i+\sqrt{\kappa\log(\max(2,N+1))/n_i}$ for $n_i>0$, and $+\infty$ otherwise. $\kappa=2$ gives a UCB1-style bonus with this simulator's update clock and feasible actions. | `test_ucb_and_budgeted_ucb_scores_use_observed_counts` |
| Thompson sampling, `ThompsonSamplingPolicy` in [thompson_sampling.py](Algorithms/thompson_sampling.py) | Standard Bernoulli–Beta model. Independently draw $\operatorname{Beta}(1+S_i,1+n_i-S_i)$ per arm. A success increments the first parameter and a failure increments the second. The unobserved-arm prior is uniform. | `test_thompson_updates_only_observed_arm_beta_parameters` |
| Reward-per-cost, `RewardPerCostPolicy.scores` in [reward_per_cost.py](Algorithms/reward_per_cost.py) | $\hat r_i/c_{t,i}$ when sampled, $+\infty$ otherwise. Greedy density heuristic with initial coverage. | `test_reward_per_cost_prefers_unobserved_feasible_arms_then_current_density` |
| Budgeted UCB, `BudgetedUCBPolicy.scores` in [budgeted_ucb.py](Algorithms/budgeted_ucb.py) | $U_i/c_{t,i}$. Cost-aware UCB adaptation. | `test_ucb_and_budgeted_ucb_scores_use_observed_counts`, `test_budgeted_ucb_and_fractional_kube_are_equivalent` |
| Fractional KUBE, `FractionalKUBEPolicy.scores` in [fractional_kube.py](Algorithms/fractional_kube.py) | $U_i/c_{t,i}$. Cost-aware fractional-knapsack index adaptation. | `test_budgeted_ucb_and_fractional_kube_are_equivalent` |
| KUBE, `KUBEPolicy.select_action` in [kube.py](Algorithms/kube.py) | KUBE-style unbounded knapsack and count-weighted sampling, detailed below. | `test_kube_recovers_mixed_counts_and_samples_by_multiplicity`, `test_kube_initial_exploration_and_rounded_capacity_fallback` |
| Epsilon-first, `EpsilonFirstPolicy.scores` in [epsilon_first.py](Algorithms/epsilon_first.py) | Spend-based exploration followed by greedy estimated density, detailed below. | `test_epsilon_first_spends_through_target_then_exploits`, `test_epsilon_first_excludes_unobserved_arms_at_threshold` |
| Hard and soft fair UCB, `HardFairPolicy` and `SoftFairPolicy` in [fairness.py](Algorithms/fairness.py) | Wrap Budgeted UCB with the quota/parity restriction or quota bonus defined below. Heuristics. | `test_hard_quota_then_parity_and_soft_quota_bonus` |
| `true_reward_per_cost`, `TrueRewardPerCostPolicy.scores` in [true_repayment_cost.py](Algorithms/true_repayment_cost.py) | $p_{t,i}/c_{t,i}$. Informed greedy benchmark. | `test_informed_greedy_benchmarks_choose_feasible_true_reward_per_cost` |

**Budgeted UCB and Fractional KUBE.** Both inherit the same UCB initialization and feedback updates, divide by the same observed cost, and use the same feasible selection and seeded tie-breaking. They agree under identical states, parameters, candidate sets and RNG states, including zero-count and tied scores. Both policy identifiers are retained. This is an implementation equivalence with scope limited to this repository. [Tran-Thanh et al. (2012), Section 3.3](https://arxiv.org/abs/1204.1909).

**KUBE.** In the feasible candidate order, sample uniformly among unobserved arms when any exist. Otherwise, with integer cost scale $k$ (default $1$), define capacity $W=\lfloor kB_t+10^{-9}\rfloor$, integer weights $w_i=\max(1,\operatorname{round}(kc_{t,i}))$ (Python nearest-integer, ties to even) and item values $U_i$. Ignoring the code's comparison tolerance, the at-most-capacity recurrence for integer $1\le j\le W$ is

$$
D(0)=0,\qquad D(j)=\max\bigl(0,\ \max_{i:\,w_i\le j}\{D(j-w_i)+U_i\}\bigr).
$$

The code retains the first maximizing item in candidate order, accepting an improvement above $10^{-12}$. Starting from $W$, follow stored items and subtract their weights until no item is recorded. If the counts $m_i$ are nonempty, draw one arm with probability $m_i/\sum_j m_j$. Otherwise use the inherited selector with $U_i/c_{t,i}$ on the feasible set. The rounded mix can exceed remaining cash in actual cost and can span more loans than the remaining rounds. Only the sampled arm is funded, and it is already feasible at actual cost. The program replans with fresh costs and the updated budget. For example, actual costs $(3.4,4.3)$ give integer weights $(3,4)$ at scale 1. With capacity 10 and values $(0.6,0.85)$, counts $(2,1)$ give value 2.05, exceeding the two single-arm mixes' values of 1.8 and 1.7. The published KUBE uses fixed arm costs and a density-ordered greedy approximation, with exhaustion of its budget governing the stopping time. This implementation uses the remaining budget as capacity for a rounded-cost dynamic program under a separate finite horizon. The published regret guarantees do not apply directly to this adaptation. [Tran-Thanh et al. (2012), Sections 2–4](https://arxiv.org/abs/1204.1909).

**Epsilon-first.** While spending before the decision satisfies $B_0-B_t<\epsilon B_0$ (default $\epsilon=0.20$), maximize $-n_i$ over feasible arms. The next feasible loan may cross the exploration target in full. At or above the target, maximize $\hat r_i/c_{t,i}$ for observed arms, giving unobserved arms $-\infty$. If only unobserved arms are feasible they tie, and if the policy has no observations at all it uses random scores. Rewards and counts continue updating during exploitation. This adapts the published exploration/exploitation split to observed variable costs and greedy one-loan selection. The published exploitation stage uses exploration-stage estimates to construct a density-ordered approximate knapsack mix. [Tran-Thanh et al. (2010), Section 3](https://ojs.aaai.org/index.php/AAAI/article/download/7758/7618).

## Contextual learning in Phases 2–3

For each candidate, $x_{t,i}\in\mathbb R^d$ is the borrower vector. `BorrowerFeatureEncoder.transform` supplies 25 default entries (10 numeric and Boolean features, 5 sector, 6 purpose and 4 region coordinates). It excludes the primary group, age, direct protected indicators, poverty-relief score (derived in part from marginalization and disability), and vulnerability score (derived in part from poverty relief and the protected-group count). The analysis and simulator retain these borrower attributes. `include_protected_attributes=True` restores the three removed features and five direct indicators in the existing 33-coordinate order. Model dimensions come from `n_features`, and delayed updates use the vector saved at funding. Legitimate predictors can still be statistically associated with protected attributes. See [BorrowerFeatureEncoder](Data/feature_encoder.py) and [Borrower](Data/borrower.py).

For this section let $H_t$ contain funded observations whose feedback the policy has processed before decision $t$. Use the vector from the funding decision and ridge parameter $\lambda>0$ (default $1$):

$$
A_t=\lambda I_d+\sum_{(s,i)\in H_t}x_{s,i}x_{s,i}^{\mathsf T},\qquad
b_t=\sum_{(s,i)\in H_t}r_{s,i}x_{s,i},\qquad
\hat\theta_t=A_t^{-1}b_t.
$$

`LinearRidgeModel` in [contextual.py](Algorithms/contextual.py) initializes $A=\lambda I_d,\ b=0$, adds the outer product and reward-weighted vector per feedback, and solves $A\hat\theta=b$. The positive ridge keeps $A$ positive definite for finite inputs. The uncertainty and covariance code form its inverse explicitly. LinUCB floors the computed quadratic form at zero before taking a square root to tolerate numerical roundoff. Define

$$
\mu_{t,i}=x_{t,i}^{\mathsf T}\hat\theta_t,\qquad
u_{t,i}=\mu_{t,i}+\alpha\sqrt{\max(0,x_{t,i}^{\mathsf T}A_t^{-1}x_{t,i})},\qquad
\tilde\theta_t\sim\mathcal N(\hat\theta_t,v^2A_t^{-1}).
$$

Here $\alpha\ge0$ is the confidence weight, $v>0$ is a standard-deviation scale, and the Gaussian policy makes one coefficient draw shared across all candidates per score call. Ridge predictions, optimism scores and Gaussian sample scores are uncalibrated. Only Phase 2 epsilon-greedy clips its exploitation prediction to $[0,1]$. That clipping does not establish calibration.

| Phase 2 family, in [contextual.py](Algorithms/contextual.py) | Score or selection rule | Verification |
| --- | --- | --- |
| Random, `RandomContextualPolicy` | Uniform feasible choice via independent random scores. Feedback only increments its update counter. | `test_every_combination_runs_and_repeats_identically` in [test_policy_matrix_reporting.py](tests/test_policy_matrix_reporting.py). |
| Contextual epsilon-greedy, `ContextualEpsilonGreedyPolicy` | With probability $0.10$ use random scores. Otherwise use $\operatorname{clip}(\mu_{t,i},0,1)$. Ridge-based adaptation. | `test_contextual_epsilon_greedy_clips_before_cost_normalization` |
| LinUCB, `LinUCBPolicy` | $u_{t,i}$ with experiment $\alpha=0.70$. Shared-parameter LinUCB index. | `test_ridge_updates_and_linucb_scores` |
| Contextual Thompson sampling, `ContextualThompsonSamplingPolicy` | $x_{t,i}^{\mathsf T}\tilde\theta_t$ with experiment $v=0.30$. Gaussian linear adaptation to binary feedback. | `test_contextual_thompson_uses_one_draw_with_squared_scale` |
| Budgeted versions of all three families | With `budget_aware=True` divide exploitation, UCB, or sampled scores by $c_{t,i}$. The epsilon-greedy exploration scores are independent of costs. These are current-cost adaptations. | Same three score tests above exercise budget-aware modes. |
| Hard and soft versions of all three families | Apply the restrictions or bonus below to the budgeted base policies. | `test_hard_quota_then_parity_and_soft_quota_bonus` and `test_generic_hard_fairness_runs_in_contextual_mode` |
| `true_contextual_reward_per_cost`, `TrueRewardContextualPolicy` | Configured score $p_{t,i}/c_{t,i}$. With `budget_aware=False` it scores $p_{t,i}$. Informed greedy benchmark. | `test_informed_greedy_benchmarks_choose_feasible_true_reward_per_cost` |

The focused Phase 2 score tests are in [test_policies.py](tests/test_policies.py). The policy-matrix tests exercise all registered Phase 2–3 variants. `test_contextual_feedback_delay_and_pool_renewal` in [test_environments.py](tests/test_environments.py) checks delayed release with the original vector. `lin_ucb.py` re-exports the implementations. For the original techniques see [Li et al. (2010)](https://arxiv.org/abs/1003.0146), [Chu et al. (2011)](https://proceedings.mlr.press/v15/chu11a.html) and [Agrawal and Goyal (2013)](https://proceedings.mlr.press/v28/agrawal13.html).

**Informed greedy benchmarks.** In Phases 1–2 their configured rule maximizes $p_{t,i}/c_{t,i}$ over $\mathcal F_t$ with the shared near-tie convention. It uses immediate density and has no lookahead or fairness restriction. Finite stochastic runs can yield higher realized reward for a learner. These benchmarks supply neither a global optimum nor a theoretical upper bound.

## Fairness metrics and Phase 1–2 wrappers

Let $F_g$ be a group’s funded-loan count, $L=\sum_g F_g$ the total funded loans, and $q_g$ its configured minimum share. The total includes all funded groups and counts each loan in a Phase 3 batch. Missing counts are zero. Let $G_Q=\{g:q_g>0\}$.

$$
a_g=\begin{cases}F_g/L&L>0\\0&L=0,\end{cases}\qquad
D_g=[q_g-a_g]_+,\qquad Q=\sum_{g\in G_Q}D_g,\qquad
\mathrm{QSR}=\frac{\#\{g\in G_Q:D_g\le10^{-12}\}}{|G_Q|}.
$$

Here $[z]_+=\max(0,z)$. With zero loans, each positive target contributes its full deficit. With no positive targets, $Q=0$ and $\mathrm{QSR}=1$ by convention. The $10^{-12}$ numerical tolerance is for individual quota reporting. The episode threshold $\tau_q=0.01$ applies to the sum $Q$, with no requirement that each deficit be within numerical tolerance.

**Phase 1 allocation-share parity.** The gap is $\max_g a_g-\min_g a_g$ across all configured arm groups, including groups funded zero times. With zero loans, the gap is zero by convention and parity is unevaluable. Applicant-level parity comparisons use a different denominator and a different population.

**Phases 2–3 applicant parity.** For $d\in\{\mathrm{population},\mathrm{feasible}\}$, set $G_d=\{g:O_{d,g}>0\}$ and $f_{d,g}=F_g/O_{d,g}$ for $g\in G_d$. The gap is $\max_{g\in G_d} f_{d,g}-\min_{g\in G_d} f_{d,g}$ when $|G_d|\ge2$ and undefined otherwise. Groups without opportunities have blank funding rates. Population opportunities include every eligible arrival before filtering. Feasible opportunities include applicants affordable at arrival, after maturities and reserve checks in Phase 3 and before any lending in the batch. The pool is counted once even if scored repeatedly. This ensures $0\le F_g\le O_{\mathrm{feasible},g}\le O_{\mathrm{population},g}$. Feasible opportunities depend on prior allocations. Population opportunities cover only pools generated in the observed episode, including a generated pool that triggers termination before a decision. Trajectory rows cover decision rounds and omit such a terminal pool. Episode summaries include it.

Both opportunity and loan counts use `BorrowerCandidate.fairness_group`, which delegates to `Borrower.primary_fairness_group`. The priority `disabled → reserved → minority → women → aged → none` partitions applicants into mutually exclusive, first-match categories. A woman with a disability contributes only to `disabled`. The categories represent that partition of the arriving applicants.

| Formulation | Implementation | Output fields |
| --- | --- | --- |
| Allocation share and spread | `allocation_shares`, `allocation_share_gap` in [metrics.py](metrics.py) | `allocation_share_G`, `allocation_share_gap` (Phase 1) |
| Population and feasible funding rates and gaps | `group_funding_rates`, `demographic_parity_gap` in [metrics.py](metrics.py), `_fairness_metrics` in [evaluate.py](evaluate.py) | `population_funding_rate_G`, `funding_rate_G`, `population_dp_gap`, `feasible_dp_gap`, `population_dp_groups`, `feasible_dp_groups` |
| Funded, feasible and population counts | `BanditEnvironment._record_current_opportunities/step`, `StatefulMicrofinanceCMDPEnv._record_pool_opportunities/fund` | `funded_G`, `opportunities_G`, `population_opportunities_G`. Phase 1's `opportunities_G` counts decisions with an affordable arm in the group. |
| Individual quota deficit, total deficit and satisfied fraction | `quota_deficits`, `total_quota_deficit`, `quota_satisfaction_rate` in [metrics.py](metrics.py) | `quota_deficit_G`, `total_quota_deficit`, `quota_satisfaction_rate` |

The legacy `demographic_parity_gap` field mirrors `allocation_share_gap` in Phase 1 and `feasible_dp_gap` in Phases 2–3. The combined report uses the explicitly named gap columns.

- **Hard wrapper.** `HardFairPolicy.select_action` sorts projected quota shortfalls $q_g(L+1)-F_g>10^{-12}$ descending, with ties in target-map order. It restricts the selector to the first group with an eligible action. If none is available and policy parity exceeds $\tau_p$, it restricts to the lowest-rate groups with eligible actions (rate ties by `isclose`). Otherwise it delegates with the original feasible set. Phase 1 retains its original intervention rate, funded loans divided by affordable group-rounds with zero for unavailable groups. The corresponding report evaluates allocation-share parity with the existing $0.35$ tolerance. Phase 2 uses feasible-opportunity parity at $0.010$. The override is a priority rule without a terminal or anytime guarantee.
- **Soft wrapper.** `SoftFairPolicy.scores` adds $\beta_{\mathrm{f}}D_{g_{t,i}}$ to finite base scores. The Phase 1 Budgeted UCB score is the optimistic reward divided by current cost and $\beta_{\mathrm{f}}=2$. Phase 2 uses optimistic LinUCB reward, sampled Thompson reward, or clipped epsilon-greedy exploitation reward divided by current cost, with $\beta_{\mathrm{f}}=0.015$. Epsilon-greedy exploration uses unitless uniform scores in $[0,1)$ before the same bonus. The deficit is a share, and the coefficient converts it into a score increment on the base policy's numeric scale. Infinite scores stay unchanged. A fixed coefficient has different practical effects across base policies, costs and exploration branches. Findings apply to the tested settings.

## Phase 3 CMDP

### Full state and observable quantities

Before the round's funding decisions, a sufficient environment state includes round index, available capital, the current applicant pool (including latent repayment parameters), outstanding records (principal, funded and due rounds, latent outcomes and saved learning contexts), funding and opportunity counts, and repayment and cumulative accounting history needed for tracking and reporting. Within a batch it also includes the runner's number of slots already used. The seeded implementation shares a random generator across borrower generation and outcomes. To reproduce its transitions exactly, include that generator's state and the sequential applicant counter. Policy-specific ridge and multiplier states come from the observations processed by the policy.

The policy-facing information includes contexts, requested principals, primary groups, feasibility and group counts, current cash and time, capital at risk, pending loan counts or scheduled principal, and observed repayment and loss summaries. The compact portfolio quantities and the discretized buckets in `state_bucket` do not on their own determine transitions, including which principal and outcomes mature when. Policies score the current information available to them. Loan outcomes stay hidden from learners until maturity. See `observation`, `fund`, `advance_round` and `state_bucket` in [cmdp_environment.py](Environments/cmdp_environment.py).

### Batches, timing and capital

Let $B_t$ be cash at the start of lending in round $t$ **after** releases due then, $\ell$ the configured reserve, $c_{\mathrm{floor}}$ the spendable-cash floor, and $K$ the batch limit. For each slot the runner intersects the remaining pool with $c_{t,i}\le B_{\mathrm{now}}-\ell+10^{-12}$, using the cash available at that slot. The feasible set is empty if $B_{\mathrm{now}}-\ell<c_{\mathrm{floor}}-10^{-12}$. `fund` removes the chosen applicant, subtracts principal from cash, increments its group count and records a pending loan due at $t+d_{\mathrm{rep}}$ for configured delay $d_{\mathrm{rep}}\ge1$. Its stored learning vector is a copy of the one supplied at funding. Recompute feasibility and scores for the next slot, stopping when $K$ loans have been funded or the selector returns `None`. Duplicate funding within the batch is prevented by removal. The runner enforces $K$. The environment enforces single-loan feasibility. A batch may be empty, with no explicit policy action to reject an affordable candidate based on its score.

The resulting batch $\mathcal A_t$ has total principal $C_t^{\mathrm{batch}}=\sum_{i\in\mathcal A_t}c_{t,i}$. Let $\mathcal M_{t+1}$ be the outstanding loans due by the next round advance. For loan $j$ with principal $c_j$ and repayment $r_j$, denote the recycle and default recovery fractions by $\rho_{\mathrm{rep}}$ and $\rho_{\mathrm{def}}$. The advance increments the round, releases due loans, and then generates the next pool below the horizon:

$$
\mathcal R_{t+1}=\sum_{j\in\mathcal M_{t+1}}
c_j\bigl(\rho_{\mathrm{rep}}r_j+\rho_{\mathrm{def}}(1-r_j)\bigr),\qquad
B_{t+1}=B_t-C_t^{\mathrm{batch}}+\mathcal R_{t+1},\qquad B_0=B_{\mathrm{init}}.
$$

Matured loans leave capital at risk, which is the sum of outstanding principal. Each matured default adds $c_j(1-\rho_{\mathrm{def}})$ to cumulative default loss. The unrecycled portion of repaid principal reduces surviving capital without entering the default-loss counter. The runner logs post-funding round values before the advance, updates the fairness multipliers, and delivers matured feedback before the next lending decision if one exists. The run ends at the configured horizon or upon an advance with no pending loans and spendable cash below $c_{\mathrm{floor}}$ (also checked at reset). Terminal advance still releases maturities even when there is no ensuing learner update.

### Objective and constraints

In Phase 3, $T$ counts all lending rounds through termination, including rounds with an empty batch. The eventual and observed repayment counts are

$$
R_{\mathrm{eventual}}=\sum_{t=0}^{T-1}\sum_{i\in\mathcal A_t}r_{t,i},\qquad
R_{\mathrm{observed}}=\sum_{t=0}^{T-1}\sum_{i\in\mathcal A_t}
\mathbf 1\{t+d_{\mathrm{rep}}\le T\}\,r_{t,i}.
$$

The reporting cutoff for observed reward is the advance into $T$. The final logged lending row precedes that advance. Outstanding outcomes contribute to eventual reward. The evaluator's `eventual_terminal_capital` specifically adds the realized future returns of pending loans to terminal cash.

One finite-horizon research formulation is to maximize $\mathbb E_\pi[R_{\mathrm{eventual}}]$ over policies obeying the pathwise slot and capital constraints above and subject to terminal fairness targets. Counts in the full state give terminal quota and parity outcomes. The default evaluation uses $\tau_q=0.01$ and $\tau_p=0.05$. The implemented score-based interventions are heuristics.

Sources are `StatefulMicrofinanceCMDPEnv` in [cmdp_environment.py](Environments/cmdp_environment.py), `CMDPEnvironmentConfig` in [environment_config.py](models/environment_config.py), `run_cmdp_episode` and `rank_results` in [evaluate.py](evaluate.py), and `_add_sweep_metrics` in [robustness.py](robustness.py). `test_cmdp_reuses_parent_generator_and_recycles_at_maturity`, `test_cmdp_terminal_advance_observes_maturities_and_preserves_pending_reward` and `test_cmdp_episode_never_overspends_liquid_capital` in [test_cmdp.py](tests/test_cmdp.py) check context, timing and cash accounting.

### Portfolio-aware scores

Use current pool median $m_t=\max(1,\operatorname{median}_{i\in\mathrm{pool}} c_{t,i})$ and $z_{t,i}=c_{t,i}/m_t$. The pool includes currently unaffordable candidates and shrinks after each funding. Let $B$ be the cash at the current slot, $P_2$ outstanding principal due within two rounds (regardless of future repayment), and $L_{\mathrm{def}}$ the accumulated observed default loss. With $h_0$ the configured base shadow price and $w_B,w_P,w_D$ the liquidity weight, due-soon relief and default-loss weight,

$$
\begin{aligned}
q_B&=\operatorname{clip}(B/B_{\mathrm{init}},0,1.5),&
q_P&=\operatorname{clip}(P_2/B_{\mathrm{init}},0,1),&
q_D&=\operatorname{clip}(L_{\mathrm{def}}/B_{\mathrm{init}},0,1),\\
h_{\mathrm{full}}&=h_0\frac{1+w_B[1-q_B]_++w_Dq_D}{1+w_Pq_P},&
h_{\mathrm{liq}}&=h_0(1+1.4[1-B/B_{\mathrm{init}}]_+).
\end{aligned}
$$

Factory values are $h_0=0.22$, $w_B=1.40$, $w_P=0.80$, $w_D=0.60$. Lower cash and higher default losses weakly increase the full penalty. More principal due soon weakly decreases it. There is no additional clipping of $h_{\mathrm{full}}$ itself. The inputs imply bounds $h_0/(1+w_P)\le h_{\mathrm{full}}\le h_0(1+w_B+w_D)$. For nonnegative cash, $h_{\mathrm{liq}}$ lies between $h_0$ and $2.4h_0$ and uses no loss or maturity adjustment. For a cash value just below zero within the feasibility tolerance, the simpler formula still uses the literal ratio. Higher $h$ reduces expensive applicants' scores more because it multiplies $z_{t,i}$. Empty pools return empty scores. A median floor of one cost unit guards normalization, and the ratio-based variants floor $z_{t,i}$ at $10^{-6}$.

| Policy in [cmdp.py](Algorithms/cmdp.py) | Implemented score | Verification in [test_cmdp.py](tests/test_cmdp.py) |
| --- | --- | --- |
| `RandomCMDPPolicy` | Independent uniform scores. Its update count is bookkeeping only. | `test_cmdp_terminal_advance_observes_maturities_and_preserves_pending_reward` |
| `ContextOnlyBudgetedLinUCBPolicy` | $u_{t,i}/\max(10^{-6},z_{t,i})$, with $\alpha=0.70$. Context and cost control. | `test_portfolio_linucb_epsilon_greedy_and_context_only_scores` |
| `PortfolioAwareLinUCBPolicy` | $u_{t,i}-h_{\mathrm{full}}z_{t,i}$, with $\alpha=0.70$. Portfolio heuristic. | `test_portfolio_linucb_epsilon_greedy_and_context_only_scores` |
| `PortfolioAwareEpsilonGreedyPolicy` | Random scores with $\epsilon=0.10$. Otherwise $\mu_{t,i}-h_{\mathrm{full}}z_{t,i}$, with no clipping of $\mu_{t,i}$. Portfolio heuristic. | `test_portfolio_linucb_epsilon_greedy_and_context_only_scores` |
| `PortfolioAwareThompsonPolicy` | $x_{t,i}^{\mathsf T}\tilde\theta_t-h_{\mathrm{liq}}z_{t,i}$, with $v=0.30$. Portfolio heuristic. | `test_portfolio_thompson_scores_and_informed_heuristic_configuration` |
| Informed Portfolio-Aware Heuristic, `OraclePortfolioPolicy` (ID `oracle_portfolio`) | Configured `state_aware=True` gives $p_{t,i}-h_{\mathrm{liq}}z_{t,i}$ with no score clipping. The optional `False` branch gives $p_{t,i}/\max(10^{-6},z_{t,i})$. | `test_portfolio_thompson_scores_and_informed_heuristic_configuration` |
| `HardFairCMDPPolicy` | Quota and parity restrictions around each of the three portfolio learners. | `test_hard_cmdp_fairness_restricts_to_affordable_candidates` |
| `LagrangianFairCMDPPolicy` | Normalized scores minus adaptive projected fairness penalties around the same three learners. | `test_lagrangian_scores_normalize_and_use_projected_shortfalls`, `test_lagrangian_updates_use_signed_excess_and_project_both_bounds` |

The portfolio learners update the ridge using funded contexts returned at maturity. Portfolio state enters selection through the cost term and fairness wrappers. The shadow prices are heuristic penalties with no established optimal dual-variable interpretation.

### Hard and adaptive fairness in Phase 3

`HardFairCMDPPolicy.select_action` uses the same priority order as the Phase 2 wrapper and re-evaluates feasible candidates per slot. The shared funding-rate calculation excludes groups with zero feasible opportunities. At zero hard fairness strength in the sweeps, only the quota override is disabled.

For the adaptive wrapper, let $b_i^{\mathrm{base}}$ be the base policy's raw score, with minimum $b_{\min}$ and maximum $b_{\max}$ over the current feasible set. Min–max rescale feasible scores to $\bar b_i=(b_i^{\mathrm{base}}-b_{\min})/(b_{\max}-b_{\min})$. If the extrema are close by `isclose`, all feasible values become $0.5$. Infeasible entries stay zero and are excluded from selection. For each possible action $i$, compute $Q^{+i}$ and $G^{+i}$ from hypothetical counts $F_g+\mathbf 1\{g=g_{t,i}\}$ and unchanged opportunity counts. With fairness strength $\sigma$ and multipliers $\lambda_q,\lambda_p$,

$$
s_i=\begin{cases}
\bar b_i & t<w,\\
\bar b_i-\sigma\bigl(\lambda_q Q^{+i}+\lambda_p [G^{+i}-\tau_p]_+\bigr)&t\ge w,
\end{cases}
$$

where $w$ is the warm-up duration. The action score uses raw projected quota deficit and positive projected feasible-parity excess. An undefined projected gap contributes zero excess. At round end, once per round including empty batches, use the actual running counts after selection. For $t\ge w$ the signed updates are

$$
\begin{aligned}
\lambda_q &\gets \operatorname{clip}\bigl(\lambda_q+\eta(Q-\tau_q),0,\lambda_{\max}\bigr),\\
\lambda_p &\gets \operatorname{clip}\bigl(\lambda_p+\eta(G-\tau_p),0,\lambda_{\max}\bigr).
\end{aligned}
$$

Defaults in [LagrangianConfig](models/fairness_config.py) and the runner are $\lambda_q=\lambda_p=0.20$ initially, learning rate $\eta=0.20$, cap $\lambda_{\max}=8$, $w=5$ and $\sigma=1$. The first five indexed rounds use normalization without a penalty or multiplier updates. An undefined feasible gap leaves $\lambda_p$ unchanged. Defined inputs raise weights above the thresholds and lower them below, subject to clipping. $\sigma=0$ removes the score penalties while normalization, wrapper tie-breaking and multiplier updates still apply. The normalized score and penalty are dimensionless. The runner exports pre-update `lambda_quota` and `lambda_parity` in trajectories and their post-update `_final` values in episode summaries. `LagrangianFairCMDPPolicy.scores/on_round_end` and `StatefulMicrofinanceCMDPEnv.projected_fairness` implement this adaptive heuristic, with no convergence, optimality or constraint-satisfaction guarantee.

### Episode compliance and opportunity screens

`episode_constraint_outcomes` in [evaluate.py](evaluate.py) checks $G_T\le\tau_p$, $Q_T\le\tau_q$, and their conjunction. It exports `parity_satisfied`, `quota_satisfied` and `constraint_satisfied`. The `population_` parity and joint fields use the same thresholds with the population gap. An undefined gap yields blank parity and joint outcomes. The corresponding aggregate `parity_satisfaction_rate`, `quota_constraint_satisfaction_rate` and `constraint_satisfaction_rate` divide the number of passes by all evaluated episodes. The `episodes_evaluated` and `*_evaluable_episodes` counts show coverage. Gap means use defined gaps, while the Phase 1 zero-loan gap keeps its documented zero convention. `rank_results` ranks Phases 2–3 by feasible joint satisfaction first, then summed mean threshold excess for incomplete compliance, followed by reward efficiency, cumulative reward and policy identifier.

The screens in [robustness.py](robustness.py) fix loan volume at the observed $L$ and use affordable-arrival counts $O_{\mathrm{feasible},g}$. `_quota_opportunity_feasible` computes the minimum attainable deficit at that volume by using available whole quota units first, then the largest available fractional units within the remaining loans. At $L=0$, it checks the full sum of targets. `_parity_opportunity_feasible` tests candidate minimum rates $r=k/O_{\mathrm{feasible},g}$ and checks that each group has a nonempty integer interval for rates in $[r,r+\tau_p]$ and that $L$ falls between the sums of the lower and upper bounds. It excludes zero denominators and returns an undefined result with fewer than two valid groups. The screens export `quota_feasible`, `parity_feasible` and `opportunity_screens_passed` (legacy alias `fairness_feasible`). `feasibility_status` distinguishes a failed screen, an undefined parity comparison with no ruled-out quota, observed compliance, and an observed violation after both screens passed. Passing both screens leaves joint feasibility unresolved. The count bounds omit heterogeneous costs, capital constraints, batch limits, repayment delays and arrival timing. Sweep `fairness_violations` counts logged rounds with a known parity or quota threshold breach.

The theoretical results for stationary UCB1, fixed-cost budgeted bandits, and linear payoff models require assumptions beyond these simulations. They do not transfer automatically to cost resampling, feasible-action filtering, delayed feedback, nonlinear synthetic reward generation or fairness interventions. [Auer, Cesa-Bianchi and Fischer (2002)](https://www2.compute.dtu.dk/pubdb/pubs/2088-full.html).
