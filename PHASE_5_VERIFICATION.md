# Phase 5 verification

Reviewed branch `fix/phase5.1` at base commit `5b1b243a38601a017b627bf8538040e4f09516a4`. Phase 5 reconciles the mathematical and research documentation with the simulator and the corrected Phase 4 outputs. The detailed numerical claims, parameters, seeds and file references are in [Research findings](RESEARCH_FINDINGS.md).

## Prior evidence reviewed

| Correction phase | Evidence |
| --- | --- |
| Phase 1 | Branch checkpoint `6cdf4c7`, the documented bandit score derivations and focused score tests. Reviewed all registered arm policies, rounded-cost KUBE, feasible exploration, and Budgeted UCB/Fractional KUBE implementation equivalence. |
| Phase 2 | Branch checkpoint `41b95ce`, fairness reporting and sweep baseline checks. Reviewed the wrapper priorities and bonuses, borrower grouping and feature encoding, eligible and population counts, quota tolerances, episode outcomes and opportunity screens. |
| Phase 3 | Branch checkpoint `b584a8b`, feedback and accounting changes, CMDP tests and simulation correctness tests. Reviewed the opening pool, observability, batch feasibility, draw streams, maturity, recovery, capital conservation, portfolio scores and multiplier timing. |
| Phase 4 | [Archived report](outputs/phase4.1/phase4_report.md), [suite manifest](outputs/phase4.1/run_manifest.json), [saved verification summary](outputs/phase4.1/verification_summary.json), and the default and scenario raw tables, aggregates, rankings, paired comparisons and sweeps. Also reviewed the verifier changes at `5b1b243` and the local Phase 4.2 smoke audits for the three phases and sweeps. Smoke settings were used for workflow validation only. |

The Phase 4 report records 34,589,515 successful checks across 66 experiments and 25,740 episodes. It also records a separate 10,304-check suite-table audit and exact replay results for the default Phase 1–2 outputs after the zero-penalty wrapper change. The final source adds shared logged-offer and funded-outcome checks and scenario-manifest validation. The archive's execution base is `b584a8b` with `git_dirty: true`. That marker records the base and working-tree status and leaves the exact build unidentified.

## Discrepancies and corrections

| Area | Correction |
| --- | --- |
| Experiment generation | Removed historical README result tables and worked examples and replaced them with observations from the corrected runs. Historical parity and ranking fields come from different definitions, and numerical seeds across generations do not identify the same underlying applicants. |
| Decision and capital model | Specified available information, no-loan transitions, ordered slot-by-slot feasibility with immediate recycling, and the horizon observation cutoff. Removed the description of a feasibility tolerance for negative Phase 3 cash. Distinguished the full environment state, the policy observation, and the formal constrained objective. |
| Fairness | Distinguished expected targets, zero-opportunity handling, realised terminal passes and reported pass fractions. Clarified the effects of protected-group priority, correlated features, phase-specific denominators, soft bonus scale and the two meanings of sweep strength. |
| Evidence and ranking | Separated successful repayment count, repayment rate, reward per principal, budget use and capital turnover. Restricted claims by phase, policy, configuration and outcome. Recorded quota and parity failures, benchmark exceptions and the absence of matched context and portfolio-state ablations. |
| Report wording | Replaced winner language with configuration ranking and explained the two informed scoring rules. Corrected the imbalance summary to list the generator probabilities with all-seed compliance. A sweep with zero baseline reward reports an undefined relative change. |

The observed simulator and policy logic agreed with the checked formulations after documentation corrections. The existing code implements greedy or exploratory policies with heuristic fairness intervention. Its scopes of evidence include finite seeded comparisons and retrospective ledger checks. Regret, convergence, global optimality, guaranteed fairness and external effectiveness lack support in the experiment.

## Files changed

- [README.md](README.md), [MATHEMATICAL_FORMULATION.md](MATHEMATICAL_FORMULATION.md), and [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) describe the corrected study, formulation and evidence.
- [evaluate.py](evaluate.py) and [main.py](main.py) update the text shown in phase summaries, tables and the suite report. The policy IDs and `winner_summary.txt` filename remain available.
- [robustness.py](robustness.py) corrects generated descriptions and how an undefined percentage change is displayed. [tests/test_mode5.py](tests/test_mode5.py) covers nonmonotone compliance settings and zero reward baselines.
- This verification record captures the review and the evidence boundaries.

The executable changes affect reporting text. Simulation, policy choices, seeds, scenarios, numerical tables, and ranking calculations retain their existing behavior. The Phase 4 results remain applicable to the reviewed implementation and the empirical claims refer to the saved evidence. Archived report text was left as generated. An original dissertation file was absent from the tracked repository. The review covered the supplied hypotheses and the available historical comparisons.

## Checks executed

| Check | Result |
| --- | --- |
| Baseline `python -m pytest -q` on the inspected commit | 121 passed. The available base interpreter used pandas 1.4.4 and emitted NumPy deprecation warnings. |
| Final `python -m pytest -q` using Python 3.10.19, NumPy 1.26.4, pandas 2.3.3, Matplotlib 3.10.8, SciPy 1.15.3 and pytest 8.4.2 | 124 passed in 61.51 seconds. |
| Saved suite manifests, episode keys, aggregate statistics, paired tables, ranking and sweep summaries | 1,382,014 passed checks with zero failures across 25,740 episodes and 66 experiments using the existing verifier functions. The separate suite-manifest and combined-table check is included in this count (14,054 passed). |
| Current verifier ledger replay for the default phases and their sweeps | 3,323,611 passed checks with zero failures across 1,950 episodes and six experiments, including the standalone Phase 3 sweep baseline comparisons. |
| Current verifier ledger replay for Phase 3 scenarios 2 and 10 | 378,435 and 232,461 checks passed respectively, with zero failures across 144 episodes per scenario. These runs cover the all-seed compliance case and the exception to the informed heuristic's reward lead. |
| Full archive ledger replay | The saved Phase 4 replay passed 34,589,515 checks across 25,740 episodes. Phase 5 interrupted its duplicate full ledger replay after completing the current all-archive table checks and targeted ledger replays. The interrupted command produced no final result. |
| Report wording on the corrected default sweep tables | Confirmed the half-capital and zero-strength relative changes and confirmed that sole all-seed imbalance setting is reported as 40% (one of five). |
| `git diff --check` and local documentation links | Passed. Local file and section references resolve in this workspace. |

Phase 5 uses the existing verifier and saved trials. The verifier checks recorded funding and maturity, available cash, opportunity pools, defined episode outcomes, statistical tables, matched-seed comparisons, rank order and sweep baselines. Its logging limits matter even when all recorded fields agree. The archive stores neither complete unselected applicant feature vectors nor unfunded outcomes. It cannot reconstruct all policy decisions and potential outcomes from those files.

## Preprint readiness

The mathematical and empirical claims are ready to use within the documented scope. Reproducibility of the logged analysis and full run provenance are separate. The full corrected archive is local and Git-ignored, and its manifest records a dirty source build. Exact source provenance and independent end-to-end reproduction from a clean source release remain open for the preprint's reproducibility materials. The scientific scope is synthetic, with descriptive operating comparisons and no matched identification of the independent effects of context or portfolio state.

- **Completed.** Formulation review, claim and terminology corrections, hypothesis assessment, and corrected-result traceability.
- **Partially complete.** The local manifests and artifacts establish partial provenance for a reproducibility release. The complete execution source remains unidentified.
- **Unverified by the archived data.** Full feature and counterfactual streams, isolated context and portfolio-state effects, and real-world lending or social outcomes.
