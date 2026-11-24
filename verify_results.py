"""Independently replay saved lending ledgers and check experimental tables."""

from argparse import ArgumentParser
from collections import Counter, defaultdict, deque
from functools import cache
import json
from math import ceil, floor, fsum, isfinite, sqrt
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

# Shared names describe the CSV schema. All expected values are computed here.
from episode_statistics import FAIRNESS_COMPARISONS
from evaluate import CMDP_METRICS, COMMON_METRICS, CONSTRAINT_RATE_FIELDS, FAIRNESS_COUNT_PREFIXES
from robustness import AGGREGATE_METRICS, SCREEN_METRICS

ABS_TOLERANCE = 1e-8
REL_TOLERANCE = 1e-12
EPISODE_KEY = ["policy", "seed"]
CASE_KEY = ["sweep_value", "parameter_kind"]


class Audit:
    def __init__(self):
        self.experiments = self.episodes = self.passed = self.failure_count = 0
        self.metrics = set()
        self.failures = []
        self.unverified = set()

    def require(self, condition, location, metric, detail=""):
        self.metrics.add(str(metric))
        if condition:
            self.passed += 1
        else:
            self.failure_count += 1
            if len(self.failures) < 100:
                self.failures.append(f"{location} | {metric} | {detail}")

    def compare(self, location, metric, actual, expected, labels=None):
        actual = np.atleast_1d(actual)
        expected = np.atleast_1d(expected)
        if actual.shape != expected.shape:
            self.require(False, location, metric, f"shape {actual.shape}, expected {expected.shape}")
            return
        if expected.dtype.kind in "ifub":
            if actual.dtype.kind in "OUS":
                actual = np.array([{"True": 1, "False": 0}.get(v, v) for v in actual], dtype=float)
            if expected.dtype.kind in "iub":
                equal = actual == expected
            else:
                equal = np.isclose(actual, expected, atol=ABS_TOLERANCE,
                                   rtol=REL_TOLERANCE, equal_nan=True)
        else:
            equal = actual == expected
        if equal.all():
            self.require(True, location, metric)
        else:
            index = np.flatnonzero(~equal)[0]
            label = f" row={labels[index]}" if labels is not None else ""
            self.require(False, f"{location}{label}", metric,
                         f"got {actual[index]!r}, expected {expected[index]!r}")

    def fields(self, location, actual, expected, labels=None):
        for name, value in expected.items():
            if name not in actual:
                self.require(False, location, name, "missing required column")
            else:
                self.compare(location, name, actual[name], value, labels)

    def report(self):
        return {
            "experiments_checked": self.experiments, "episodes_checked": self.episodes,
            "metrics_checked": len(self.metrics), "passed_checks": self.passed,
            "failed_checks": self.failure_count, "discrepancies": self.failures,
            "could_not_independently_verify": sorted(self.unverified),
        }


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""],
                       dtype={"identifier": str, "applicant_id": str, "fairness_group": str}, **kwargs)


def json_file(path):
    return json.loads(path.read_text())


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def fairness(funded, affordable, population, targets, parity_tolerance, quota_tolerance):
    total = sum(funded.values())
    groups = set(funded) | set(affordable) | set(targets) | set(population or ())
    shares = {group: ratio(funded[group], total) for group in groups}
    deficits = {g: max(0, target - shares[g]) for g, target in targets.items()}
    deficit = sum(deficits.values())
    relevant = [deficits[g] for g, target in targets.items() if target > 0]
    met = deficit <= quota_tolerance
    result = {
        "total_quota_deficit": deficit, "quota_satisfied": met,
        "quota_satisfaction_rate": (
            sum(d <= 1e-12 for d in relevant) / len(relevant) if relevant else 1.0
        ),
    }

    def outcomes(prefix, gap):
        known = isfinite(gap)
        parity_pass = gap <= parity_tolerance
        return {
            f"{prefix}parity_evaluable": known,
            f"{prefix}parity_satisfied": int(parity_pass) if known else np.nan,
            f"{prefix}constraint_satisfied": int(parity_pass and met) if known else np.nan,
        }

    if population is None:
        group_shares = [shares[g] for g in funded]
        gap = max(group_shares, default=0) - min(group_shares, default=0)
        result["allocation_share_gap"] = gap
        result.update(outcomes("", gap if total else np.nan))
    else:
        for prefix, opportunities in (("feasible", affordable), ("population", population)):
            rates = [funded[g] / n for g, n in opportunities.items() if n]
            gap = max(rates) - min(rates) if len(rates) >= 2 else np.nan
            result[f"{prefix}_dp_gap"] = gap
            result[f"{prefix}_dp_groups"] = len(rates)
            result.update(outcomes("" if prefix == "feasible" else "population_", gap))
        gap = result["feasible_dp_gap"]
    result["demographic_parity_gap"] = gap
    for g in sorted(groups):
        result[f"funded_{g}"] = funded[g]
        result[f"opportunities_{g}"] = affordable[g]
        result[f"allocation_share_{g}"] = shares[g]
        result[f"quota_deficit_{g}"] = deficits.get(g, 0.0)
        result[f"funding_rate_{g}"] = (
            funded[g] / affordable[g] if affordable[g] else (0.0 if population is None else np.nan)
        )
        if population is not None:
            result[f"population_opportunities_{g}"] = population[g]
            result[f"population_funding_rate_{g}"] = funded[g] / population[g] if population[g] else np.nan
    return result


def replay_episode(audit, location, mode, config, thresholds, summary, trajectory, funding, maturity, policy_settings):
    """Reconstruct cash, due events, and group counts from the opening pools and loans."""
    cmdp, arm = mode == "cmdp", mode == "arm"
    horizon = config["max_rounds"]
    initial = config["initial_capital" if cmdp else "total_budget"]
    reserve = config.get("minimum_liquidity_reserve", 0)
    delay = config.get("repayment_delay_rounds", config.get("feedback_delay_rounds", 0))
    recycle, recovery = config.get("recycle_fraction", 0), config.get("default_recovery_fraction", 0)
    targets = config["target_shares"]
    tolerances = (thresholds["parity_tolerance"], thresholds["quota_tolerance"])

    dual = (policy_settings["parameters"]["config"]
            if policy_settings["class"] == "LagrangianFairCMDPPolicy" else None)
    quota_multiplier = dual["initial_lambda_quota"] if dual else 0.0
    parity_multiplier = dual["initial_lambda_parity"] if dual else 0.0
    cash = initial
    spending = returns = losses = unrecycled = at_risk = 0.0
    observed = eventual = funded_total = matured_total = 0
    pending = defaultdict(list)
    history = deque(maxlen=30)
    expected_maturity, expected_rounds = [], []
    population = None if arm else Counter()
    arm_specs = config.get("arms", [])
    arm_groups = {spec["group"] for spec in arm_specs}
    funded = Counter(dict.fromkeys(arm_groups, 0))
    affordable = funded.copy()
    seen_applicants = set()
    funding_by_round = {
        r: frame for r, frame in funding.groupby("round", sort=False)
    } if not funding.empty else {}
    if cmdp and not funding.empty:
        audit.require(funding["round"].is_monotonic_increasing, location, "funding_order")
    if not trajectory.empty:
        audit.compare(location, "round", trajectory["round"], np.arange(len(trajectory)))
    audit.require(len(trajectory) <= horizon and (arm or len(trajectory) == horizon),
                  location, "decision_horizon", f"{len(trajectory)} rounds, horizon {horizon}")

    def settle(round_number):
        nonlocal cash, returns, observed, matured_total, losses, unrecycled, at_risk
        loans = pending.pop(round_number, ())
        for identifier, group, principal, reward in loans:
            observed += reward
            matured_total += 1
            history.append(reward)
            if cmdp:
                returned = principal * (recycle if reward else recovery)
                cash += returned
                returns += returned
                at_risk -= principal
                unrecycled += principal - returned if reward else 0
                losses += principal - returned if not reward else 0
                expected_maturity.append({
                    "round": round_number, "applicant_id": identifier, "fairness_group": group,
                    "reward": reward, "principal": principal, "capital_returned": returned,
                    "available_capital_after": cash,
                })
        return len(loans)

    for r, row in enumerate(trajectory.itertuples(index=False)):
        matured = settle(r)
        pool = json.loads(row.opportunity_pool)
        pool_ids = {opportunity[0]: tuple(opportunity[1:]) for opportunity in pool}
        opening_cost_limit = cash - reserve
        valid_pool = all(isfinite(cost) and cost > 0 for _, _, cost in pool)
        if arm:
            valid_pool &= [(identifier, group) for identifier, group, _ in pool] == [
                (spec["name"], spec["group"]) for spec in arm_specs
            ]
            affordable.update({group for _, group, principal in pool if principal <= opening_cost_limit})
        else:
            valid_pool &= len(pool) == len(pool_ids) == config["pool_size"]
            valid_pool &= seen_applicants.isdisjoint(pool_ids)
            seen_applicants.update(pool_ids)
            population.update(group for _, group, _ in pool)
            affordable.update(group for _, group, principal in pool if principal <= opening_cost_limit)
        audit.require(valid_pool, f"{location} round={r}", "opening_opportunity_pool")

        if cmdp:
            batch = funding_by_round.pop(r, None)
            loans = [(loan.applicant_id, loan.fairness_group, loan.principal, loan.hidden_reward,
                      loan.due_round, loan.available_capital_after, loan.true_repayment_probability)
                     for loan in batch.itertuples()] if batch is not None else []
            audit.require(len(loans) <= config["max_loans_per_round"],
                          f"{location} round={r}", "batch_limit")
        elif pd.notna(row.selected_index):
            index = int(row.selected_index)
            valid = row.selected_index == index and 0 <= index < len(pool)
            audit.require(valid and pool[index][0] == row.identifier and
                          tuple(pool[index][1:]) == (row.fairness_group, row.cost),
                          f"{location} round={r}", "selected_opportunity")
            if arm and valid:
                audit.compare(f"{location} round={r}", "arm.true_repayment_probability",
                              row.true_repayment_probability, arm_specs[index]["repayment_probability"])
            loans = [(row.identifier, row.fairness_group, row.cost, row.reward, r + delay, None,
                      row.true_repayment_probability)]
        else:
            audit.require(row.reward == row.cost == 0 and pd.isna(row.identifier) and
                          all(cost > opening_cost_limit for _, _, cost in pool),
                          f"{location} round={r}", "skipped_round")
            loans = []

        for identifier, group, principal, reward, due, cash_after, probability in loans:
            valid = reward in (0, 1) and isfinite(principal) and 0 < principal <= cash - reserve
            valid &= 0 <= probability <= 1 and (0 < probability < 1 or probability == reward)
            if cmdp:
                valid &= pool_ids.pop(identifier, None) == (group, principal)
                valid &= due == r + delay
            audit.require(valid, f"{location} round={r} loan={identifier}", "funding_feasibility")
            spending += principal
            cash -= principal
            funded[group] += 1
            funded_total += 1
            eventual += int(reward)
            pending[r + delay].append((identifier, group, principal, int(reward)))
            if cmdp:
                audit.compare(f"{location} round={r} loan={identifier}",
                              "funding.available_capital_after", cash_after, cash)
                at_risk += principal
            if delay == 0:
                matured += settle(r)

        fair = fairness(funded, affordable, population, targets, *tolerances)
        expected = {"observed_cumulative_reward": observed, **fair}
        if cmdp:
            recent_rate = sum(history) / len(history) if history else 0.5
            gap = fair["feasible_dp_gap"]
            shortfall = fair["total_quota_deficit"]
            liquidity_bucket = sum(cash / initial >= edge for edge in (0.20, 0.55))
            parity_bucket = sum(gap > multiple * tolerances[0] for multiple in (1, 2)) if isfinite(gap) else "NA"
            repayment_bucket = sum(recent_rate >= edge for edge in (0.55, 0.75))
            quota_bucket = sum(shortfall > edge for edge in (0.01, 0.10))
            expected.update({
                "funded_this_round": len(loans), "matured_this_round": matured,
                "hidden_reward_funded_this_round": sum(loan[3] for loan in loans),
                "disbursed_this_round": fsum(loan[2] for loan in loans),
                "cumulative_funded": funded_total, "eventual_cumulative_reward": eventual,
                "observed_repayments": observed, "observed_defaults": matured_total - observed,
                "observed_repayment_rate": observed / matured_total if matured_total else 0.5,
                "recent_repayment_rate": recent_rate,
                "lambda_quota": quota_multiplier, "lambda_parity": parity_multiplier,
                "state_bucket": f"B{liquidity_bucket}-F{parity_bucket}-R{repayment_bucket}-Q{quota_bucket}",
                "available_capital": cash, "capital_at_risk": at_risk,
                "pending_loans": funded_total - matured_total,
                "cumulative_disbursed": spending, "cumulative_recycled": returns,
                "cumulative_default_loss": losses,
                "cumulative_unrecycled_successful_principal": unrecycled,
                "capital_turnover": ratio(spending, initial),
                "liquidity_utilization": ratio(at_risk, cash + at_risk),
                "capital_survival_ratio": ratio(cash + at_risk, initial),
            })
            if dual and r >= dual["warmup_rounds"]:
                if dual["target_shares"] != targets:
                    shortfall = sum(max(0, target - ratio(funded[group], funded_total))
                                    for group, target in dual["target_shares"].items())
                quota_multiplier = min(dual["max_lambda"], max(0, quota_multiplier +
                                       dual["dual_learning_rate"] * (shortfall - dual["quota_tolerance"])))
                if isfinite(gap):
                    parity_multiplier = min(dual["max_lambda"], max(0, parity_multiplier +
                                            dual["dual_learning_rate"] * (gap - dual["parity_tolerance"])))
        else:
            expected.update({
                "cumulative_reward": eventual, "cumulative_cost": spending,
                "budget_remaining": cash, "budget_utilization": ratio(spending, initial),
                "reward_per_cost": ratio(eventual, spending),
                "repayment_rate": ratio(eventual, funded_total),
                "observed_feedback_count": matured,
            })
        expected_rounds.append(expected)

    if expected_rounds:
        expected_frame = pd.DataFrame(expected_rounds)
        compare_group_columns(audit, location, trajectory, expected_frame, trajectory["round"].to_numpy())
        audit.fields(location, trajectory, expected_frame, trajectory["round"].to_numpy())
    if arm:
        lower_bound = min(config["minimum_cost"] if arm_spec["cost_std"] else
                          max(config["minimum_cost"], arm_spec["cost_mean"])
                          for arm_spec in config["arms"])
        audit.require(len(trajectory) == horizon or cash < lower_bound,
                      location, "arm_early_stop", f"remaining {cash}, lower bound {lower_bound}")

    settle(len(trajectory))
    fair = fairness(funded, affordable, population, targets, *tolerances)
    expected = {
        "cumulative_reward": eventual, "observed_reward_by_horizon": observed,
        "reward_per_cost": ratio(eventual, spending), "repayment_rate": ratio(eventual, funded_total),
        **fair,
    }
    if cmdp:
        all_pending = [loan for loans in pending.values() for loan in loans]
        pending_principal = fsum(loan[2] for loan in all_pending)
        audit.require(not funding_by_round, location, "no_funding_after_horizon")
        expected_maturity_frame = pd.DataFrame(expected_maturity)
        audit.compare(location, "maturity_records", len(maturity), len(expected_maturity_frame))
        audit.fields(location, maturity, expected_maturity_frame,
                     expected_maturity_frame["round"].to_numpy() if expected_maturity else None)
        expected.update({
            "rounds": len(trajectory), "loans_funded": funded_total,
            "cumulative_disbursed": spending, "capital_turnover": ratio(spending, initial),
            "cumulative_recycled": returns, "default_loss_by_horizon": losses,
            "unrecycled_successful_principal_by_horizon": unrecycled,
            "final_available_capital": cash, "final_capital_at_risk": pending_principal,
            "eventual_terminal_capital": cash + fsum(
                principal * (recycle if reward else recovery) for _, _, principal, reward in all_pending
            ),
            "capital_survival_ratio": ratio(cash + pending_principal, initial),
            "pending_loans": len(all_pending),
            "lambda_quota_final": quota_multiplier, "lambda_parity_final": parity_multiplier,
        })
        check_conservation(audit, location, trajectory, summary, initial)
    else:
        expected.update({
            "rounds": funded_total, "decision_rounds": len(trajectory), "cumulative_cost": spending,
            "budget_utilization": ratio(spending, initial), "pending_feedback": funded_total - matured_total,
        })
        audit.require(-ABS_TOLERANCE <= cash <= initial, location, "budget_conservation")
    if "sweep_name" in summary:
        expected.update(sweep_metrics(config, tolerances, expected, expected_rounds, affordable, population))
    compare_group_columns(audit, location, summary, expected)
    audit.fields(location, summary, expected)


def compare_group_columns(audit, location, actual, expected, labels=None):
    """Groups absent from an episode can leave blank count columns in mixed CSVs."""
    for name in actual.keys():
        if name.startswith(FAIRNESS_COUNT_PREFIXES):
            if name in expected:
                values = actual[name]
                if labels is not None:
                    expected[name] = expected[name].fillna(0).astype(int)
                    actual[name] = values.fillna(0)
                continue
            value = actual[name]
            value = value.fillna(0) if labels is not None else (0 if pd.isna(value) else value)
            audit.compare(location, name, value,
                          np.zeros(len(labels), dtype=int) if labels is not None else 0, labels)


def check_conservation(audit, location, trajectory, summary, initial):
    for suffix, row, columns in (
        ("round", trajectory, ("available_capital", "capital_at_risk", "cumulative_default_loss",
                               "cumulative_unrecycled_successful_principal")),
        ("horizon", summary, ("final_available_capital", "final_capital_at_risk", "default_loss_by_horizon",
                              "unrecycled_successful_principal_by_horizon")),
    ):
        cash, pending, losses, unrecycled = (row[column] for column in columns)
        labels = trajectory["round"].to_numpy() if suffix == "round" else None
        total = cash + pending + losses + unrecycled
        audit.compare(location, f"{suffix}.capital_conservation", total,
                      np.full(np.shape(total), initial, dtype=float), labels)
        expected_cash = initial - row["cumulative_disbursed"] + row["cumulative_recycled"]
        audit.compare(location, f"{suffix}.cash_conservation", cash, expected_cash, labels)
        audit.require(bool(np.all(np.asarray(cash) >= -ABS_TOLERANCE)),
                      location, f"{suffix}.capital_nonnegative")


def sweep_metrics(config, thresholds, expected, rounds, affordable, population):
    """Recompute diagnostics for one sweep case from episode and round arithmetic."""
    parity_tolerance, quota_tolerance = thresholds
    initial = config["initial_capital"]
    targets = config["target_shares"]
    loans = expected["loans_funded"]
    parity = expected["feasible_dp_gap"]
    quota_excess = max(0, expected["total_quota_deficit"] - quota_tolerance)
    parity_excess = max(0, parity - parity_tolerance) if isfinite(parity) else np.nan

    # Each additional group allocation removes at most one loan's share of deficit.
    if loans:
        gains = sorted(
            (min(1, max(0, target * loans - index))
             for group, target in targets.items()
             for index in range(min(affordable[group], ceil(target * loans)))),
            reverse=True,
        )
        quota_screen = sum(targets.values()) - fsum(gains[:loans]) / loans <= quota_tolerance + 1e-12
    else:
        quota_screen = sum(targets.values()) <= quota_tolerance

    # With a fixed minimum funding rate, each group has an integer interval.
    # A feasible minimum is an integer allocation divided by a group count.
    counts = [count for count in affordable.values() if count]
    parity_screen = np.nan
    if len(counts) >= 2:
        parity_screen = False
        for count in counts:
            for n in range(min(count, loans) + 1):
                rate = n / count
                lows = [ceil(max(0, rate * size - 1e-12)) for size in counts]
                highs = [floor(min(1, rate + parity_tolerance) * size + 1e-12) for size in counts]
                if all(lo <= hi for lo, hi in zip(lows, highs)) and (
                    sum(lows) <= loans <= sum(highs)
                ):
                    parity_screen = True
                    break
            if parity_screen:
                break
    screens_passed = parity_screen if quota_screen else False
    if screens_passed is False:
        status = "opportunity_infeasible"
    elif not isfinite(parity):
        status = "parity_not_evaluable"
    elif expected["constraint_satisfied"] == 1:
        status = "constraint_satisfied"
    else:
        status = "policy_constraint_violation"

    return {
        "initial_budget": initial, "rounds_completed": len(rounds),
        "eventual_cumulative_reward": expected["cumulative_reward"],
        "cumulative_immediate_reward": expected["observed_reward_by_horizon"],
        "cumulative_cost": expected["cumulative_disbursed"],
        "recycled_capital": expected["cumulative_recycled"],
        "final_remaining_capital": expected["final_available_capital"],
        "budget_utilization": ratio(initial - expected["final_available_capital"], initial),
        "parity_excess": parity_excess, "quota_excess": quota_excess,
        "constraint_violation": parity_excess + quota_excess,
        "fairness_violation": parity_excess + quota_excess,
        "fairness_violations": sum(
            row["parity_satisfied"] == 0 or not row["quota_satisfied"] for row in rounds
        ),
        "quota_feasible": quota_screen, "parity_feasible": parity_screen,
        "opportunity_screens_passed": screens_passed, "fairness_feasible": screens_passed,
        "feasibility_status": status, "total_opportunities": sum(affordable.values()),
        "observed_primary_minority_share": ratio(affordable["minority"], sum(affordable.values())),
        "total_population_opportunities": sum(population.values()),
        "observed_population_primary_minority_share": population["minority"] / sum(population.values()),
    }


@cache
def t_critical(n):
    return t.ppf(0.975, n - 1)


@cache
def wilson(successes, n):
    interval = binomtest(successes, n).proportion_ci(method="wilson")
    return interval.low, interval.high


def independent_statistics(values, *, proportion=False, missing_zero=False):
    values = [0.0 if pd.isna(value) else float({"True": 1, "False": 0}.get(value, value))
              for value in values if missing_zero or not pd.isna(value)]
    if not all(isfinite(value) for value in values):
        raise ValueError("non-finite statistical observation")
    n = len(values)
    mean = fsum(values) / n if n else np.nan
    std = statistics.stdev(values) if n > 1 else np.nan
    se = std / sqrt(n) if n else np.nan
    low = high = np.nan
    if proportion and n:
        if any(value not in (0, 1) for value in values):
            raise ValueError("non-binary proportion observation")
        low, high = wilson(int(sum(values)), n)
    elif n > 1:
        low, high = mean - se * t_critical(n), mean + se * t_critical(n)
    return dict(mean=mean, std=std, se=se, min=min(values, default=np.nan),
                max=max(values, default=np.nan), n=n, ci95_low=low, ci95_high=high)


def check_aggregate(audit, location, mode, summaries, aggregate, *, sweep=False):
    source_metrics = AGGREGATE_METRICS if sweep else (CMDP_METRICS if mode == "cmdp" else COMMON_METRICS)
    other_phase_metrics = ({"allocation_share_gap"} if mode != "arm" else
                           {"feasible_dp_gap", "population_dp_gap", "feasible_dp_groups", "population_dp_groups"})
    sources = {metric: metric for metric in source_metrics if metric not in other_phase_metrics}
    sources.update((name, name) for name in summaries if name.startswith(FAIRNESS_COUNT_PREFIXES))
    for rate, outcome in CONSTRAINT_RATE_FIELDS.items():
        if mode != "arm" or not rate.startswith("population_"):
            sources[rate] = outcome
    if sweep:
        # Optional when minority has no arrivals in the entire run.
        if "funding_rate_minority" not in summaries:
            sources.pop("funding_rate_minority", None)
        expected_index = {(policy, metric) for policy in summaries["policy"] for metric in sources}
    else:
        expected_index = set(summaries["policy"])
    audit.require(not aggregate.index.has_duplicates and set(aggregate.index) == expected_index,
                  location, "aggregate_keys", "missing, unexpected or duplicate aggregate rows")
    if aggregate.index.has_duplicates:
        return

    for policy, frame in summaries.reset_index(drop=True).groupby("policy"):
        label = f"{location} policy={policy} aggregate"
        evaluable = {"episodes_evaluated": len(frame)}
        evaluable.update({
            f"{field}_episodes": int(frame[field].sum())
            for field in ("parity_evaluable", "population_parity_evaluable") if field in frame
        })
        for metric, source in sources.items():
            if source not in frame:
                audit.require(False, label, source, "missing required episode metric")
                continue
            proportion = metric in CONSTRAINT_RATE_FIELDS or metric in SCREEN_METRICS
            statistics = independent_statistics(
                frame[source], proportion=proportion,
                missing_zero=metric in CONSTRAINT_RATE_FIELDS or source.startswith(FAIRNESS_COUNT_PREFIXES),
            )
            if sweep and (policy, metric) in aggregate.index:
                row = aggregate.loc[(policy, metric)]
                fields = {"n_seeds": statistics.pop("n"),
                          **{f"metric_{key}": value for key, value in statistics.items()}, **evaluable}
                audit.fields(f"{label} metric={metric}", row, fields)
            elif not sweep and policy in aggregate.index:
                row = aggregate.loc[policy]
                fields = {(metric, "Unnamed" if key == "mean" and proportion else key): value
                          for key, value in statistics.items()}
                audit.fields(label, row, fields)
        if not sweep and policy in aggregate.index:
            audit.fields(label, aggregate.loc[policy],
                         {(field, "Unnamed"): count for field, count in evaluable.items()})


def check_paired(audit, location, mode, summaries, pairs):
    parity = "allocation_share_gap" if mode == "arm" else "feasible_dp_gap"
    metrics = ("cumulative_reward", "reward_per_cost", parity, "total_quota_deficit")
    matrices = {metric: summaries.pivot(index="seed", columns="policy", values=metric)
                for metric in metrics}
    expected = {}
    for policy, baseline in FAIRNESS_COMPARISONS[mode].items():
        if policy in summaries["policy"].values:
            for metric, matrix in matrices.items():
                expected[policy, baseline, metric] = independent_statistics(matrix[policy] - matrix[baseline])
    audit.require(not pairs.index.has_duplicates and set(expected) == set(pairs.index),
                  location, "paired_keys", "missing, unexpected or duplicate paired comparisons")
    for key, fields in expected.items():
        if key in pairs.index and pairs.index.is_unique:
            audit.fields(f"{location} paired policy={key[0]} baseline={key[1]} metric={key[2]}", pairs.loc[key], fields)


def group_frames(frame, keys):
    return dict(tuple(frame.groupby(keys, sort=False))) if not frame.empty else {}


def sweep_cases(manifest):
    for case in manifest["cases"]:
        yield (
    (case["value"], case["parameter_kind"]), case["policy_configurations"],
            {**case["environment"], "target_shares": case["target_shares"]},
            {"parity_tolerance": case["parity_tolerance"], "quota_tolerance": manifest["quota_tolerance"]},
        )


def verify_experiment(audit, directory, location):
    sweep = (directory / "config.json").is_file()
    manifest = json_file(directory / ("config.json" if sweep else "run_manifest.json"))
    mode = "cmdp" if sweep else manifest["mode"]
    audit.require(
        isinstance(manifest.get("git_commit"), str) and isinstance(manifest.get("git_dirty"), bool)
        and {"python", "numpy", "pandas", "matplotlib", "scipy"} <= manifest.get("package_versions", {}).keys(),
        location, "manifest_provenance", "missing Git or software version metadata",
    )
    if manifest.get("audit_format") != 1:
        raise ValueError("raw opportunity records require audit format 1")
    cases = list(sweep_cases(manifest)) if sweep else [
        ((), manifest["policy_configurations"], manifest["configuration"], manifest["fairness_thresholds"])
    ]
    seeds = manifest["seeds"]
    policy_lists = [case["policies"] for case in manifest["cases"]] if sweep else [manifest["policies"]]
    for (case, configs, _, _), policies in zip(cases, policy_lists):
        audit.require(set(configs) == set(policies) and len(policies) == len(set(policies)),
                      f"{location} case={case}", "manifest_policy_configurations")
    audit.require(len(cases) == len({case[0] for case in cases}) and len(seeds) == len(set(seeds)),
                  location, "manifest_unique_cases_and_seeds")
    keys = [*CASE_KEY, *EPISODE_KEY] if sweep else EPISODE_KEY
    summaries = read_csv(directory / ("raw_results.csv" if sweep else "episode_summaries.csv"))
    trajectories = group_frames(read_csv(directory / "trajectories.csv"), keys)
    funding = group_frames(read_csv(directory / "funding_log.csv"), keys) if mode == "cmdp" else {}
    maturity = group_frames(read_csv(directory / "maturity_log.csv"), keys) if mode == "cmdp" else {}
    empty = pd.DataFrame()
    expected = {(*case, policy, seed) for case, policies, _, _ in cases
                for policy in policies for seed in seeds}
    observed = set(summaries[keys].itertuples(index=False, name=None))
    audit.compare(location, "manifest.expected_episodes", manifest["expected_episodes"], len(expected))
    audit.compare(location, "manifest.actual_episodes", manifest["actual_episodes"], len(summaries))
    audit.compare(location, "manifest.status", manifest["status"], "complete")
    audit.require(len(summaries) == len(observed), location, "duplicate_episode_identifiers",
                  str(summaries.loc[summaries.duplicated(keys), keys].to_dict(orient="records")))
    for label, actual in (("episode", observed), ("trajectory", trajectories),
                          ("funding", funding), ("maturity", maturity)):
        extra = set(actual) - expected
        audit.require(not extra, location, f"unexpected_{label}_identifiers", str(sorted(extra)))
    missing = expected - observed
    audit.require(not missing, location, "missing_episode_identifiers", str(sorted(missing)))
    if len(summaries) != len(observed):
        return

    audit.experiments += 1
    aggregate = (read_csv(directory / "aggregate_results.csv") if sweep else
                 read_csv(directory / "aggregate_results.csv", header=[0, 1], index_col=0))
    if sweep:
        audit.require(aggregate["sweep_name"].eq(manifest["sweep_name"]).all(),
                      location, "aggregate_sweep_name")
        expected_cases = {case for case, *_ in cases}
        aggregate_cases = set(aggregate[CASE_KEY].itertuples(index=False, name=None))
        audit.require(expected_cases == aggregate_cases, location, "aggregate_case_keys")
    else:
        aggregate.columns = pd.MultiIndex.from_tuples(
            (metric, "Unnamed" if statistic.startswith("Unnamed:") else statistic)
            for metric, statistic in aggregate.columns
        )
    summaries = summaries.set_index(keys, drop=False)
    for index, (case, policies, config, thresholds) in enumerate(cases):
        case_location = f"{location} case={case}" if case else location
        if sweep and not any(key[:len(case)] == case for key in observed):
            continue
        case_summaries = summaries.xs(case, level=CASE_KEY, drop_level=False) if sweep else summaries
        opening_pools = {}
        potential_outcomes = defaultdict(dict)
        for policy, policy_settings in policies.items():
            for seed in seeds:
                key = (*case, policy, seed)
                if key not in observed:
                    continue
                episode_location = f"{case_location} policy={policy} seed={seed}"
                try:
                    replay_episode(audit, episode_location, mode, config, thresholds, summaries.loc[key].copy(),
                                   trajectories.get(key, empty).copy(), funding.get(key, empty),
                                   maturity.get(key, empty), policy_settings)
                    audit.episodes += 1
                except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as error:
                    audit.require(False, episode_location, "invalid_raw_record", str(error))
        if sweep:
            case_manifest = manifest["cases"][index]
            for label, frame in (("summary", case_summaries), ("aggregate", aggregate)):
                if label == "aggregate":
                    frame = frame.loc[frame["sweep_value"].eq(case[0]) & frame["parameter_kind"].eq(case[1])]
                audit.require(frame["is_baseline"].eq(case_manifest["is_baseline"]).all(),
                              case_location, f"{label}.baseline_flag")
            audit.require(case_summaries["sweep_name"].eq(manifest["sweep_name"]).all(),
                          case_location, "summary_sweep_name")
            for name, value in case_manifest["parameter_details"].items():
                if isinstance(value, (str, int, float, bool)):
                    audit.fields(case_location, case_summaries, {name: np.full(len(case_summaries), value)})
            value, kind = case
            groups = aggregate.loc[aggregate["sweep_value"].eq(value) & aggregate["parameter_kind"].eq(kind)]
            groups = groups.set_index(["policy", "metric"])
        else:
            groups = aggregate
        check_aggregate(audit, case_location, mode, case_summaries, groups, sweep=sweep)

    if not sweep and observed == expected:
        check_paired(audit, location, mode, summaries.reset_index(drop=True),
                     read_csv(directory / "paired_comparisons.csv").set_index(["policy", "baseline", "metric"]))
    if sweep:
        check_sweep_summary(audit, location, directory, summaries)
    else:
        check_ranked(audit, location, directory, mode, manifest, aggregate)


def check_common_random_numbers(audit, location, mode, seed, trajectory, funding, pools, outcomes):
    if trajectory.empty:
        return
    pool = trajectory["opportunity_pool"].to_numpy()
    previous = pools.get(seed, ())
    shared_rounds = min(len(previous), len(pool))
    if shared_rounds:
        audit.compare(location, "paired_opportunity_pools", pool[:shared_rounds], previous[:shared_rounds],
                      np.arange(shared_rounds))
    if len(pool) > len(previous):
        pools[seed] = pool

    if mode == "cmdp":
        columns = ("round", "applicant_id", "true_repayment_probability", "hidden_reward")
    else:
        funding = trajectory.loc[trajectory["selected_index"].notna()]
        columns = ("round", "selected_index", "true_repayment_probability", "reward")
    if not funding.empty:
        for round_number, action, probability, reward in funding.loc[:, columns].itertuples(index=False, name=None):
            previous = outcomes[seed].setdefault((round_number, action), (probability, reward))
            audit.require(previous == (probability, reward), f"{location} round={round_number} action={action}",
                          "paired_potential_outcome", f"got {(probability, reward)}, expected {previous}")


def check_sweep_summary(audit, location, directory, summaries):
    status = summaries["feasibility_status"].value_counts()
    fields = {"episodes": len(summaries),
              "parity_not_evaluable_episodes": int((~summaries["parity_evaluable"]).sum()),
              "population_parity_not_evaluable_episodes": int((~summaries["population_parity_evaluable"]).sum())}
    fields.update({f"{name}_episodes": int(status.get(name, 0)) for name in (
        "opportunity_infeasible", "policy_constraint_violation", "constraint_satisfied"
    )})
    audit.fields(f"{location} summary", json_file(directory / "summary.json"), fields)


def check_ranked(audit, location, directory, mode, manifest, aggregate):
    if mode == "arm":
        return
    ranked = read_csv(directory / "ranked_results.csv").set_index("policy")
    audit.require(ranked.index.is_unique and set(ranked.index) == set(aggregate.index),
                  location, "ranked_policy_keys")
    if not ranked.index.is_unique:
        return
    thresholds = manifest["fairness_thresholds"]
    order = []
    for policy, row in aggregate.iterrows():
        if policy not in ranked.index:
            continue
        label = f"{location} policy={policy} ranked"
        parity = row["demographic_parity_gap", "mean"]
        quota = row["total_quota_deficit", "mean"]
        compliance = row["constraint_satisfaction_rate", "Unnamed"]
        parity_excess = max(0, parity - thresholds["parity_tolerance"]) if isfinite(parity) else np.nan
        quota_excess = max(0, quota - thresholds["quota_tolerance"])
        violation = parity_excess + quota_excess
        fields = {metric if statistic == "Unnamed" else f"{metric}_{statistic}": value
                  for (metric, statistic), value in row.items()}
        fields.update(parity_excess=parity_excess, quota_excess=quota_excess,
                      fairness_violation=violation, fairness_feasible=compliance == 1,
                      mean_thresholds_met=violation == 0)
        audit.fields(label, ranked.loc[policy], fields)
        order.append((-compliance, 0 if compliance == 1 else (violation if isfinite(violation) else np.inf),
                      -row["reward_per_cost", "mean"], -row["cumulative_reward", "mean"], policy))
    expected_order = [entry[-1] for entry in sorted(order)]
    audit.compare(location, "ranked_policy_order", ranked.index.to_numpy(), expected_order)
    audit.compare(location, "rank", ranked["rank"], np.arange(1, len(expected_order) + 1))


def check_sweep_baselines(audit, directory, location):
    manifest = json_file(directory / "run_manifest.json")
    standard = directory.parent / "cmdp"
    standard_manifest = json_file(standard / "run_manifest.json") if standard.exists() else None
    reference_config = reference = reference_policies = reference_tolerances = None
    for name in ("budget", "group_imbalance", "fairness_strength"):
        path = directory / "sweeps" / name
        sweep = json_file(path / "config.json")
        raw = read_csv(path / "raw_results.csv")
        for case in sweep["cases"]:
            selected = raw["sweep_value"].eq(case["value"]) & raw["parameter_kind"].eq(case["parameter_kind"])
            case_label = f"{location} {name} case={case['value'], case['parameter_kind']}"
            audit.compare(case_label, "baseline_flag", raw.loc[selected, "is_baseline"],
                          np.full(selected.sum(), case["is_baseline"], dtype=bool))
            if not case["is_baseline"]:
                continue
            candidate = raw.loc[selected].set_index(EPISODE_KEY).sort_index()
            config = {**case["environment"], "target_shares": case["target_shares"]}
            tolerances = {"parity_tolerance": case["parity_tolerance"],
                          "quota_tolerance": sweep["quota_tolerance"]}
            if reference is None:
                if standard_manifest is not None:
                    reference_config = {key: value for key, value in standard_manifest["configuration"].items()
                                        if key != "protected_attributes_in_prediction"}
                    reference_policies = standard_manifest["policy_configurations"]
                    reference_tolerances = standard_manifest["fairness_thresholds"]
                    reference = read_csv(standard / "episode_summaries.csv").set_index(EPISODE_KEY).sort_index()
                else:
                    reference_config, reference = config, candidate
                    reference_policies, reference_tolerances = case["policy_configurations"], tolerances
                    audit.unverified.add(f"{location}: standalone CMDP baseline (no sibling CMDP run)")
            labels = candidate.index
            subset = reference.loc[reference.index.get_level_values("policy").isin(case["policies"])]
            matched = (
                subset.index.equals(labels) and labels.is_unique
                and config == reference_config and tolerances == reference_tolerances
                and all(settings == reference_policies.get(policy)
                        for policy, settings in case["policy_configurations"].items())
            )
            audit.require(matched, case_label, "baseline_configuration_and_episode_keys")
            if matched:
                fields = set(subset) & set(candidate)
                for field in sorted(fields):
                    left, right = candidate[field].to_numpy(), subset[field].to_numpy()
                    if field.startswith(FAIRNESS_COUNT_PREFIXES):
                        left = np.nan_to_num(left)
                        right = np.nan_to_num(right).astype(int)
                    audit.compare(case_label, f"baseline.{field}", left, right, labels)


def check_suite(audit, root, manifest):
    phases = ("arm", "contextual", "cmdp", "cmdp_robustness")
    configurations = manifest["scenarios"]
    expected_children = [root / phase for phase in phases] + [
        root / "all_scenarios" / scenario / phase
        for scenario in configurations for phase in phases
    ]
    expected = actual = 0
    for directory in expected_children:
        location = str(directory.relative_to(root))
        path = directory / "run_manifest.json"
        if not path.is_file():
            audit.require(False, location, "suite_child_manifest", "missing required run")
            continue
        child = json_file(path)
        audit.compare(location, "manifest.status", child["status"], "complete")
        expected += child["expected_episodes"]
        actual += child["actual_episodes"]
    audit.fields(str(root), manifest, {
        "expected_episodes": expected, "actual_episodes": actual, "status": "complete",
    })
    audit.require(len(configurations) == 10, str(root), "suite_scenario_count")

    combined = read_csv(root / "combined_aggregate_results.csv").set_index(["mode", "policy"])
    expected_keys = set()
    for phase in phases[:3]:
        table = read_csv(root / phase / "aggregate_results.csv", header=[0, 1], index_col=0)
        reward = table["cumulative_reward", "mean"]
        ranks = reward.rank(method="min", ascending=False)
        for policy, row in table.iterrows():
            key = (phase, policy)
            expected_keys.add(key)
            if key not in combined.index or not combined.index.is_unique:
                continue
            fields = {metric if column.startswith("Unnamed:") else f"{metric}_{column}": value
                      for (metric, column), value in row.items() if metric != "demographic_parity_gap"}
            fields["rank_within_mode"] = ranks.loc[policy]
            audit.fields(f"combined phase={phase} policy={policy}", combined.loc[key], fields)
    audit.require(combined.index.is_unique and set(combined.index) == expected_keys,
                  str(root), "combined_policy_keys")


def audit_outputs(root, *, summary_path=None):
    """Check saved leaf experiments and baseline consistency beneath a run directory."""
    root = Path(root)
    audit = Audit()
    leaves = set()
    sweeps = []
    suites = []
    for manifest_path in root.rglob("run_manifest.json"):
        directory = manifest_path.parent
        try:
            manifest = json_file(manifest_path)
            mode = manifest["mode"]
            if mode == 5:
                suites.append((directory, manifest))
            if mode in ("arm", "contextual", "cmdp"):
                leaves.add(directory)
            elif mode == "Phase 3 CMDP robustness sweeps":
                sweeps.append(directory)
                leaves.update(directory / "sweeps" / name for name in (
                    "budget", "group_imbalance", "fairness_strength"
                ))
        except (ValueError, KeyError, OSError) as error:
            audit.require(False, str(manifest_path.relative_to(root)), "invalid_manifest", str(error))
    # Also catch data files with missing manifests and accept a single sweep directory.
    for filename in ("aggregate_results.csv", "episode_summaries.csv", "raw_results.csv"):
        leaves.update(path.parent for path in root.rglob(filename))
    audit.require(bool(leaves), str(root), "experiment_outputs_found", "no experiment outputs found")
    for directory in sorted(leaves):
        location = str(directory.relative_to(root))
        try:
            verify_experiment(audit, directory, location)
        except (KeyError, ValueError, TypeError, OSError, OverflowError) as error:
            audit.require(False, location, "invalid_experiment", str(error))
    for directory in sorted(sweeps):
        location = str(directory.relative_to(root))
        try:
            check_sweep_baselines(audit, directory, location)
        except (KeyError, ValueError, TypeError, OSError) as error:
            audit.require(False, location, "invalid_sweep_baselines", str(error))
    for directory, manifest in suites:
        try:
            check_suite(audit, directory, manifest)
        except (KeyError, ValueError, TypeError, OSError) as error:
            audit.require(False, str(directory.relative_to(root)), "invalid_suite", str(error))
    report = audit.report()
    if summary_path is not None:
        Path(summary_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="A completed phase, sweep, or suite directory")
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit_outputs(args.output, summary_path=args.report)
    print(json.dumps(report, indent=2))
    return int(report["failed_checks"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
