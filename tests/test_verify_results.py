import json
from shutil import copytree

import pandas as pd
import pytest

import evaluate
import main as runner
import robustness
from main import parse_args, run_cmdp
from verify_results import Audit, audit_outputs, check_suite, main


def test_saved_output_verifier_detects_ledger_and_aggregate_discrepancies(tmp_path, monkeypatch):
    args = parse_args(["--mode", "3", "--seeds", "1", "--rounds", "3", "--pool-size", "4",
                       "--batch-size", "1", "--repayment-delay", "1"])
    run_cmdp(args, tmp_path)
    valid = audit_outputs(tmp_path)
    assert valid["episodes_checked"] == 12
    assert valid["failed_checks"] == 0

    maturity_path = tmp_path / "maturity_log.csv"
    maturities = pd.read_csv(maturity_path)
    maturities.loc[0, "capital_returned"] += 50
    maturities.to_csv(maturity_path, index=False)
    results_path = tmp_path / "aggregate_results.csv"
    aggregates = pd.read_csv(results_path, header=[0, 1], index_col=0)
    aggregates.iloc[0, aggregates.columns.get_loc(("cumulative_reward", "mean"))] += 1
    aggregates.to_csv(results_path)
    report = tmp_path / "verification.json"
    monkeypatch.setattr("sys.argv", ["verify_results.py", str(tmp_path), "--report", str(report)])
    assert main() == 1
    discrepancies = json.loads(report.read_text())["discrepancies"]
    assert any("seed=0" in entry and "capital_returned" in entry for entry in discrepancies)
    assert any("cumulative_reward" in entry and "mean" in entry for entry in discrepancies)


def test_saved_arm_outputs_pass_with_different_episode_lengths(tmp_path):
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps({
        "arms": [
            {"name": group, "group": group, "repayment_probability": probability,
             "cost_mean": cost, "cost_std": 0}
            for group, probability, cost in (("A", 0.8, 12), ("B", 0.65, 8), ("C", 0.5, 4))
        ],
        "target_shares": {"A": 0.2, "B": 0.3, "C": 0.3},
    }))
    args = parse_args(["--mode", "1", "--seeds", "2", "--rounds", "6", "--budget", "20",
                       "--arms-config", str(arms)])
    runner.run_arm(args, tmp_path)
    episodes = pd.read_csv(tmp_path / "episode_summaries.csv")
    assert episodes.groupby("seed")["decision_rounds"].nunique().gt(1).all()
    report = audit_outputs(tmp_path)
    assert report["episodes_checked"] == 26
    assert report["failed_checks"] == 0, report["discrepancies"]


@pytest.fixture(scope="module")
def contextual_outputs(tmp_path_factory):
    output = tmp_path_factory.mktemp("contextual-audit")
    args = parse_args(["--mode", "2", "--seeds", "2", "--rounds", "3", "--pool-size", "4",
                       "--feedback-delay", "1"])
    runner.run_contextual(args, output)
    return output


def test_saved_contextual_outputs_pass(contextual_outputs):
    report = audit_outputs(contextual_outputs)
    assert report["episodes_checked"] == 28
    assert report["failed_checks"] == 0, report["discrepancies"]


@pytest.mark.parametrize(("filename", "check"), [
    ("trajectories.csv", "paired_opportunity_pools"),
    ("episode_summaries.csv", "population_opportunities_none"),
    ("episode_summaries.csv", "missing_episode_identifiers"),
    ("episode_summaries.csv", "duplicate_episode_identifiers"),
    ("paired_comparisons.csv", "ci95_low"),
])
def test_saved_contextual_verifier_detects_corruption(contextual_outputs, tmp_path, filename, check):
    output = copytree(contextual_outputs, tmp_path / "corrupted")
    path = output / filename
    table = pd.read_csv(path, float_precision="round_trip")
    if check == "paired_opportunity_pools":
        index = table.loc[table["policy"].eq(table["policy"].max())].index[0]
        pool = json.loads(table.loc[index, "opportunity_pool"])
        unfunded = next(opportunity for opportunity in pool if opportunity[0] != table.loc[index, "identifier"])
        unfunded[2] += 1
        table.loc[index, "opportunity_pool"] = json.dumps(pool, separators=(",", ":"))
    elif check == "missing_episode_identifiers":
        table = table.iloc[1:]
    elif check == "duplicate_episode_identifiers":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    else:
        index = table[check].first_valid_index()
        table.loc[index, check] += 1
    table.to_csv(path, index=False)

    report = audit_outputs(output)
    assert report["failed_checks"] > 0
    assert any(f" | {check} | " in entry and (check != "ci95_low" or " paired policy=" in entry)
               for entry in report["discrepancies"]), report["discrepancies"]
    if check == "paired_opportunity_pools":
        assert report["failed_checks"] == 1, report["discrepancies"]


def test_saved_sweep_outputs_pass_and_detect_inconsistent_shared_repayment(tmp_path):
    args = parse_args(["--mode", "4", "--quick-test", "--seeds", "1", "--rounds", "3",
                       "--pool-size", "4", "--batch-size", "1", "--repayment-delay", "1"])
    runner.run_cmdp_sweeps(args, tmp_path)
    report = audit_outputs(tmp_path)
    assert report["experiments_checked"] == 3
    assert report["episodes_checked"] == 60
    assert report["failed_checks"] == 0, report["discrepancies"]

    budget = tmp_path / "sweeps" / "budget"
    funding_path = budget / "funding_log.csv"
    funding = pd.read_csv(funding_path, float_precision="round_trip")
    shared = funding.duplicated(["sweep_value", "seed", "round", "applicant_id"], keep=False)
    index = funding.loc[shared].sort_values("policy").index[-1]
    funding.loc[index, "hidden_reward"] = 1 - funding.loc[index, "hidden_reward"]
    funding.to_csv(funding_path, index=False)

    report = audit_outputs(budget)
    assert report["failed_checks"] > 0
    assert any(" | paired_potential_outcome | " in entry for entry in report["discrepancies"])


def test_suite_checks_saved_scenario_configurations_and_seeds(tmp_path, monkeypatch):
    for name, config in runner.SCENARIOS.items():
        monkeypatch.setitem(runner.SCENARIOS, name, dict(config, seeds=1, rounds=1, pool_size=4, batch_size=1))
    monkeypatch.setattr(runner, "BUDGET_SWEEP_VALUES", runner.QUICK_BUDGET_SWEEP_VALUES)
    monkeypatch.setattr(runner, "IMBALANCE_SWEEP_VALUES", runner.QUICK_IMBALANCE_SWEEP_VALUES)
    monkeypatch.setattr(runner, "FAIRNESS_SWEEP_VALUES", runner.QUICK_FAIRNESS_SWEEP_VALUES)
    monkeypatch.setattr(evaluate, "plot_metric_over_time", lambda *args: None)
    monkeypatch.setattr(robustness, "_plot_sweep", lambda *args: ())
    args = parse_args(["--mode", "5", "--quick-test", "--seeds", "1", "--rounds", "1",
                       "--pool-size", "4", "--batch-size", "1", "--output", str(tmp_path)])
    runner.run_all(args)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    audit = Audit()
    check_suite(audit, tmp_path, manifest)
    assert not audit.failures

    scenario = tmp_path / "all_scenarios" / "scenario_1"
    contextual_path = scenario / "contextual" / "run_manifest.json"
    contextual = json.loads(contextual_path.read_text())
    contextual["configuration"]["feedback_delay_rounds"] += 1
    contextual_path.write_text(json.dumps(contextual))
    sweep_path = scenario / "cmdp_robustness" / "sweeps" / "budget" / "config.json"
    sweep = json.loads(sweep_path.read_text())
    sweep["seeds"][0] += 1
    sweep["default_phase3_parameters"]["minimum_liquidity_reserve"] += 1
    sweep_path.write_text(json.dumps(sweep))

    audit = Audit()
    check_suite(audit, tmp_path, manifest)
    assert any("scenario_1/contextual configuration | feedback_delay_rounds | " in entry
               for entry in audit.failures)
    assert any("scenario_1/cmdp_robustness sweep=budget | seeds | " in entry for entry in audit.failures)
    assert any("scenario_1/cmdp_robustness sweep=budget baseline | minimum_liquidity_reserve | " in entry
               for entry in audit.failures)
