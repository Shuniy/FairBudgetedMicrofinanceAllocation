import json

import pandas as pd

from main import parse_args, run_cmdp
from verify_results import audit_outputs, main


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
