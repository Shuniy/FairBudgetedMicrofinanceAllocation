from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_phase3 import build_policy_factories
from phase2.feature_encoder import BorrowerFeatureEncoder
from phase3.models import CMDPEnvironmentConfig
from phase3.runner import compare_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-3 repayment-delay sweep")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--delays", type=int, nargs="+", default=[2, 5, 8, 12, 20])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs_phase3_delay_sweep.csv")
    args = parser.parse_args()

    target_shares = {
        "disabled": 0.04,
        "reserved": 0.08,
        "minority": 0.12,
        "women": 0.25,
    }
    encoder = BorrowerFeatureEncoder(include_protected_attributes=False)
    rows: list[pd.DataFrame] = []
    for delay in args.delays:
        config = CMDPEnvironmentConfig(
            initial_capital=600_000.0,
            max_rounds=args.rounds,
            pool_size=20,
            max_loans_per_round=3,
            repayment_delay_rounds=delay,
        )
        summaries, _, _, _ = compare_policies(
            env_config=config,
            policy_factories=build_policy_factories(target_shares, 0.05),
            target_shares=target_shares,
            parity_tolerance=0.05,
            seeds=list(range(args.seeds)),
            encoder=encoder,
        )
        summaries["delay"] = delay
        rows.append(summaries)

    result = pd.concat(rows, ignore_index=True)
    grouped = result.groupby(["delay", "policy"])[
        [
            "cumulative_reward",
            "capital_turnover",
            "eventual_terminal_capital",
            "demographic_parity_gap",
            "quota_satisfaction_rate",
        ]
    ].agg(["mean", "std"])
    grouped.to_csv(args.output)
    print(grouped.round(4).to_string())
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
