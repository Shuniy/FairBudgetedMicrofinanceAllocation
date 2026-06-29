from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    num_seeds: int = 10
    log_every_round: bool = True
    save_curves: bool = True
    compute_demographic_parity: bool = True
    compute_quota_satisfaction: bool = True
    compute_equal_opportunity_proxy: bool = False
