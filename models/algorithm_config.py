from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    """
    They are algorithm hyperparameters, not borrower fields.
    """

    # Exploration
    epsilon: float = 0.1
    epsilon_decay: float = 0.99
    exploration_budget_fraction: float = 0.2

    # Fairness penalty / cost encoding
    penalty_lambda: float = 0.0
    cost_lambda: float = 0.0

    # Optional reward mixing
    reward_weight: float = 1.0
    social_impact_weight: float = 0.0
    poverty_relief_weight: float = 0.0
    indebtedness_penalty_weight: float = 0.0

    # Conservative selection
    use_reward_per_cost: bool = True
