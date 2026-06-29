import numpy as np

from models.arm_spec import ArmSpec


class Arms:
    def __init__(self):
        self.arms = None
        self.set_arms(self.get_arms_group_combination())

        self.groups = {
            "A": "High Repayment, High Cost",
            "B": "Medium Repayment, Medium Cost",
            "C": "Low Repayment, Low Cost",
            "D": "High Repayment, Low Cost",
            "E": "Low Repayment, High Cost",
        }

    def set_arms(self, arms):
        self.arms = arms

    def get_arms(self):
        return self.arms

    def simulate_random_arms(self):
        high_repayment_high_cost = ArmSpec(
            name="High Repayment, High Cost",
            group="A",
            repayment_probability=np.random.uniform(0.7, 0.9),
            cost_mean=np.random.uniform(10, 15),
        )
        medium_repayment_medium_cost = ArmSpec(
            name="Medium Repayment, Medium Cost",
            group="B",
            repayment_probability=np.random.uniform(0.5, 0.7),
            cost_mean=np.random.uniform(5, 10),
        )
        low_repayment_low_cost = ArmSpec(
            name="Low Repayment, Low Cost",
            group="C",
            repayment_probability=np.random.uniform(0.3, 0.5),
            cost_mean=np.random.uniform(1, 5),
        )
        high_repayment_low_cost = ArmSpec(
            name="High Repayment, Low Cost",
            group="D",
            repayment_probability=np.random.uniform(0.7, 0.9),
            cost_mean=np.random.uniform(1, 5),
        )
        low_repayment_high_cost = ArmSpec(
            name="Low Repayment, High Cost",
            group="E",
            repayment_probability=np.random.uniform(0.3, 0.5),
            cost_mean=np.random.uniform(10, 15),
        )
        return [
            high_repayment_high_cost,
            medium_repayment_medium_cost,
            low_repayment_low_cost,
            high_repayment_low_cost,
            low_repayment_high_cost,
        ]

    def get_arms_group_combination(self):
        return [
            ArmSpec(
                name="Lorem Ipsum1",
                group="A",
                repayment_probability=0.80,
                cost_mean=12.0,
            ),
            ArmSpec(
                name="Lorem Ipsum2",
                group="B",
                repayment_probability=0.65,
                cost_mean=8.0,
            ),
            ArmSpec(
                name="Lorem Ipsum3",
                group="C",
                repayment_probability=0.50,
                cost_mean=4.0,
            ),
            ArmSpec(
                name="Lorem Ipsum4",
                group="D",
                repayment_probability=0.80,
                cost_mean=3.0,
            ),
            ArmSpec(
                name="Lorem Ipsum5",
                group="D",
                repayment_probability=0.30,
                cost_mean=10.0,
            ),
        ]
