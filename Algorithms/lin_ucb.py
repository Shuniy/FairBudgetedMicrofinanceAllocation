import numpy as np


class LinUCB:
    """
    Disjoint LinUCB:
    One linear model per arm.

    reward_hat = theta_a.T x + alpha * sqrt(x.T A_a^-1 x)
    """

    def __init__(self, n_arms, n_features, alpha=1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # A_a starts as identity matrix for each arm
        self.A = [np.eye(n_features) for _ in range(n_arms)]

        # b_a starts as zero vector for each arm
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, contexts):
        """
        contexts shape:
        - (n_arms, n_features)
        One feature vector per arm/borrower.
        """

        scores = []

        for arm in range(self.n_arms):
            x = contexts[arm]

            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]

            mean_reward = theta @ x
            uncertainty = np.sqrt(x @ A_inv @ x)

            ucb_score = mean_reward + self.alpha * uncertainty
            scores.append(ucb_score)

        return int(np.argmax(scores))

    def update(self, arm, context, reward):
        """
        Update only the selected arm.
        """

        x = context

        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
