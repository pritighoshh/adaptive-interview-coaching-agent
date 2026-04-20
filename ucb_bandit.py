"""
ucb_bandit.py
Upper Confidence Bound (UCB1) Contextual Bandit for Interview Topic Selection.

The bandit decides WHICH topic to focus on next, balancing:
  - Exploitation: topics where the candidate historically scores well (high reward signal)
  - Exploration: topics not yet sufficiently tested (high uncertainty)

This complements the DQN: the DQN decides HOW to ask (action type),
the bandit decides WHAT to ask about (topic).
"""

import numpy as np
from typing import List, Dict, Optional
import math


class UCBBandit:
    """
    UCB1 bandit over interview topics.

    Reward signal: answer_quality scores returned by the environment.
    The bandit maximises cumulative reward — meaning it focuses on topics
    where coaching yields the most improvement (high quality + high variance).

    UCB1 selection rule:
        a* = argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ]

    where:
        Q(a)  = empirical mean reward for arm a
        N(a)  = number of times arm a was pulled
        t     = total pulls so far
        c     = exploration coefficient (default 1.0)
    """

    def __init__(self, arms: List[str], c: float = 1.0):
        self.arms  = arms
        self.n     = len(arms)
        self.c     = c

        # Per-arm statistics
        self.counts:  np.ndarray = np.zeros(self.n)        # N(a)
        self.values:  np.ndarray = np.zeros(self.n)        # Q(a) – empirical mean
        self.rewards_log: Dict[str, List[float]] = {a: [] for a in arms}

        self.total_pulls: int = 0
        self.selection_history: List[str] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def select(self) -> str:
        """Return the topic arm to pull next."""
        # Initialisation phase: pull each arm at least once
        for i, arm in enumerate(self.arms):
            if self.counts[i] == 0:
                self.selection_history.append(arm)
                return arm

        ucb_scores = self._compute_ucb()
        chosen_idx = int(np.argmax(ucb_scores))
        chosen_arm = self.arms[chosen_idx]
        self.selection_history.append(chosen_arm)
        return chosen_arm

    def update(self, arm: str, reward: float):
        """Update statistics after observing reward for the chosen arm."""
        idx = self.arms.index(arm)
        self.counts[idx]  += 1
        self.total_pulls  += 1
        # Incremental mean update
        n = self.counts[idx]
        self.values[idx] += (reward - self.values[idx]) / n
        self.rewards_log[arm].append(reward)

    def get_stats(self) -> Dict[str, dict]:
        """Return per-arm statistics for reporting."""
        return {
            arm: {
                "pulls":        int(self.counts[i]),
                "mean_reward":  float(self.values[i]),
                "ucb_score":    float(self._compute_ucb()[i])
                                if self.total_pulls > 0 else float("inf"),
                "rewards":      self.rewards_log[arm],
            }
            for i, arm in enumerate(self.arms)
        }

    def topic_coverage(self) -> Dict[str, float]:
        """Return fraction of total pulls per topic (for coverage analysis)."""
        total = max(1, self.total_pulls)
        return {arm: float(self.counts[i]) / total for i, arm in enumerate(self.arms)}

    def reset(self):
        self.counts       = np.zeros(self.n)
        self.values       = np.zeros(self.n)
        self.rewards_log  = {a: [] for a in self.arms}
        self.total_pulls  = 0
        self.selection_history = []

    # ── Private ──────────────────────────────────────────────────────────────

    def _compute_ucb(self) -> np.ndarray:
        """Compute UCB1 scores for all arms."""
        exploration = self.c * np.sqrt(
            np.log(self.total_pulls + 1) / (self.counts + 1e-9)
        )
        return self.values + exploration


class ThompsonSamplingBandit:
    """
    Alternative bandit using Thompson Sampling (Beta-Bernoulli conjugate).
    Treat answer_quality > 0.5 as a 'success', else 'failure'.
    Included as a comparative baseline.
    """

    def __init__(self, arms: List[str]):
        self.arms  = arms
        self.alpha = np.ones(len(arms))   # Beta distribution α (successes + 1)
        self.beta_ = np.ones(len(arms))   # Beta distribution β (failures  + 1)
        self.selection_history: List[str] = []

    def select(self) -> str:
        samples = np.random.beta(self.alpha, self.beta_)
        chosen  = self.arms[int(np.argmax(samples))]
        self.selection_history.append(chosen)
        return chosen

    def update(self, arm: str, reward: float):
        idx = self.arms.index(arm)
        if reward >= 0.5:
            self.alpha[idx] += 1
        else:
            self.beta_[idx] += 1

    def topic_coverage(self) -> Dict[str, float]:
        total = float(self.alpha.sum() + self.beta_.sum() - 2 * len(self.arms))
        total = max(1.0, total)
        pulls = self.alpha + self.beta_ - 2
        return {arm: float(pulls[i]) / total for i, arm in enumerate(self.arms)}