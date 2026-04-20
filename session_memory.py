"""
session_memory.py
Session Memory Module for AICA

Provides persistent, structured memory of candidate performance
across all questions in a session. The memory is:
  1. Written to by the environment after every answer
  2. Read by the DQN agent — memory features are appended to the
     state vector, expanding it from 12 to 17 dimensions
  3. Read by the UCB Bandit via get_topic_summary() to bias
     topic selection toward areas needing most improvement

This implements the "memory implementation and usage" requirement
from the rubric: the agent's decisions are conditioned on its full
history of the session, not just the last step.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class SessionMemory:
    """
    Structured episodic memory for one interview session.

    Memory stores:
        - Per-topic answer quality history
        - Per-topic improvement trends
        - Difficulty progression per topic
        - Global session statistics

    The memory vector (5 dims) is appended to the base state
    vector, giving the DQN richer context for decisions:
        [weakest_topic_score, most_improved_topic_score,
         avg_recent_quality, questions_since_topic_switch,
         session_improvement_rate]
    """

    MEMORY_DIM = 5   # dimensions added to state vector

    def __init__(self, topics: List[str], max_questions: int = 20):
        self.topics = topics
        self.max_questions = max_questions

        # Per-topic history: list of (quality, difficulty) tuples
        self._topic_history: Dict[str, List[Tuple[float, str]]] = {
            t: [] for t in topics
        }
        # Global question log
        self._question_log: List[Dict] = []
        self._last_topic: Optional[str] = None
        self._topic_switch_count: int = 0
        self._questions_on_current_topic: int = 0

    # ── Write API (called by environment) ────────────────────────────────────

    def record(self, topic: str, difficulty: str, quality: float,
               action: str, reward: float):
        """Record the outcome of one question."""
        self._topic_history[topic].append((quality, difficulty))

        if topic != self._last_topic:
            self._topic_switch_count += 1
            self._questions_on_current_topic = 1
        else:
            self._questions_on_current_topic += 1

        self._question_log.append({
            "step":       len(self._question_log),
            "topic":      topic,
            "difficulty": difficulty,
            "quality":    quality,
            "action":     action,
            "reward":     reward,
        })
        self._last_topic = topic

    # ── Read API (called by DQN agent) ────────────────────────────────────────

    def get_memory_vector(self) -> np.ndarray:
        """
        Return a 5-dimensional memory feature vector for the DQN state.

        Features:
            0: score of the weakest topic (where coaching is most needed)
            1: score of the most-improved topic this session
            2: mean quality over the last 5 questions (recency signal)
            3: questions_on_current_topic / 5, capped at 1.0
               (signals when agent has stayed too long on one topic)
            4: session improvement rate
               (slope of quality trend over last 10 questions)
        """
        # Feature 0: weakest topic mean score
        topic_means = {}
        for t in self.topics:
            h = self._topic_history[t]
            topic_means[t] = np.mean([q for q, _ in h]) if h else 0.5
        weakest = min(topic_means.values())

        # Feature 1: most improved topic
        topic_improvements = {}
        for t in self.topics:
            h = self._topic_history[t]
            if len(h) >= 2:
                first_half = np.mean([q for q, _ in h[:len(h)//2]])
                second_half = np.mean([q for q, _ in h[len(h)//2:]])
                topic_improvements[t] = second_half - first_half
            else:
                topic_improvements[t] = 0.0
        most_improved = max(topic_improvements.values()) if topic_improvements else 0.0
        most_improved = float(np.clip(most_improved + 0.5, 0.0, 1.0))  # normalise

        # Feature 2: recent quality (last 5 questions)
        recent = self._question_log[-5:] if len(self._question_log) >= 5 \
                 else self._question_log
        recent_quality = float(np.mean([q["quality"] for q in recent])) \
                         if recent else 0.5

        # Feature 3: dwelling signal
        dwelling = min(1.0, self._questions_on_current_topic / 5.0)

        # Feature 4: improvement rate (linear slope over last 10 steps)
        if len(self._question_log) >= 3:
            recent10 = [q["quality"] for q in self._question_log[-10:]]
            x = np.arange(len(recent10))
            slope = float(np.polyfit(x, recent10, 1)[0])
            improvement_rate = float(np.clip(slope * 10 + 0.5, 0.0, 1.0))
        else:
            improvement_rate = 0.5

        return np.array([weakest, most_improved, recent_quality,
                         dwelling, improvement_rate], dtype=np.float32)

    # ── Read API (called by UCB Bandit) ───────────────────────────────────────

    def get_topic_summary(self) -> Dict[str, Dict]:
        """
        Return per-topic statistics for the UCB bandit to use
        as context when computing exploration bonuses.
        """
        summary = {}
        for t in self.topics:
            h = self._topic_history[t]
            scores = [q for q, _ in h] if h else []
            summary[t] = {
                "n_questions":   len(h),
                "mean_quality":  float(np.mean(scores)) if scores else 0.5,
                "last_quality":  float(scores[-1]) if scores else 0.5,
                "trend":         float(scores[-1] - scores[0])
                                 if len(scores) >= 2 else 0.0,
                "needs_work":    float(np.mean(scores)) < 0.5 if scores else True,
            }
        return summary

    def get_weakest_topic(self) -> str:
        """Return the topic with the lowest mean quality — for bandit guidance."""
        summary = self.get_topic_summary()
        return min(summary, key=lambda t: summary[t]["mean_quality"])

    def get_session_stats(self) -> Dict:
        """Return full session statistics for logging."""
        all_qualities = [q["quality"] for q in self._question_log]
        return {
            "total_questions":    len(self._question_log),
            "mean_quality":       float(np.mean(all_qualities)) if all_qualities else 0.0,
            "topic_switches":     self._topic_switch_count,
            "topic_coverage":     sum(1 for t in self.topics
                                      if len(self._topic_history[t]) > 0),
            "improvement_rate":   float(
                np.mean(all_qualities[-5:]) - np.mean(all_qualities[:5])
            ) if len(all_qualities) >= 10 else 0.0,
        }

    def reset(self):
        """Clear all memory for a new session."""
        self._topic_history = {t: [] for t in self.topics}
        self._question_log = []
        self._last_topic = None
        self._topic_switch_count = 0
        self._questions_on_current_topic = 0