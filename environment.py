"""
interview_env.py
Simulated Interview Coaching Environment for RL Agent Training
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from answer_evaluator import AnswerEvaluatorTool

# ── Topic & difficulty definitions ──────────────────────────────────────────
TOPICS = ["algorithms", "system_design", "behavioral", "databases", "ml_concepts"]
DIFFICULTIES = ["easy", "medium", "hard"]

QUESTION_BANK: Dict[str, Dict[str, List[str]]] = {
    "algorithms": {
        "easy":   ["What is a linked list?",
                   "Explain the difference between a stack and a queue."],
        "medium": ["Describe Dijkstra's algorithm.",
                   "What is the time complexity of merge sort?"],
        "hard":   ["Design an LRU cache from scratch.",
                   "Solve the travelling salesman problem optimally."],
    },
    "system_design": {
        "easy":   ["What is a REST API?",
                   "Explain client-server architecture."],
        "medium": ["How would you design a URL shortener?",
                   "Describe horizontal vs vertical scaling."],
        "hard":   ["Design Twitter's real-time feed at 100M users.",
                   "How would you build a distributed key-value store?"],
    },
    "behavioral": {
        "easy":   ["Tell me about yourself.",
                   "Why do you want this role?"],
        "medium": ["Describe a time you resolved a conflict.",
                   "Give an example of handling failure."],
        "hard":   ["Describe leading a project under ambiguity with stakeholder pressure.",
                   "Tell me about a time you changed an organisation's direction."],
    },
    "databases": {
        "easy":   ["What is a primary key?",
                   "Explain SQL vs NoSQL."],
        "medium": ["What are database indexes and when should you use them?",
                   "Explain ACID properties."],
        "hard":   ["How would you shard a PostgreSQL database at scale?",
                   "Design a schema for a multi-tenant SaaS application."],
    },
    "ml_concepts": {
        "easy":   ["What is supervised learning?",
                   "Explain bias vs variance."],
        "medium": ["How does gradient descent work?",
                   "What is cross-validation?"],
        "hard":   ["Explain the attention mechanism in transformers.",
                   "How would you handle severe class imbalance in production?"],
    },
}

# Action types the DQN can choose
QUESTION_ACTIONS = [
    "ask_new_topic",        # 0 – switch to a fresh topic
    "drill_deeper",         # 1 – follow up on the current topic, harder
    "clarify",              # 2 – ask the candidate to clarify their answer
    "pivot_easier",         # 3 – lower difficulty if candidate is struggling
    "give_hint_then_ask",   # 4 – scaffold before asking
]
N_ACTIONS = len(QUESTION_ACTIONS)


@dataclass
class CandidateProfile:
    """Tracks simulated candidate knowledge across topics."""
    knowledge: Dict[str, float] = field(default_factory=lambda: {
        t: random.uniform(0.2, 0.8) for t in TOPICS
    })
    confidence: float = field(default_factory=lambda: random.uniform(0.3, 0.9))
    fatigue: float = 0.0          # increases each turn
    questions_asked: int = 0


class InterviewEnvironment:
    """
    Simulates an interview session.

    State vector (dim=12):
        [topic_scores × 5, current_difficulty_enc × 3, confidence, fatigue,
         questions_asked_norm, last_answer_quality]

    Reward:
        +answer_quality             (did the candidate do well?)
        +0.3 if improvement         (better than last turn on same topic?)
        -0.2 if fatigue > 0.7       (agent shouldn't exhaust candidate)
        -0.1 if repeated topic      (encourage breadth)
    """

    STATE_DIM = 12
    _evaluator: "AnswerEvaluatorTool | None" = None   # shared across instances

    def __init__(self, max_questions: int = 20, use_evaluator: bool = True):
        self.max_questions = max_questions
        self.use_evaluator = use_evaluator
        if use_evaluator and InterviewEnvironment._evaluator is None:
            InterviewEnvironment._evaluator = AnswerEvaluatorTool()
        self.reset()

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.candidate = CandidateProfile()
        self.current_topic = random.choice(TOPICS)
        self.current_difficulty = "easy"
        self.last_answer_quality = 0.5
        self.topic_history: List[str] = []
        self.score_history: List[float] = []
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply action, simulate candidate response, return (state, reward, done, info)."""
        assert not self.done, "Episode finished – call reset()."

        # ── Apply action ────────────────────────────────────────────────────
        self._apply_action(action)

        # ── Simulate candidate answer ───────────────────────────────────────
        answer_quality = self._simulate_answer()

        # ── Compute reward ──────────────────────────────────────────────────
        reward = self._compute_reward(answer_quality)

        # ── Update state ────────────────────────────────────────────────────
        self.topic_history.append(self.current_topic)
        self.score_history.append(answer_quality)
        self.last_answer_quality = answer_quality
        self.candidate.questions_asked += 1
        self.candidate.fatigue = min(1.0, self.candidate.fatigue + 0.05)

        self.done = self.candidate.questions_asked >= self.max_questions
        info = {
            "topic": self.current_topic,
            "difficulty": self.current_difficulty,
            "answer_quality": answer_quality,
            "action_name": QUESTION_ACTIONS[action],
        }
        return self._get_state(), reward, self.done, info

    def get_available_question(self) -> str:
        bank = QUESTION_BANK[self.current_topic][self.current_difficulty]
        return random.choice(bank)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        diff_order = DIFFICULTIES  # ["easy", "medium", "hard"]
        idx = diff_order.index(self.current_difficulty)

        if action == 0:   # ask_new_topic
            others = [t for t in TOPICS if t != self.current_topic]
            self.current_topic = random.choice(others)
            self.current_difficulty = "easy"

        elif action == 1: # drill_deeper
            self.current_difficulty = diff_order[min(idx + 1, 2)]

        elif action == 2: # clarify – same topic/difficulty
            pass

        elif action == 3: # pivot_easier
            self.current_difficulty = diff_order[max(idx - 1, 0)]

        elif action == 4: # give_hint_then_ask – slight boost to knowledge
            self.candidate.knowledge[self.current_topic] = min(
                1.0, self.candidate.knowledge[self.current_topic] + 0.05
            )

    def _simulate_answer(self) -> float:
        """
        Generate a synthetic candidate answer and score it via the
        AnswerEvaluatorTool (custom tool).  The tool's structured
        EvaluationResult becomes the ground-truth reward signal.
        """
        base = self.candidate.knowledge[self.current_topic]
        diff_penalty = {"easy": 0.0, "medium": 0.1, "hard": 0.25}
        noise = np.random.normal(0, 0.08)
        fatigue_penalty = self.candidate.fatigue * 0.15
        raw_quality = float(np.clip(
            base - diff_penalty[self.current_difficulty] - fatigue_penalty + noise,
            0.0, 1.0
        ))

        # ── Custom tool call ─────────────────────────────────────────────────
        # Build a synthetic answer whose richness scales with raw_quality.
        # This lets the evaluator's keyword/structure signals give a
        # second opinion that slightly adjusts the raw quality estimate,
        # mimicking the role a real LLM evaluator would play.
        if self.use_evaluator and self._evaluator is not None:
            synthetic_answer = self._build_synthetic_answer(raw_quality)
            result = self._evaluator.evaluate(
                answer=synthetic_answer,
                topic=self.current_topic,
                difficulty=self.current_difficulty,
                session_id=id(self),   # session-level improvement shaping
            )
            # Blend raw simulation with tool's score (70/30 weighting)
            quality = 0.70 * raw_quality + 0.30 * result.score
        else:
            quality = raw_quality

        # Candidate improves slightly on repeated practice
        self.candidate.knowledge[self.current_topic] = min(
            1.0, self.candidate.knowledge[self.current_topic] + quality * 0.02
        )
        return float(quality)

    def _build_synthetic_answer(self, quality: float) -> str:
        """
        Construct a plausible synthetic answer string whose keyword
        density correlates with quality, so the AnswerEvaluatorTool
        produces a meaningful score signal.
        """
        from answer_evaluator import KEYWORD_RUBRIC, LENGTH_THRESHOLDS
        rubric   = KEYWORD_RUBRIC.get(self.current_topic, {}).get(self.current_difficulty, {})
        keywords = sorted(rubric.keys(), key=lambda k: rubric[k], reverse=True)

        # Include top-weighted keywords proportional to quality
        n_include = max(1, int(len(keywords) * quality))
        chosen    = keywords[:n_include]

        threshold = LENGTH_THRESHOLDS.get(self.current_difficulty, 30)
        word_target = int(threshold * min(1.0, quality + 0.2))

        # Build a minimal but plausible sentence using chosen keywords
        answer_parts = ["The answer involves"]
        answer_parts += chosen
        if quality > 0.5:
            answer_parts += ["for example,", "this is important because", "therefore"]
        if quality > 0.7:
            answer_parts += ["compared to alternatives,", "the result is", "as a consequence"]

        answer = " ".join(answer_parts)
        # Pad to approximate word target with filler if needed
        while len(answer.split()) < word_target:
            answer += " this approach is used in practice"
        return answer

    def _compute_reward(self, answer_quality: float) -> float:
        reward = answer_quality

        # Bonus for measurable improvement on this topic
        prior_scores = [s for s, t in zip(self.score_history, self.topic_history)
                        if t == self.current_topic]
        if prior_scores and answer_quality > np.mean(prior_scores):
            reward += 0.3

        # Penalty for over-fatiguing the candidate
        if self.candidate.fatigue > 0.7:
            reward -= 0.2

        # Penalty for staying on the same topic more than 3 turns
        recent = self.topic_history[-3:] if len(self.topic_history) >= 3 else self.topic_history
        if recent and all(t == self.current_topic for t in recent):
            reward -= 0.1

        return float(reward)

    def _get_state(self) -> np.ndarray:
        topic_scores = np.array([self.candidate.knowledge[t] for t in TOPICS])
        diff_enc = np.array({
            "easy": [1, 0, 0], "medium": [0, 1, 0], "hard": [0, 0, 1]
        }[self.current_difficulty], dtype=float)
        misc = np.array([
            self.candidate.confidence,
            self.candidate.fatigue,
            self.candidate.questions_asked / self.max_questions,
            self.last_answer_quality,
        ])
        return np.concatenate([topic_scores, diff_enc, misc]).astype(np.float32)