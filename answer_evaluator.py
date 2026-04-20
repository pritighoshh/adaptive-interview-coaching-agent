"""
answer_evaluator.py
Custom Tool: Answer Quality Evaluator

This tool evaluates free-text interview answers using a multi-signal
rubric. It is integrated into the RL pipeline as a custom tool that
the agent calls after each candidate response to obtain a structured
quality score used as the reward signal.

Without this tool, the agent would have no meaningful feedback signal —
it is the bridge between raw candidate text and the numerical reward
the DQN and UCB Bandit learn from.

Usage:
    evaluator = AnswerEvaluatorTool()
    result = evaluator.evaluate(
        answer="A linked list is a data structure where each node...",
        topic="algorithms",
        difficulty="easy"
    )
    print(result.score)        # float in [0, 1]
    print(result.feedback)     # human-readable breakdown
    print(result.reward)       # reward signal passed to RL agent
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Knowledge bases ───────────────────────────────────────────────────────────

# Keywords expected in good answers, by topic and difficulty.
# Weighted: critical terms score more than supporting terms.
KEYWORD_RUBRIC: Dict[str, Dict[str, Dict[str, float]]] = {
    "algorithms": {
        "easy": {
            "node": 0.8, "pointer": 0.8, "linked": 0.7, "stack": 0.9,
            "queue": 0.9, "lifo": 1.0, "fifo": 1.0, "push": 0.7,
            "pop": 0.7, "data structure": 0.8,
        },
        "medium": {
            "dijkstra": 1.0, "shortest path": 1.0, "graph": 0.8,
            "priority queue": 0.9, "weighted": 0.7, "greedy": 0.8,
            "merge sort": 1.0, "o(n log n)": 1.0, "divide": 0.8,
            "conquer": 0.7, "recursive": 0.7,
        },
        "hard": {
            "lru": 1.0, "least recently used": 1.0, "hash map": 0.9,
            "doubly linked": 0.9, "eviction": 0.8, "o(1)": 1.0,
            "np-hard": 1.0, "travelling salesman": 0.9,
            "dynamic programming": 0.8, "approximation": 0.7,
        },
    },
    "system_design": {
        "easy": {
            "rest": 0.9, "api": 0.8, "http": 0.8, "endpoint": 0.7,
            "client": 0.9, "server": 0.9, "request": 0.7, "response": 0.7,
            "stateless": 0.8,
        },
        "medium": {
            "url shortener": 1.0, "hash": 0.9, "redirect": 0.9,
            "database": 0.7, "collision": 0.8, "base62": 0.9,
            "horizontal": 1.0, "vertical": 1.0, "scaling": 0.9,
            "load balancer": 0.9, "throughput": 0.7,
        },
        "hard": {
            "sharding": 0.9, "replication": 0.9, "caching": 0.8,
            "cdn": 0.8, "kafka": 0.8, "message queue": 0.8,
            "consistency": 0.9, "availability": 0.8, "cap theorem": 1.0,
            "distributed": 0.9, "consensus": 0.8,
        },
    },
    "behavioral": {
        "easy": {
            "experience": 0.6, "skill": 0.6, "team": 0.7,
            "role": 0.7, "passion": 0.7, "goal": 0.7, "background": 0.6,
        },
        "medium": {
            "situation": 0.8, "task": 0.7, "action": 0.9, "result": 0.9,
            "conflict": 0.9, "resolved": 0.9, "failure": 0.9,
            "learned": 0.8, "star": 0.7, "outcome": 0.7,
        },
        "hard": {
            "stakeholder": 1.0, "ambiguity": 1.0, "leadership": 0.9,
            "strategy": 0.8, "influence": 0.8, "direction": 0.7,
            "organisation": 0.7, "alignment": 0.8, "decision": 0.7,
        },
    },
    "databases": {
        "easy": {
            "primary key": 1.0, "unique": 0.8, "index": 0.7,
            "sql": 0.9, "nosql": 0.9, "relational": 0.8,
            "table": 0.6, "schema": 0.7,
        },
        "medium": {
            "index": 1.0, "b-tree": 0.9, "performance": 0.8,
            "query": 0.7, "acid": 1.0, "atomicity": 0.9,
            "consistency": 0.9, "isolation": 0.9, "durability": 0.9,
        },
        "hard": {
            "shard": 1.0, "partition": 0.9, "replication": 0.9,
            "horizontal": 0.8, "multi-tenant": 1.0, "tenant": 0.9,
            "row-level": 0.8, "schema": 0.7, "foreign key": 0.7,
        },
    },
    "ml_concepts": {
        "easy": {
            "supervised": 1.0, "label": 0.9, "training": 0.8,
            "predict": 0.7, "bias": 1.0, "variance": 1.0,
            "overfitting": 0.9, "underfitting": 0.9,
        },
        "medium": {
            "gradient descent": 1.0, "loss": 0.9, "derivative": 0.8,
            "learning rate": 0.9, "backpropagation": 0.8,
            "cross-validation": 1.0, "fold": 0.9, "generalisation": 0.8,
            "train": 0.7, "validation": 0.8,
        },
        "hard": {
            "attention": 1.0, "transformer": 1.0, "query": 0.8,
            "key": 0.8, "value": 0.8, "self-attention": 1.0,
            "class imbalance": 1.0, "smote": 0.9, "oversampling": 0.9,
            "undersampling": 0.8, "precision": 0.7, "recall": 0.7,
            "f1": 0.8,
        },
    },
}

# Structural quality signals — topic-agnostic
STRUCTURE_SIGNALS = {
    "defines_term":    (r"\b(is a|refers to|means|defined as|is when)\b", 0.10),
    "gives_example":   (r"\b(for example|such as|e\.g\.|like|consider|instance)\b", 0.12),
    "uses_comparison": (r"\b(whereas|however|unlike|compared to|on the other hand)\b", 0.08),
    "quantifies":      (r"\b(o\(|complexity|percent|time|space|o\s*\()\b", 0.10),
    "explains_why":    (r"\b(because|therefore|since|as a result|thus)\b", 0.08),
    "multi_sentence":  (r"[.!?].*[.!?]", 0.05),   # at least 2 sentences
}

# Difficulty length expectations (min words for full length credit)
LENGTH_THRESHOLDS = {"easy": 20, "medium": 40, "hard": 70}

# Difficulty multipliers — harder questions demand more
DIFFICULTY_MULTIPLIERS = {"easy": 0.9, "medium": 1.0, "hard": 1.1}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    score: float                          # normalised [0, 1]
    reward: float                         # shaped reward for RL (may differ from score)
    keyword_score: float                  # fraction of weighted keywords matched
    structure_score: float                # structural quality signal
    length_score: float                   # answer length adequacy
    keywords_found: List[str]             # which keywords were matched
    keywords_missed: List[str]            # which critical keywords were absent
    feedback: str                         # human-readable coaching feedback
    topic: str
    difficulty: str
    word_count: int


# ── Main tool ─────────────────────────────────────────────────────────────────

class AnswerEvaluatorTool:
    """
    Custom tool that scores a free-text interview answer against a
    multi-signal rubric and returns a structured EvaluationResult.

    Scoring pipeline:
        1. Keyword matching — weighted coverage of domain terms
        2. Structural signals — does the answer explain, exemplify, quantify?
        3. Length adequacy — is the answer substantive enough?
        4. Difficulty adjustment — harder questions are scored more leniently
           on keywords but expect more depth overall.
        5. Reward shaping — adds bonus for improvement over session baseline.

    Integration with RL:
        The `reward` field of EvaluationResult is what the DQN and UCB
        Bandit receive as their feedback signal. The `score` is the raw
        answer quality; `reward` may include session-level shaping.
    """

    def __init__(self):
        self._session_history: Dict[str, List[float]] = {}
        self._call_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        answer: str,
        topic: str,
        difficulty: str,
        session_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a free-text answer and return a structured EvaluationResult.

        Args:
            answer:      The candidate's text response.
            topic:       One of the 5 interview topics.
            difficulty:  'easy', 'medium', or 'hard'.
            session_id:  Optional key for session-level reward shaping.

        Returns:
            EvaluationResult with score, reward, and diagnostic breakdown.
        """
        self._call_count += 1
        answer_lower = answer.lower()
        words = answer_lower.split()
        word_count = len(words)

        # ── Signal 1: Keyword matching ────────────────────────────────────────
        rubric = KEYWORD_RUBRIC.get(topic, {}).get(difficulty, {})
        matched_weighted = 0.0
        total_weight = sum(rubric.values()) if rubric else 1.0
        keywords_found, keywords_missed = [], []

        for kw, weight in rubric.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', answer_lower):
                matched_weighted += weight
                keywords_found.append(kw)
            else:
                keywords_missed.append(kw)

        keyword_score = min(1.0, matched_weighted / max(total_weight * 0.6, 1e-9))

        # ── Signal 2: Structural quality ──────────────────────────────────────
        structure_score = 0.0
        for name, (pattern, weight) in STRUCTURE_SIGNALS.items():
            if re.search(pattern, answer_lower, re.IGNORECASE):
                structure_score += weight
        structure_score = min(1.0, structure_score / 0.45)  # normalise to ~[0,1]

        # ── Signal 3: Length adequacy ─────────────────────────────────────────
        threshold = LENGTH_THRESHOLDS.get(difficulty, 30)
        length_score = min(1.0, word_count / threshold)

        # ── Composite score ───────────────────────────────────────────────────
        raw_score = (
            0.55 * keyword_score +
            0.25 * structure_score +
            0.20 * length_score
        )
        # Apply difficulty multiplier
        mult = DIFFICULTY_MULTIPLIERS.get(difficulty, 1.0)
        score = float(min(1.0, raw_score * mult))

        # ── Reward shaping ────────────────────────────────────────────────────
        reward = score
        if session_id:
            key = f"{session_id}:{topic}"
            history = self._session_history.setdefault(key, [])
            if history and score > sum(history) / len(history):
                reward += 0.15   # improvement bonus
            history.append(score)

        # ── Feedback generation ───────────────────────────────────────────────
        feedback = self._generate_feedback(
            score, keyword_score, structure_score, length_score,
            keywords_found, keywords_missed, topic, difficulty, word_count
        )

        return EvaluationResult(
            score=round(score, 4),
            reward=round(reward, 4),
            keyword_score=round(keyword_score, 4),
            structure_score=round(structure_score, 4),
            length_score=round(length_score, 4),
            keywords_found=keywords_found,
            keywords_missed=keywords_missed[:5],   # top 5 misses
            feedback=feedback,
            topic=topic,
            difficulty=difficulty,
            word_count=word_count,
        )

    def batch_evaluate(
        self,
        answers: List[Dict],
        session_id: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate a list of answers in one call.
        Each dict must have keys: 'answer', 'topic', 'difficulty'.
        """
        return [
            self.evaluate(
                a["answer"], a["topic"], a["difficulty"], session_id
            )
            for a in answers
        ]

    def reset_session(self, session_id: str):
        """Clear session history for a new interview session."""
        keys_to_delete = [k for k in self._session_history if k.startswith(session_id)]
        for k in keys_to_delete:
            del self._session_history[k]

    def get_stats(self) -> Dict:
        return {
            "total_evaluations": self._call_count,
            "active_sessions":   len(set(k.split(":")[0] for k in self._session_history)),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _generate_feedback(
        self, score, kw_score, struct_score, len_score,
        found, missed, topic, difficulty, word_count
    ) -> str:
        lines = []

        # Overall verdict
        if score >= 0.8:
            lines.append("Strong answer.")
        elif score >= 0.6:
            lines.append("Solid answer with room to improve.")
        elif score >= 0.4:
            lines.append("Partial answer — key concepts missing.")
        else:
            lines.append("Weak answer — significant gaps detected.")

        # Length feedback
        threshold = LENGTH_THRESHOLDS.get(difficulty, 30)
        if word_count < threshold * 0.5:
            lines.append(f"Too brief ({word_count} words). Aim for at least {threshold}.")
        elif word_count >= threshold:
            lines.append(f"Good depth ({word_count} words).")

        # Keyword feedback
        if found:
            lines.append(f"Good use of: {', '.join(found[:3])}.")
        if missed and kw_score < 0.7:
            lines.append(f"Consider mentioning: {', '.join(missed[:3])}.")

        # Structure feedback
        if struct_score < 0.4:
            lines.append("Try adding an example or explaining your reasoning.")

        return " ".join(lines)


# ── Demo / standalone test ────────────────────────────────────────────────────

def demo():
    tool = AnswerEvaluatorTool()

    test_cases = [
        {
            "answer": "A linked list is a data structure where each node contains data and a pointer to the next node. Unlike arrays, they allow O(1) insertion at the head.",
            "topic": "algorithms", "difficulty": "easy",
        },
        {
            "answer": "Gradient descent is an optimisation algorithm. You compute the derivative of the loss function with respect to the weights, then update weights by subtracting the learning rate times the gradient.",
            "topic": "ml_concepts", "difficulty": "medium",
        },
        {
            "answer": "I don't know.",
            "topic": "system_design", "difficulty": "hard",
        },
        {
            "answer": "ACID stands for Atomicity, Consistency, Isolation, and Durability. Atomicity means all operations in a transaction succeed or all fail. Consistency ensures the database remains valid after a transaction. Isolation means concurrent transactions don't interfere. Durability means committed data persists even after a crash.",
            "topic": "databases", "difficulty": "medium",
        },
    ]

    print("=" * 60)
    print("  AnswerEvaluatorTool — Demo")
    print("=" * 60)

    for i, tc in enumerate(test_cases, 1):
        result = tool.evaluate(tc["answer"], tc["topic"], tc["difficulty"], session_id="demo")
        print(f"\n[{i}] Topic: {tc['topic']} | Difficulty: {tc['difficulty']}")
        print(f"    Score:     {result.score:.3f}")
        print(f"    Reward:    {result.reward:.3f}")
        print(f"    Keywords:  {result.keyword_score:.2f}  |  "
              f"Structure: {result.structure_score:.2f}  |  "
              f"Length: {result.length_score:.2f}")
        print(f"    Feedback:  {result.feedback}")

    print(f"\nTool stats: {tool.get_stats()}")
    print()


if __name__ == "__main__":
    demo()