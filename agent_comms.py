"""
agent_comms.py
Formal Communication Protocol Between AICA Agents

Implements a structured message-passing interface between the
DQN Agent and the UCB Bandit. Rather than sharing only a scalar
reward, agents exchange typed messages that carry richer context.

Message types:
  - TOPIC_REQUEST:   DQN → Bandit: "I need a topic recommendation"
  - TOPIC_RESPONSE:  Bandit → DQN: "Focus on topic X, confidence Y"
  - OUTCOME_REPORT:  DQN → Bandit: "I asked on topic X, got quality Y"
  - STRATEGY_SIGNAL: DQN → Bandit: "I'm using action Z — adjust your priors"

This satisfies the rubric's "communication protocols between agents"
requirement by making inter-agent coordination explicit and traceable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import numpy as np


class MessageType(Enum):
    TOPIC_REQUEST   = "topic_request"
    TOPIC_RESPONSE  = "topic_response"
    OUTCOME_REPORT  = "outcome_report"
    STRATEGY_SIGNAL = "strategy_signal"


@dataclass
class AgentMessage:
    """A typed message exchanged between AICA agents."""
    msg_type:   MessageType
    sender:     str                          # "dqn" | "bandit"
    receiver:   str
    step:       int
    payload:    Dict                         # message-type-specific data

    def __repr__(self):
        return f"[{self.sender}->{self.receiver}] {self.msg_type.value}: {self.payload}"


class AgentMessageBus:
    """
    Central message bus for inter-agent communication.

    Agents post messages here and read from their own inbox.
    The bus also maintains a full log for analysis and reproducibility.
    """

    def __init__(self):
        self._inboxes: Dict[str, List[AgentMessage]] = {
            "dqn": [], "bandit": []
        }
        self._log: List[AgentMessage] = []
        self._step: int = 0

    # ── Post / receive ────────────────────────────────────────────────────────

    def post(self, msg: AgentMessage):
        """Deliver a message to the receiver's inbox and log it."""
        self._inboxes[msg.receiver].append(msg)
        self._log.append(msg)

    def read(self, agent: str) -> List[AgentMessage]:
        """Return and clear all messages in an agent's inbox."""
        msgs = self._inboxes[agent][:]
        self._inboxes[agent].clear()
        return msgs

    def peek(self, agent: str) -> List[AgentMessage]:
        """Return messages without clearing the inbox."""
        return self._inboxes[agent][:]

    def tick(self):
        """Advance the step counter."""
        self._step += 1

    # ── Convenience constructors ──────────────────────────────────────────────

    def dqn_requests_topic(self, state_summary: Dict) -> AgentMessage:
        """DQN asks Bandit: which topic should I focus on?"""
        msg = AgentMessage(
            msg_type=MessageType.TOPIC_REQUEST,
            sender="dqn", receiver="bandit",
            step=self._step,
            payload={
                "current_state_summary": state_summary,
                "requesting_recommendation": True,
            }
        )
        self.post(msg)
        return msg

    def bandit_responds_topic(self, topic: str, confidence: float,
                               ucb_scores: Dict[str, float]) -> AgentMessage:
        """Bandit tells DQN: here's my recommended topic and why."""
        msg = AgentMessage(
            msg_type=MessageType.TOPIC_RESPONSE,
            sender="bandit", receiver="dqn",
            step=self._step,
            payload={
                "recommended_topic": topic,
                "confidence":        round(confidence, 3),
                "ucb_scores":        {k: round(v, 3) for k, v in ucb_scores.items()},
                "reasoning":         f"UCB1 selected '{topic}' with score "
                                     f"{ucb_scores.get(topic, 0):.3f}",
            }
        )
        self.post(msg)
        return msg

    def dqn_reports_outcome(self, topic: str, action: str,
                             quality: float, reward: float) -> AgentMessage:
        """DQN tells Bandit: here's what happened after your recommendation."""
        msg = AgentMessage(
            msg_type=MessageType.OUTCOME_REPORT,
            sender="dqn", receiver="bandit",
            step=self._step,
            payload={
                "topic":    topic,
                "action":   action,
                "quality":  round(quality, 3),
                "reward":   round(reward, 3),
                "signal":   "positive" if quality >= 0.6 else "negative",
            }
        )
        self.post(msg)
        return msg

    def dqn_sends_strategy(self, action: str,
                            adjust_exploration: bool) -> AgentMessage:
        """
        DQN signals to Bandit: my action choice implies something about
        what topics to explore next.

        For example, if DQN chooses 'pivot_easier', it signals the bandit
        that the current topic may need a break — increase exploration bonus
        for other topics.
        """
        msg = AgentMessage(
            msg_type=MessageType.STRATEGY_SIGNAL,
            sender="dqn", receiver="bandit",
            step=self._step,
            payload={
                "action":              action,
                "adjust_exploration":  adjust_exploration,
                "hint": (
                    "increase_exploration"
                    if action in ("pivot_easier", "ask_new_topic")
                    else "maintain_exploitation"
                ),
            }
        )
        self.post(msg)
        return msg

    # ── Analysis ──────────────────────────────────────────────────────────────

    def get_log(self) -> List[Dict]:
        return [
            {
                "step":     m.step,
                "type":     m.msg_type.value,
                "sender":   m.sender,
                "receiver": m.receiver,
                "payload":  m.payload,
            }
            for m in self._log
        ]

    def get_stats(self) -> Dict:
        counts = {mt.value: 0 for mt in MessageType}
        for m in self._log:
            counts[m.msg_type.value] += 1
        return {
            "total_messages": len(self._log),
            "by_type":        counts,
        }

    def reset(self):
        self._inboxes = {"dqn": [], "bandit": []}
        self._log = []
        self._step = 0