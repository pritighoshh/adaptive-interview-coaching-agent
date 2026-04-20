# Adaptive Interview Coaching Agent (AICA)

**Reinforcement Learning for Agentic AI Systems — Final Project**

An agentic AI system that uses reinforcement learning to personalise and optimise interview coaching sessions. Two RL components work together: a **Double DQN** selects question strategies, and a **UCB1 Bandit** selects which topic to probe. A custom **AnswerEvaluatorTool** provides structured quality scoring that drives both learning components. A formal **AgentMessageBus** handles communication between agents, and a **SessionMemory** module gives the DQN a rich history-aware state.

**Result:** 39.9% improvement over random baseline, statistically validated (p=0.0007, n=30 runs).

---

## Project Structure

```
adaptive-interview-coaching-agent/
├── answer_evaluator.py    # Custom tool: answer quality scorer (3-signal rubric)
├── agent_comms.py         # Agent communication protocol (message bus)
├── session_memory.py      # Episodic memory module (17-dim state vector)
├── dqn_agent.py           # Double DQN agent (pure NumPy)
├── environment.py         # Interview simulation environment (MDP)
├── main.py                # Training loop, statistical validation, visualisations
├── ucb_bandit.py          # UCB1 Bandit + Thompson Sampling baseline
├── architecture.png       # System architecture diagram
├── README.md
├── LICENSE
├── Reinforcement Learning for Agentic AI Systems.pdf
└── results/               # Generated after running main.py
    ├── training_results.png
    ├── before_after.png
    ├── statistical_validation.png
    ├── training_logs.json
    └── dqn_checkpoint.pt.npz
```

---

## Requirements

Python 3.8+ with:

```
pip install numpy matplotlib
```

No PyTorch or TensorFlow required. The DQN is implemented in pure NumPy.

---

## Running the Code

### 1. Test the custom tool standalone

```
python answer_evaluator.py
```

Shows the AnswerEvaluatorTool scoring 4 sample answers with keyword, structure, and length signals. Takes ~1 second.

**Expected output:**

```
AnswerEvaluatorTool — Demo

[1] Topic: algorithms | Difficulty: easy
    Score:     0.653
    Feedback:  Solid answer. Good use of: node, pointer, linked...

[4] Topic: databases | Difficulty: medium
    Score:     0.810
    Feedback:  Strong answer. Good depth (43 words)...

Tool stats: {'total_evaluations': 4, 'active_sessions': 1}
```

### 2. Run full RL training

```
python main.py
```

Trains for 800 episodes (~60–90 seconds). Outputs:

- `results/training_results.png` — 6-panel training dashboard
- `results/before_after.png` — trained vs random comparison
- `results/statistical_validation.png` — 30-run boxplot with Welch t-test
- `results/training_logs.json` — full metrics log
- `results/dqn_checkpoint.pt.npz` — saved model weights

**Expected output:**

```
Ep   50 | ε=0.778 | train_reward=12.03 | eval_score=13.55
Ep  100 | ε=0.606 | train_reward=12.78 | eval_score=13.28
...
Ep  800 | ε=0.050 | train_reward=12.17 | eval_score=12.66

AnswerEvaluatorTool calls: 22400
Message bus total messages: 80 (last episode)

Random  reward: 9.074 ± 3.593
Trained reward: 12.696 ± 4.651
Improvement:    +39.9%
Welch t-stat:   3.376  |  p-value: 0.0007 (significant at α=0.05)
```

---

## System Design

### Four-Module Architecture

| Module | File | Role |
|--------|------|------|
| DQN Agent | `dqn_agent.py` | Selects question strategy using 17-dim state |
| UCB1 Bandit | `ucb_bandit.py` | Selects topic via UCB1 exploration |
| Answer Evaluator | `answer_evaluator.py` | Custom tool: scores candidate answers |
| Session Memory | `session_memory.py` | Tracks full session history, feeds DQN state |
| Message Bus | `agent_comms.py` | Formal communication protocol between agents |
| Environment | `environment.py` | MDP simulation of interview session |

### System Architecture

![AICA Architecture](architecture.png)

### Agent Communication Protocol

At every step, agents exchange typed messages via `AgentMessageBus`:

```
DQN    →  Bandit : TOPIC_REQUEST    (state summary, weakest topic)
Bandit →  DQN   : TOPIC_RESPONSE   (recommended topic + UCB scores)
DQN    →  Bandit : OUTCOME_REPORT   (quality, reward, signal)
DQN    →  Bandit : STRATEGY_SIGNAL  (action taken, exploration hint)
```

80 messages are exchanged per episode (4 per question × 20 questions). The full message log is saved to `results/training_logs.json` for reproducibility and analysis.

### State Space (17 dimensions)

**Base state (12 dims):**

- Topic knowledge scores × 5 (algorithms, system_design, behavioral, databases, ml_concepts)
- Current difficulty one-hot encoding × 3 (easy, medium, hard)
- Candidate confidence, fatigue, session progress, last answer quality

**Memory features (5 dims) — from `SessionMemory`:**

- Weakest topic score (where coaching is most needed)
- Most-improved topic score this session
- Mean quality over last 5 questions (recency signal)
- Topic-dwelling signal (questions on current topic / 5)
- Session improvement rate (linear slope over last 10 steps)

### Action Space (5 actions)

| Action | Description |
|--------|-------------|
| `ask_new_topic` | Switch to a fresh topic at easy difficulty |
| `drill_deeper` | Increase difficulty on current topic |
| `clarify` | Ask candidate to expand their answer |
| `pivot_easier` | Reduce difficulty if candidate is struggling |
| `give_hint_then_ask` | Provide scaffolding then ask (+0.05 knowledge boost) |

### Reward Function

```
r = Q(answer) + 0.3·δ(improvement) − 0.2·δ(fatigue>0.7) − 0.1·δ(topic_repeat≥3)
```

### AnswerEvaluatorTool Scoring

```
score = 0.55 × keyword_coverage
      + 0.25 × structural_quality   # defines, exemplifies, compares, quantifies
      + 0.20 × length_adequacy      # easy≥20w, medium≥40w, hard≥70w
```

---

## Key Results

| Metric | Random baseline | Trained AICA | Change |
|--------|----------------|--------------|--------|
| Total session reward (mean, n=30) | 9.074 ± 3.593 | 12.696 ± 4.651 | **+39.9%** |
| Mean answer quality | 0.484 | 0.642 | **+32.6%** |
| Welch t-statistic | — | — | 3.376 |
| p-value | — | — | **0.0007** |
| Statistical significance | — | — | Yes (α=0.05) |
| AnswerEvaluatorTool calls | — | 22,400 | across training |
| Message bus messages | — | 80/episode | agent communication |

**Emergent strategy:** The trained agent independently discovered that focused drilling in one high-variance topic combined with hint-first scaffolding (`give_hint_then_ask`) outperforms breadth-first coverage — consistent with Vygotsky's zone of proximal development.

---

## DQN Implementation Details

Implemented from scratch in NumPy with all standard components:

- **Double DQN** — online net selects action, target net evaluates it (reduces overestimation bias)
- **Experience replay** — ring buffer of 10,000 transitions, mini-batches of 64
- **Soft target updates** — Polyak averaging with τ=0.01 for training stability
- **Huber loss** — robust to outlier transitions during early high-variance exploration
- **Gradient clipping** — global norm clipped to 1.0
- **ε-greedy decay** — 1.0 → 0.05 over 800 episodes (×0.995 per episode)

---

## Extending the System

To connect a real LLM evaluator instead of the keyword rubric:

```python
# In answer_evaluator.py — only this method needs to change.
# The RL pipeline (DQN + bandit + memory + comms) requires zero changes.
def evaluate(self, answer, topic, difficulty, session_id=None):
    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": f"Score this {difficulty} {topic} answer 0-1: {answer}"}]
    )
    score = float(response.content[0].text.strip())
    # reward shaping, session memory update, feedback generation unchanged
    ...
```

---

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Van Hasselt et al. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Auer et al. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47, 235–256.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Vygotsky, L. S. (1978). *Mind in Society*. Harvard University Press.
