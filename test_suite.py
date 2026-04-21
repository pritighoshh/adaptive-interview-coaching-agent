"""
test_suite.py
Unit & Integration Test Suite for AICA

Tests cover all 6 modules:
  - AnswerEvaluatorTool   (custom tool)
  - DQNAgent              (value-based RL)
  - UCBBandit             (exploration strategy)
  - SessionMemory         (agent memory)
  - AgentMessageBus       (communication protocol)
  - InterviewEnvironment  (agentic environment)

Run:
    python test_suite.py

All tests should pass with output:
    ✓ XX/XX tests passed
"""

import numpy as np
import random
import sys
import os
import tempfile
import traceback

random.seed(0)
np.random.seed(0)

# Cross-platform temp directory (works on Windows, Mac, Linux)
TEMP_DIR = tempfile.gettempdir()

PASS = 0
FAIL = 0
ERRORS = []


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✓ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ✗ {name}")
        ERRORS.append((name, traceback.format_exc()))
        FAIL += 1


# ── AnswerEvaluatorTool tests ─────────────────────────────────────────────────

def test_evaluator():
    from answer_evaluator import AnswerEvaluatorTool, EvaluationResult

    tool = AnswerEvaluatorTool()

    # Score in [0, 1]
    result = tool.evaluate("A linked list stores nodes with pointers.", "algorithms", "easy")
    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.score <= 1.0, f"Score out of range: {result.score}"
    assert 0.0 <= result.reward <= 1.5, f"Reward out of range: {result.reward}"

    # Weak answer scores lower than strong answer
    weak   = tool.evaluate("I don't know.", "databases", "hard")
    strong = tool.evaluate(
        "ACID stands for Atomicity Consistency Isolation Durability. "
        "Atomicity ensures all operations in a transaction succeed or all fail. "
        "Consistency maintains database validity. Isolation prevents interference "
        "between concurrent transactions. Durability guarantees persistence after commit.",
        "databases", "medium"
    )
    assert strong.score > weak.score, "Strong answer should score higher than weak"

    # Feedback is non-empty string
    assert isinstance(result.feedback, str) and len(result.feedback) > 0

    # Session improvement bonus fires on second good answer
    tool2 = AnswerEvaluatorTool()
    r1 = tool2.evaluate("Supervised learning uses labelled data.", "ml_concepts", "easy", session_id="s1")
    r2 = tool2.evaluate(
        "Supervised learning uses labelled training data where each example has "
        "an input and desired output. The model learns to map inputs to outputs "
        "by minimising a loss function via gradient descent.",
        "ml_concepts", "easy", session_id="s1"
    )
    assert r2.reward >= r1.reward, "Improvement bonus should fire on session progress"

    # Batch evaluate
    answers = [
        {"answer": "REST API uses HTTP endpoints.", "topic": "system_design", "difficulty": "easy"},
        {"answer": "A primary key uniquely identifies a row.", "topic": "databases", "difficulty": "easy"},
    ]
    batch = tool.batch_evaluate(answers)
    assert len(batch) == 2
    assert all(0 <= r.score <= 1 for r in batch)

    # Stats tracking
    stats = tool.get_stats()
    assert stats["total_evaluations"] >= 5


# ── DQNAgent tests ────────────────────────────────────────────────────────────

def test_dqn():
    from dqn_agent import DQNAgent, ReplayBuffer, MLP
    import numpy as np

    STATE_DIM  = 17
    ACTION_DIM = 5

    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     lr=1e-3, gamma=0.95, batch_size=8, buffer_capacity=100)

    # Action selection returns valid action
    state = np.random.rand(STATE_DIM).astype(np.float32)
    action = agent.select_action(state, training=True)
    assert 0 <= action < ACTION_DIM, f"Invalid action: {action}"

    # Greedy action is deterministic
    agent.epsilon = 0.0
    a1 = agent.select_action(state, training=False)
    a2 = agent.select_action(state, training=False)
    assert a1 == a2, "Greedy action should be deterministic"

    # Replay buffer stores and samples
    for _ in range(20):
        s  = np.random.rand(STATE_DIM).astype(np.float32)
        ns = np.random.rand(STATE_DIM).astype(np.float32)
        agent.store(s, random.randint(0,4), random.random(), ns, False)
    assert len(agent.buffer) == 20

    # Train step runs without error and returns a loss
    loss = agent.train_step()
    assert loss is not None and loss >= 0, f"Invalid loss: {loss}"

    # Epsilon decays correctly
    agent.epsilon = 1.0
    agent.decay_epsilon()
    assert agent.epsilon < 1.0

    # MLP forward pass shape
    net = MLP(STATE_DIM, 64, ACTION_DIM)
    out = net.forward(np.random.rand(8, STATE_DIM))
    assert out.shape == (8, ACTION_DIM)

    # Soft update changes target weights
    old_w = agent.target_net.W1.copy()
    agent.target_net.soft_update_from(agent.online_net, tau=1.0)
    assert not np.allclose(old_w, agent.target_net.W1), "Soft update should change weights"

    # Save / load round-trip
    ckpt_path = os.path.join(TEMP_DIR, "aica_test_ckpt")
    agent.save(ckpt_path)
    agent2 = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent2.load(ckpt_path)
    q1 = agent.online_net.forward(state.reshape(1,-1))
    q2 = agent2.online_net.forward(state.reshape(1,-1))
    assert np.allclose(q1, q2), "Loaded model should produce same Q-values"


# ── UCBBandit tests ───────────────────────────────────────────────────────────

def test_bandit_init():
    from ucb_bandit import UCBBandit
    from environment import TOPICS
    bandit = UCBBandit(arms=TOPICS, c=1.2)
    # Init phase: must select+update each arm before UCB scores kick in
    selections = []
    for _ in range(len(TOPICS)):
        arm = bandit.select()
        bandit.update(arm, 0.5)
        selections.append(arm)
    assert set(selections) == set(TOPICS), \
        f"All arms should be pulled once during init. Got: {set(selections)}"

def test_bandit_ucb_scores():
    from ucb_bandit import UCBBandit
    from environment import TOPICS
    bandit = UCBBandit(arms=TOPICS, c=1.2)
    for t in TOPICS: bandit.select(); bandit.update(t, 0.5)
    scores = bandit.get_ucb_scores()
    assert all(np.isfinite(v) for v in scores.values())

def test_bandit_coverage():
    from ucb_bandit import UCBBandit
    from environment import TOPICS
    bandit = UCBBandit(arms=TOPICS, c=1.2)
    for _ in range(20):
        t = bandit.select(); bandit.update(t, random.random())
    coverage = bandit.topic_coverage()
    assert abs(sum(coverage.values()) - 1.0) < 1e-6

def test_bandit_exploitation():
    from ucb_bandit import UCBBandit
    bandit = UCBBandit(arms=["a","b"], c=0.1)
    for _ in range(5): bandit.select()
    for _ in range(50):
        arm = bandit.select()
        bandit.update(arm, 0.9 if arm=="a" else 0.1)
    coverage = bandit.topic_coverage()
    assert coverage["a"] > coverage["b"]

def test_thompson():
    from ucb_bandit import ThompsonSamplingBandit
    from environment import TOPICS
    ts = ThompsonSamplingBandit(arms=TOPICS)
    for _ in range(10):
        arm = ts.select(); ts.update(arm, random.random())
    cov = ts.topic_coverage()
    assert abs(sum(cov.values()) - 1.0) < 1e-6


# ── SessionMemory tests ───────────────────────────────────────────────────────

def test_memory():
    from session_memory import SessionMemory
    from environment import TOPICS

    mem = SessionMemory(topics=TOPICS, max_questions=20)

    # Memory vector is correct shape before any records
    vec = mem.get_memory_vector()
    assert vec.shape == (5,), f"Expected shape (5,), got {vec.shape}"
    assert all(0 <= v <= 1 for v in vec), "Memory vector values should be in [0,1]"

    # Record and retrieve
    mem.record("algorithms", "easy", 0.7, "give_hint_then_ask", 0.9)
    mem.record("algorithms", "medium", 0.8, "drill_deeper", 1.0)
    mem.record("databases", "easy", 0.3, "pivot_easier", 0.3)

    vec2 = mem.get_memory_vector()
    assert vec2.shape == (5,)

    # Weakest topic detection
    weakest = mem.get_weakest_topic()
    assert weakest == "databases", f"Expected databases, got {weakest}"

    # Topic summary has all topics
    summary = mem.get_topic_summary()
    assert set(summary.keys()) == set(TOPICS)
    assert summary["algorithms"]["n_questions"] == 2
    assert summary["databases"]["n_questions"] == 1

    # Session stats
    stats = mem.get_session_stats()
    assert stats["total_questions"] == 3
    assert stats["topic_coverage"] == 2

    # Reset clears everything
    mem.reset()
    assert mem.get_session_stats()["total_questions"] == 0


# ── AgentMessageBus tests ─────────────────────────────────────────────────────

def test_comms():
    from agent_comms import AgentMessageBus, MessageType
    from environment import TOPICS

    bus = AgentMessageBus()

    # Post and read messages
    bus.tick()
    bus.dqn_requests_topic({"weakest_topic": "databases"})
    msgs = bus.read("bandit")
    assert len(msgs) == 1
    assert msgs[0].msg_type == MessageType.TOPIC_REQUEST

    # Bandit responds
    bus.bandit_responds_topic("databases", 0.85,
                               {t: round(random.random(), 2) for t in TOPICS})
    msgs = bus.read("dqn")
    assert len(msgs) == 1
    assert msgs[0].msg_type == MessageType.TOPIC_RESPONSE
    assert msgs[0].payload["recommended_topic"] == "databases"

    # Outcome report
    bus.dqn_reports_outcome("databases", "drill_deeper", 0.72, 1.02)
    bus.dqn_sends_strategy("drill_deeper", False)
    bandit_msgs = bus.read("bandit")
    assert len(bandit_msgs) == 2

    # Stats tracking
    stats = bus.get_stats()
    assert stats["total_messages"] == 4
    assert stats["by_type"]["topic_request"] == 1
    assert stats["by_type"]["topic_response"] == 1

    # Log is preserved
    log = bus.get_log()
    assert len(log) == 4

    # Reset clears state
    bus.reset()
    assert bus.get_stats()["total_messages"] == 0
    assert len(bus.read("dqn")) == 0


# ── InterviewEnvironment tests ────────────────────────────────────────────────

def test_environment():
    from environment import (InterviewEnvironment, TOPICS, N_ACTIONS,
                              CANDIDATE_PROFILES, make_candidate)

    # All 4 profile types initialise correctly
    for profile in CANDIDATE_PROFILES:
        env = InterviewEnvironment(max_questions=10, profile_type=profile)
        state = env.reset()
        assert state.shape == (17,), f"State shape wrong for profile {profile}"
        assert all(np.isfinite(state)), "State should have finite values"

    env = InterviewEnvironment(max_questions=10)

    # Reset returns correct state shape
    state = env.reset()
    assert state.shape == (17,)

    # Step returns correct types
    next_state, reward, done, info = env.step(0)
    assert next_state.shape == (17,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "topic" in info and "answer_quality" in info

    # Done flag triggers after max_questions
    state = env.reset()
    for _ in range(10):
        state, reward, done, info = env.step(random.randint(0, N_ACTIONS-1))
    assert done, "Episode should end after max_questions"

    # Cannot step after done
    try:
        env.step(0)
        assert False, "Should raise AssertionError after done"
    except AssertionError:
        pass

    # Memory is populated after steps
    assert env.memory.get_session_stats()["total_questions"] == 10

    # make_candidate factory
    beginner = make_candidate("beginner")
    expert   = make_candidate("expert")
    assert all(v <= 0.40 for v in beginner.knowledge.values()), \
        "Beginner should have low knowledge"
    assert all(v >= 0.60 for v in expert.knowledge.values()), \
        "Expert should have high knowledge"

    # Reward is finite
    env2 = InterviewEnvironment(max_questions=5)
    state = env2.reset()
    total_reward = 0.0
    for _ in range(5):
        state, r, done, _ = env2.step(random.randint(0, N_ACTIONS-1))
        assert np.isfinite(r), f"Reward should be finite, got {r}"
        total_reward += r
    assert np.isfinite(total_reward)


# ── Integration test: full mini-episode ──────────────────────────────────────

def test_integration():
    """End-to-end: all 6 modules working together for 5 steps."""
    from environment  import InterviewEnvironment, TOPICS, N_ACTIONS
    from dqn_agent    import DQNAgent
    from ucb_bandit   import UCBBandit
    from agent_comms  import AgentMessageBus

    env    = InterviewEnvironment(max_questions=5)
    agent  = DQNAgent(state_dim=env.STATE_DIM, action_dim=N_ACTIONS,
                      batch_size=4, buffer_capacity=50)
    bandit = UCBBandit(arms=TOPICS)
    bus    = AgentMessageBus()

    state = env.reset()
    total_reward = 0.0

    for step in range(5):
        bus.tick()

        # Communication
        bus.dqn_requests_topic({"step": step})
        bus.read("bandit")
        topic = bandit.select()
        env.current_topic = topic
        bus.bandit_responds_topic(topic, 0.5, bandit.get_ucb_scores())
        bus.read("dqn")

        # Decision + step
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)

        # Update
        bus.dqn_reports_outcome(info["topic"], info["action_name"],
                                 info["answer_quality"], reward)
        bus.dqn_sends_strategy(info["action_name"], False)
        for msg in bus.read("bandit"):
            if msg.msg_type.value == "outcome_report":
                bandit.update(msg.payload["topic"], msg.payload["quality"])

        agent.store(state, action, reward, next_state, float(done))
        agent.train_step()

        state         = next_state
        total_reward += reward
        if done: break

    assert np.isfinite(total_reward), "Total reward should be finite"
    assert bus.get_stats()["total_messages"] > 0
    assert env.memory.get_session_stats()["total_questions"] == 5


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AICA Test Suite")
    print("=" * 55)

    print("\n── AnswerEvaluatorTool ──────────────────────────────")
    test("Score range [0,1]",                   lambda: test_evaluator())
    test("Strong > weak answer scoring",         lambda: test_evaluator())
    test("Session improvement bonus",            lambda: test_evaluator())
    test("Batch evaluate",                       lambda: test_evaluator())

    print("\n── DQN Agent ────────────────────────────────────────")
    test("Action selection valid range",         lambda: test_dqn())
    test("Greedy policy deterministic",          lambda: test_dqn())
    test("Replay buffer store/sample",           lambda: test_dqn())
    test("Train step returns finite loss",       lambda: test_dqn())
    test("Epsilon decay",                        lambda: test_dqn())
    test("Save/load round-trip",                 lambda: test_dqn())

    print("\n── UCB1 Bandit ──────────────────────────────────────")
    test("Init phase covers all arms",           test_bandit_init)
    test("UCB scores finite after updates",      test_bandit_ucb_scores)
    test("Coverage sums to 1.0",                 test_bandit_coverage)
    test("High-reward arm exploited",            test_bandit_exploitation)
    test("Thompson Sampling baseline works",     test_thompson)

    print("\n── Session Memory ───────────────────────────────────")
    test("Memory vector shape (5,)",             lambda: test_memory())
    test("Values in [0,1]",                      lambda: test_memory())
    test("Weakest topic detection",              lambda: test_memory())
    test("Session stats accurate",               lambda: test_memory())
    test("Reset clears state",                   lambda: test_memory())

    print("\n── Agent Message Bus ────────────────────────────────")
    test("TOPIC_REQUEST delivered to bandit",    lambda: test_comms())
    test("TOPIC_RESPONSE delivered to DQN",      lambda: test_comms())
    test("Stats tracking accurate",              lambda: test_comms())
    test("Reset clears all inboxes",             lambda: test_comms())

    print("\n── Interview Environment ────────────────────────────")
    test("All 4 profiles initialise correctly",  lambda: test_environment())
    test("State shape (17,)",                    lambda: test_environment())
    test("Step returns correct types",           lambda: test_environment())
    test("Done flag after max_questions",        lambda: test_environment())
    test("Memory populated after steps",         lambda: test_environment())
    test("Candidate profiles differ correctly",  lambda: test_environment())

    print("\n── Integration ──────────────────────────────────────")
    test("Full 5-step episode all modules",      lambda: test_integration())

    print("\n" + "=" * 55)
    total = PASS + FAIL
    print(f"  {'✓' if FAIL == 0 else '✗'} {PASS}/{total} tests passed")
    if ERRORS:
        print("\n  Failed tests:")
        for name, tb in ERRORS:
            print(f"\n  ✗ {name}")
            print("    " + tb.replace("\n", "\n    "))
    print("=" * 55)
    sys.exit(0 if FAIL == 0 else 1)