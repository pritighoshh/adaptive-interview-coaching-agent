"""
main.py
Training pipeline for the Adaptive Interview Coaching Agent.

Pipeline:
  1. UCB Bandit selects the TOPIC to probe next.
  2. DQN Agent selects the QUESTION TYPE (action) given current state.
  3. Environment simulates candidate answer and returns reward.
  4. Both components are updated from observed reward.
  5. After training, evaluation runs show before/after performance.

Run:
    python main.py
"""

import numpy as np
import random
import os
import json
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from environment import InterviewEnvironment, TOPICS, N_ACTIONS, QUESTION_ACTIONS
from dqn_agent    import DQNAgent
from ucb_bandit   import UCBBandit, ThompsonSamplingBandit

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPISODES          = 500       # training episodes
MAX_QUESTIONS     = 20        # questions per episode
EVAL_EVERY        = 50        # evaluate every N episodes
EVAL_EPISODES     = 20        # episodes per evaluation run
LR                = 1e-3
GAMMA             = 0.95
EPSILON_START     = 1.0
EPSILON_END       = 0.05
EPSILON_DECAY     = 0.995
BATCH_SIZE        = 64
UCB_C             = 1.2       # exploration coefficient

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Adaptive Interview Coaching Agent — RL Training")
    print("=" * 60)

    env    = InterviewEnvironment(max_questions=MAX_QUESTIONS)
    agent  = DQNAgent(
        state_dim      = env.STATE_DIM,
        action_dim     = N_ACTIONS,
        lr             = LR,
        gamma          = GAMMA,
        epsilon_start  = EPSILON_START,
        epsilon_end    = EPSILON_END,
        epsilon_decay  = EPSILON_DECAY,
        batch_size     = BATCH_SIZE,
    )
    bandit = UCBBandit(arms=TOPICS, c=UCB_C)

    # Logging
    episode_rewards:   list = []
    episode_avg_q:     list = []
    eval_scores:       list = []
    eval_episodes_log: list = []
    action_counts      = defaultdict(int)

    print(f"\nTraining for {EPISODES} episodes …\n")

    for ep in range(1, EPISODES + 1):
        state   = env.reset()
        ep_reward = 0.0

        for step in range(MAX_QUESTIONS):
            # 1. Bandit picks topic
            suggested_topic = bandit.select()
            env.current_topic = suggested_topic

            # 2. DQN picks action type
            action = agent.select_action(state, training=True)
            action_counts[QUESTION_ACTIONS[action]] += 1

            # 3. Step environment
            next_state, reward, done, info = env.step(action)

            # 4. Store transition and train DQN
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()

            # 5. Update bandit with observed reward
            bandit.update(info["topic"], info["answer_quality"])

            state      = next_state
            ep_reward += reward

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)

        # Mean Q-value tracking
        if agent.q_values_log:
            episode_avg_q.append(np.mean(agent.q_values_log[-MAX_QUESTIONS:]))

        # Periodic evaluation
        if ep % EVAL_EVERY == 0:
            score = evaluate(agent, bandit, n_episodes=EVAL_EPISODES)
            eval_scores.append(score)
            eval_episodes_log.append(ep)
            print(f"  Ep {ep:>4d} | ε={agent.epsilon:.3f} | "
                  f"train_reward={np.mean(episode_rewards[-50:]):.3f} | "
                  f"eval_score={score:.3f}")

    print("\nTraining complete.")
    agent.save(os.path.join(OUTPUT_DIR, "dqn_checkpoint.pt"))

    # Collect evaluator tool stats
    from answer_evaluator import AnswerEvaluatorTool
    from environment import InterviewEnvironment as _IE
    eval_tool = _IE._evaluator
    evaluator_stats = eval_tool.get_stats() if eval_tool else {}
    print(f"  AnswerEvaluatorTool calls: {evaluator_stats.get('total_evaluations', 'N/A')}")

    return agent, bandit, {
        "episode_rewards":   episode_rewards,
        "episode_avg_q":     episode_avg_q,
        "eval_scores":       eval_scores,
        "eval_episodes":     eval_episodes_log,
        "action_counts":     dict(action_counts),
        "bandit_stats":      bandit.get_stats(),
        "bandit_coverage":   bandit.topic_coverage(),
        "evaluator_stats":   evaluator_stats,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(agent: DQNAgent, bandit: UCBBandit, n_episodes: int = 20) -> float:
    """Run n_episodes with greedy policy; return mean total reward."""
    env = InterviewEnvironment(max_questions=MAX_QUESTIONS)
    total_rewards = []

    for _ in range(n_episodes):
        state     = env.reset()
        ep_reward = 0.0
        for _ in range(MAX_QUESTIONS):
            topic  = bandit.select()
            env.current_topic = topic
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            bandit.update(info["topic"], info["answer_quality"])
            ep_reward += reward
            if done:
                break
        total_rewards.append(ep_reward)

    return float(np.mean(total_rewards))


def baseline_random(n_episodes: int = 100) -> float:
    """Random agent baseline for comparison."""
    env = InterviewEnvironment(max_questions=MAX_QUESTIONS)
    total = []
    for _ in range(n_episodes):
        state = env.reset()
        ep_r  = 0.0
        for _ in range(MAX_QUESTIONS):
            action = random.randint(0, N_ACTIONS - 1)
            state, r, done, _ = env.step(action)
            ep_r += r
            if done:
                break
        total.append(ep_r)
    return float(np.mean(total))


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_results(logs: dict):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    colors = {
        "primary":  "#4fc3f7",
        "accent":   "#ff7043",
        "green":    "#66bb6a",
        "purple":   "#ba68c8",
        "yellow":   "#ffca28",
        "grid":     "#2a2d3a",
        "text":     "#e0e0e0",
    }
    plt.rcParams.update({"text.color": colors["text"], "axes.labelcolor": colors["text"],
                          "xtick.color": colors["text"], "ytick.color": colors["text"]})

    def styled_ax(ax, title):
        ax.set_facecolor("#1a1d27")
        ax.spines[:].set_color(colors["grid"])
        ax.tick_params(colors=colors["text"])
        ax.set_title(title, color=colors["text"], fontsize=11, pad=10)
        ax.grid(color=colors["grid"], linestyle="--", linewidth=0.5, alpha=0.7)

    # 1. Training reward (smoothed)
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = logs["episode_rewards"]
    smoothed = np.convolve(rewards, np.ones(20)/20, mode="valid")
    ax1.plot(range(len(rewards)), rewards, alpha=0.2, color=colors["primary"], linewidth=0.8)
    ax1.plot(range(len(smoothed)), smoothed, color=colors["primary"], linewidth=2)
    styled_ax(ax1, "Training Reward (smoothed)")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")

    # 2. Eval score over training
    ax2 = fig.add_subplot(gs[0, 1])
    rand_baseline = baseline_random(50)
    ax2.plot(logs["eval_episodes"], logs["eval_scores"],
             color=colors["green"], linewidth=2, marker="o", markersize=5)
    ax2.axhline(rand_baseline, color=colors["accent"], linestyle="--",
                linewidth=1.5, label=f"Random baseline ({rand_baseline:.2f})")
    ax2.legend(facecolor="#1a1d27", edgecolor=colors["grid"])
    styled_ax(ax2, "Evaluation Score vs Training Progress")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Mean Eval Reward")

    # 3. Action distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ac   = logs["action_counts"]
    acts = list(ac.keys())
    vals = [ac[a] for a in acts]
    bars = ax3.barh(acts, vals, color=colors["purple"])
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 str(v), va="center", color=colors["text"], fontsize=8)
    styled_ax(ax3, "Action Distribution (Training)")
    ax3.set_xlabel("Count")

    # 4. Bandit topic coverage (pie)
    ax4 = fig.add_subplot(gs[1, 0])
    coverage = logs["bandit_coverage"]
    wedge_colors = [colors["primary"], colors["accent"], colors["green"],
                    colors["purple"], colors["yellow"]]
    ax4.pie(
        list(coverage.values()), labels=list(coverage.keys()),
        colors=wedge_colors, autopct="%1.1f%%",
        textprops={"color": colors["text"]}, startangle=90,
    )
    ax4.set_facecolor("#1a1d27")
    ax4.set_title("UCB Bandit Topic Coverage", color=colors["text"], fontsize=11, pad=10)

    # 5. Bandit mean reward per topic
    ax5 = fig.add_subplot(gs[1, 1])
    stats = logs["bandit_stats"]
    topics      = list(stats.keys())
    mean_rewards = [stats[t]["mean_reward"] for t in topics]
    pulls        = [stats[t]["pulls"]       for t in topics]
    x = np.arange(len(topics))
    b1 = ax5.bar(x - 0.2, mean_rewards, 0.4, label="Mean Reward", color=colors["green"])
    ax5_twin = ax5.twinx()
    ax5_twin.bar(x + 0.2, pulls, 0.4, label="# Pulls", color=colors["yellow"], alpha=0.7)
    ax5.set_xticks(x); ax5.set_xticklabels(topics, rotation=25, ha="right")
    ax5.set_ylabel("Mean Reward", color=colors["green"])
    ax5_twin.set_ylabel("# Pulls", color=colors["yellow"])
    ax5_twin.tick_params(axis="y", colors=colors["yellow"])
    styled_ax(ax5, "UCB Bandit: Reward & Pull Counts per Topic")

    # 6. Average Q-values over training
    ax6 = fig.add_subplot(gs[1, 2])
    q_vals = logs["episode_avg_q"]
    if q_vals:
        ax6.plot(range(len(q_vals)), q_vals, color=colors["accent"], linewidth=1.5)
    styled_ax(ax6, "Mean Q-Value During Training")
    ax6.set_xlabel("Episode"); ax6.set_ylabel("Avg Max Q-Value")

    fig.suptitle("Adaptive Interview Coaching Agent — RL Results",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    path = os.path.join(OUTPUT_DIR, "training_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Plot saved → {path}")
    plt.close()
    return path


# ── Before / After demonstration ─────────────────────────────────────────────

def before_after_demo(trained_agent: DQNAgent, bandit: UCBBandit):
    """Compare random agent vs trained agent on identical candidate profiles."""
    np.random.seed(99)
    random.seed(99)
    env = InterviewEnvironment(max_questions=MAX_QUESTIONS)

    def run_session(use_trained: bool) -> dict:
        state     = env.reset()
        ep_reward = 0.0
        q_scores  = []
        topics_hit = set()

        for _ in range(MAX_QUESTIONS):
            topic = bandit.select() if use_trained else random.choice(TOPICS)
            env.current_topic = topic
            action = (trained_agent.select_action(state, training=False)
                      if use_trained else random.randint(0, N_ACTIONS - 1))
            state, reward, done, info = env.step(action)
            if use_trained:
                bandit.update(info["topic"], info["answer_quality"])
            q_scores.append(info["answer_quality"])
            topics_hit.add(info["topic"])
            ep_reward += reward
            if done:
                break

        return {
            "total_reward":    ep_reward,
            "mean_quality":    float(np.mean(q_scores)),
            "quality_trend":   q_scores,
            "topic_coverage":  len(topics_hit),
        }

    results = {
        "random":  run_session(use_trained=False),
        "trained": run_session(use_trained=True),
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.spines[:].set_color("#2a2d3a")
        ax.tick_params(colors="#e0e0e0")
        ax.grid(color="#2a2d3a", linestyle="--", linewidth=0.5, alpha=0.7)

    for label, color in [("random", "#ff7043"), ("trained", "#4fc3f7")]:
        axes[0].plot(results[label]["quality_trend"], label=label.title(),
                     color=color, linewidth=2)
    axes[0].set_title("Answer Quality Per Question", color="white")
    axes[0].set_xlabel("Question #"); axes[0].set_ylabel("Quality Score")
    axes[0].legend(facecolor="#1a1d27", labelcolor="white")

    labels  = ["Random Agent", "Trained Agent"]
    totals  = [results["random"]["total_reward"], results["trained"]["total_reward"]]
    axes[1].bar(labels, totals, color=["#ff7043", "#4fc3f7"])
    axes[1].set_title("Total Session Reward", color="white")
    axes[1].set_ylabel("Cumulative Reward")
    for i, v in enumerate(totals):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha="center", color="white", fontsize=11)

    fig.suptitle("Before vs After RL Training", color="white", fontsize=13,
                 fontweight="bold")
    path = os.path.join(OUTPUT_DIR, "before_after.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Before/after plot saved → {path}")
    plt.close()

    print("\n── Before / After Summary ──────────────────────────────")
    for label, r in results.items():
        print(f"  {label.upper():8s} | reward={r['total_reward']:.2f} | "
              f"mean_quality={r['mean_quality']:.3f} | "
              f"topics_covered={r['topic_coverage']}/5")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent, bandit, logs = train()

    print("\nGenerating visualisations …")
    plot_results(logs)

    print("\nRunning before/after demonstration …")
    ba_results = before_after_demo(agent, bandit)

    # Save logs as JSON
    logs_serialisable = {k: v for k, v in logs.items()
                         if not isinstance(v, dict) or k in ("action_counts", "bandit_coverage")}
    logs_serialisable["before_after"] = ba_results
    with open(os.path.join(OUTPUT_DIR, "training_logs.json"), "w") as f:
        json.dump(logs_serialisable, f, indent=2)
    print(f"\n  ✓ Logs saved → {OUTPUT_DIR}/training_logs.json")
    print("\n✅  All done! Check the 'results/' folder.\n")