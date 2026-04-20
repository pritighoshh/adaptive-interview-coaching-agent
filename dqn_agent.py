"""
dqn_agent.py - Deep Q-Network in pure NumPy (no PyTorch required)
"""
import numpy as np
from collections import deque
import random
from typing import List, Optional

def relu(x): return np.maximum(0.0, x)
def relu_grad(x): return (x > 0).astype(float)

def huber_loss(pred, target, delta=1.0):
    err = pred - target
    abs_err = np.abs(err)
    loss = np.where(abs_err <= delta, 0.5*err**2, delta*(abs_err - 0.5*delta))
    grad = np.where(abs_err <= delta, err, delta*np.sign(err))
    return loss.mean(), grad / len(pred)

class MLP:
    def __init__(self, in_dim, hidden, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/in_dim),  (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2/hidden),  (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, np.sqrt(2/hidden),  (hidden, out_dim))
        self.b3 = np.zeros(out_dim)
        self._c = {}

    def forward(self, x):
        h1 = relu(x @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        self._c = {"x": x, "h1": h1, "h2": h2}
        return out

    def backward(self, lg, lr, clip=1.0):
        x, h1, h2 = self._c["x"], self._c["h1"], self._c["h2"]
        B = x.shape[0]
        dW3 = h2.T @ lg / B;  db3 = lg.mean(0);   dh2 = lg @ self.W3.T
        dh2p= dh2 * relu_grad(h2)
        dW2 = h1.T @ dh2p / B; db2 = dh2p.mean(0); dh1 = dh2p @ self.W2.T
        dh1p= dh1 * relu_grad(h1)
        dW1 = x.T @ dh1p / B;  db1 = dh1p.mean(0)
        grads = [dW1,db1,dW2,db2,dW3,db3]
        norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if norm > clip:
            grads = [g*(clip/(norm+1e-8)) for g in grads]
        dW1,db1,dW2,db2,dW3,db3 = grads
        self.W1-=lr*dW1; self.b1-=lr*db1
        self.W2-=lr*dW2; self.b2-=lr*db2
        self.W3-=lr*dW3; self.b3-=lr*db3

    def copy_from(self, o):
        for attr in ("W1","b1","W2","b2","W3","b3"):
            setattr(self, attr, getattr(o, attr).copy())

    def soft_update_from(self, o, tau):
        for attr in ("W1","b1","W2","b2","W3","b3"):
            setattr(self, attr, tau*getattr(o,attr) + (1-tau)*getattr(self,attr))

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buf.append((np.array(s,dtype=np.float32),int(a),float(r),
                         np.array(ns,dtype=np.float32),float(d)))
    def sample(self, n):
        b = random.sample(self.buf, n)
        s,a,r,ns,d = zip(*b)
        return np.array(s),np.array(a),np.array(r),np.array(ns),np.array(d)
    def __len__(self): return len(self.buf)

class DQNAgent:
    """Double DQN with experience replay, soft target updates, eps-greedy."""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, buffer_capacity=10_000, target_update_tau=0.01, hidden=128):
        self.action_dim = action_dim
        self.gamma=gamma; self.epsilon=epsilon_start
        self.eps_end=epsilon_end; self.eps_decay=epsilon_decay
        self.batch_size=batch_size; self.tau=target_update_tau; self.lr=lr
        self.online_net = MLP(state_dim, hidden, action_dim, seed=42)
        self.target_net = MLP(state_dim, hidden, action_dim, seed=42)
        self.target_net.copy_from(self.online_net)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.losses: List[float] = []
        self.q_values_log: List[float] = []

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        q = self.online_net.forward(state.reshape(1,-1))[0]
        self.q_values_log.append(float(q.max()))
        return int(q.argmax())

    def store(self, s, a, r, ns, d): self.buffer.push(s,a,r,ns,d)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size: return None
        states,actions,rewards,next_states,dones = self.buffer.sample(self.batch_size)
        q_all = self.online_net.forward(states)
        q_pred = q_all[np.arange(len(actions)), actions]
        best_a = self.online_net.forward(next_states).argmax(axis=1)
        q_next = self.target_net.forward(next_states)[np.arange(len(best_a)), best_a]
        target_q = rewards + self.gamma * q_next * (1 - dones)
        loss_val, lg = huber_loss(q_pred, target_q)
        grad_full = np.zeros_like(q_all)
        grad_full[np.arange(len(actions)), actions] = lg
        self.online_net.backward(grad_full, lr=self.lr)
        self.target_net.soft_update_from(self.online_net, self.tau)
        self.losses.append(float(loss_val))
        return float(loss_val)

    def decay_epsilon(self): self.epsilon = max(self.eps_end, self.epsilon*self.eps_decay)

    def save(self, path):
        np.savez(path, W1=self.online_net.W1, b1=self.online_net.b1,
                 W2=self.online_net.W2, b2=self.online_net.b2,
                 W3=self.online_net.W3, b3=self.online_net.b3,
                 epsilon=np.array([self.epsilon]))
        print(f"  ✓ Model saved → {path}")

    def load(self, path):
        d = np.load(path+".npz")
        for net in (self.online_net, self.target_net):
            for k in ("W1","b1","W2","b2","W3","b3"): setattr(net,k,d[k])
        self.epsilon = float(d["epsilon"][0])