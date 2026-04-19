"""PPO (Proximal Policy Optimisation) trading strategy.

The :class:`PPOAgent` implements a minimal, dependency-free PPO learner
built on NumPy.  It maintains two small feed-forward networks:

* **Policy network** – maps the current observation to a probability
  distribution over three discrete actions (HOLD, BUY, SELL).
* **Value network** – estimates the expected discounted return from the
  current state (used as a baseline to reduce gradient variance).

The :class:`PPOStrategy` wraps the agent inside the :class:`Strategy`
interface.  On every ``execute()`` call it:

1. Builds an observation from the raw market data *and* the ML model's
   predicted return (so the agent learns when to *trust* that prediction).
2. Samples an action from the current policy.
3. Stores the transition in a replay buffer.
4. After a configurable number of steps it runs a PPO update to improve
   the policy in-place.

This lets the agent adapt online during a single simulation run, which
mirrors the incremental nature of the back-test loop already present in
the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .strategy import Strategy
from .strategy_registry import register_strategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_DIM = 5  # [pred_return, position_flag, portfolio_ratio, days_held_norm, abs_pred_return]
_ACTION_DIM = 3  # 0 = HOLD, 1 = BUY, 2 = SELL

_ACTION_HOLD = 0
_ACTION_BUY = 1
_ACTION_SELL = 2


# ---------------------------------------------------------------------------
# Tiny NumPy neural-network helpers
# ---------------------------------------------------------------------------


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, scale, (fan_in, fan_out))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-12)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@dataclass
class _NetworkGrads:
    """Accumulated gradients for a two-layer network."""

    grad_w1: np.ndarray
    grad_b1: np.ndarray
    grad_w2: np.ndarray
    grad_b2: np.ndarray

    def __add__(self, other: "_NetworkGrads") -> "_NetworkGrads":
        return _NetworkGrads(
            self.grad_w1 + other.grad_w1,
            self.grad_b1 + other.grad_b1,
            self.grad_w2 + other.grad_w2,
            self.grad_b2 + other.grad_b2,
        )

    def as_list(self) -> List[np.ndarray]:
        return [self.grad_w1, self.grad_b1, self.grad_w2, self.grad_b2]


class _Network:
    """Two-layer feed-forward network with tanh hidden activations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rng: np.random.Generator):
        self.w1 = _xavier_init(in_dim, hidden_dim, rng)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = _xavier_init(hidden_dim, out_dim, rng)
        self.b2 = np.zeros(out_dim)

        # Adam optimiser state
        self._adam_m = [np.zeros_like(p) for p in self._params()]
        self._adam_v = [np.zeros_like(p) for p in self._params()]
        self._adam_t = 0

    def _params(self) -> List[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def zero_grads(self) -> _NetworkGrads:
        """Return a zeroed gradient container matching this network's shapes."""
        return _NetworkGrads(
            np.zeros_like(self.w1),
            np.zeros_like(self.b1),
            np.zeros_like(self.w2),
            np.zeros_like(self.b2),
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pre_activations_h1, output)``."""
        pre_h1 = x @ self.w1 + self.b1
        h1 = _tanh(pre_h1)
        out = h1 @ self.w2 + self.b2
        return pre_h1, out

    def adam_update(self, grads: _NetworkGrads, lr: float, beta1: float = 0.9, beta2: float = 0.999) -> None:
        self._adam_t += 1
        t = self._adam_t
        for idx, (p, g) in enumerate(zip(self._params(), grads.as_list())):
            self._adam_m[idx] = beta1 * self._adam_m[idx] + (1 - beta1) * g
            self._adam_v[idx] = beta2 * self._adam_v[idx] + (1 - beta2) * g**2
            m_hat = self._adam_m[idx] / (1 - beta1**t)
            v_hat = self._adam_v[idx] / (1 - beta2**t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


@dataclass
class _ReplayBuffer:
    """Stores transitions collected between PPO updates."""

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()


class PPOAgent:  # pylint: disable=too-many-instance-attributes
    """Lightweight PPO agent for stock trading decisions.

    The agent uses two small networks:

    * ``policy_net``:  state → action logits (softmax → probabilities)
    * ``value_net``:   state → scalar value estimate

    Parameters
    ----------
    hidden_dim:
        Number of hidden units in each network.
    lr:
        Learning rate for the Adam optimiser.
    gamma:
        Discount factor for computing returns.
    clip_eps:
        PPO clipping parameter (ε).
    entropy_coef:
        Coefficient for the entropy bonus (encourages exploration).
    value_coef:
        Coefficient for the value-function loss.
    update_epochs:
        Number of gradient steps per PPO update.
    update_every:
        Number of environment steps between PPO updates.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        update_every: int = 32,
        seed: int = 42,
    ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.update_every = update_every
        self.lr = lr

        rng = np.random.default_rng(seed)
        self._rng = rng
        self.policy_net = _Network(_STATE_DIM, hidden_dim, _ACTION_DIM, rng)
        self.value_net = _Network(_STATE_DIM, hidden_dim, 1, rng)
        self._buffer = _ReplayBuffer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        """Sample an action from the current policy.

        Returns ``(action, log_prob)`` where *action* is an integer
        (0 = HOLD, 1 = BUY, 2 = SELL).
        """
        _, logits = self.policy_net.forward(state)
        probs = _softmax(logits)
        action = int(self._rng.choice(_ACTION_DIM, p=probs))
        log_prob = float(np.log(probs[action] + 1e-12))
        return action, log_prob

    def store(self, state: np.ndarray, action: int, log_prob: float, reward: float, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self._buffer.states.append(state.copy())
        self._buffer.actions.append(action)
        self._buffer.log_probs.append(log_prob)
        self._buffer.rewards.append(reward)
        self._buffer.dones.append(done)

    def maybe_update(self) -> None:
        """Run a PPO update if enough steps have been collected."""
        if self._buffer.size >= self.update_every:
            self._ppo_update()
            self._buffer.clear()

    def value(self, state: np.ndarray) -> float:
        """Estimate the value of *state*."""
        _, v_out = self.value_net.forward(state)
        return float(v_out[0])

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Compute discounted returns using Monte-Carlo rollouts.

        Each return is the sum of discounted future rewards up to the end of the
        episode (or buffer boundary).  The ``dones`` array resets the running sum
        at terminal steps.
        """
        returns = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            returns[t] = running
        return returns

    def _value_step_grads(self, state: np.ndarray, ret: float) -> _NetworkGrads:
        """Compute value-network gradients for a single transition."""
        n = self._buffer.size
        pre_h1, v_out = self.value_net.forward(state)
        v_pred = v_out[0]

        dl_dv = 2.0 * (v_pred - ret) * self.value_coef / n
        dv_dz2 = np.array([1.0])
        h1 = _tanh(pre_h1)
        grad_w2 = np.outer(h1, dv_dz2 * dl_dv)
        grad_b2 = dv_dz2 * dl_dv
        dl_dh1 = (dv_dz2 * dl_dv) @ self.value_net.w2.T
        dl_dz1 = dl_dh1 * _tanh_grad(pre_h1)
        grad_w1 = np.outer(state, dl_dz1)
        grad_b1 = dl_dz1

        return _NetworkGrads(grad_w1, grad_b1, grad_w2, grad_b2)

    def _policy_step_grads(
        self, state: np.ndarray, action: int, old_log_prob: float, advantage: float
    ) -> _NetworkGrads:
        """Compute policy-network gradients for a single transition."""
        n = self._buffer.size
        pre_h1, logits = self.policy_net.forward(state)
        probs = _softmax(logits)
        new_log_prob = float(np.log(probs[action] + 1e-12))

        ratio = np.exp(new_log_prob - old_log_prob)
        clipped = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        loss = -min(ratio * advantage, clipped * advantage) / n
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        loss -= self.entropy_coef * entropy / n

        indicator = np.zeros(_ACTION_DIM)
        indicator[action] = 1.0
        dl_dlogits = (probs - indicator) * loss

        h1 = _tanh(pre_h1)
        grad_w2 = np.outer(h1, dl_dlogits)
        grad_b2 = dl_dlogits
        dl_dh1 = dl_dlogits @ self.policy_net.w2.T
        dl_dz1 = dl_dh1 * _tanh_grad(pre_h1)
        grad_w1 = np.outer(state, dl_dz1)
        grad_b1 = dl_dz1

        return _NetworkGrads(grad_w1, grad_b1, grad_w2, grad_b2)

    def _ppo_update(self) -> None:
        """Run PPO gradient updates over the collected buffer."""
        states = np.array(self._buffer.states, dtype=np.float32)
        actions = np.array(self._buffer.actions, dtype=np.int32)
        old_log_probs = np.array(self._buffer.log_probs, dtype=np.float32)
        rewards = np.array(self._buffer.rewards, dtype=np.float32)
        dones = np.array(self._buffer.dones, dtype=np.float32)

        returns = self._compute_returns(rewards, dones)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.update_epochs):
            policy_grads = self.policy_net.zero_grads()
            value_grads = self.value_net.zero_grads()

            for idx, state in enumerate(states):
                action = int(actions[idx])
                old_lp = float(old_log_probs[idx])
                ret = float(returns[idx])

                vg = self._value_step_grads(state, ret)
                value_grads = value_grads + vg

                _, v_out = self.value_net.forward(state)
                advantage = ret - float(v_out[0])
                pg = self._policy_step_grads(state, action, old_lp, advantage)
                policy_grads = policy_grads + pg

            self.policy_net.adam_update(policy_grads, self.lr)
            self.value_net.adam_update(value_grads, self.lr)

    @property
    def buffer_size(self) -> int:
        """Number of transitions currently stored in the replay buffer."""
        return self._buffer.size


# ---------------------------------------------------------------------------
# PPO Strategy
# ---------------------------------------------------------------------------


@register_strategy("ppo")
class PPOStrategy(Strategy):
    """Trading strategy driven by a Proximal Policy Optimisation agent.

    The agent observes both raw market data *and* the ML model's predicted
    return, then decides whether to buy, sell, or hold.  Over time it
    learns when the ML predictions are reliable and how to size positions
    accordingly.

    Parameters
    ----------
    simulator:
        The :class:`~src.simulation.trading_simulator.TradingSimulator`
        instance (injected by the framework).
    capital:
        Initial capital available to the strategy.
    initial_capital:
        Reference capital used to normalise the portfolio ratio
        observation.  Defaults to *capital*.
    hidden_dim:
        Number of hidden units in each PPO network.
    lr:
        Learning rate for the Adam optimiser inside the PPO agent.
    gamma:
        Discount factor.
    clip_eps:
        PPO clipping parameter (ε).
    entropy_coef:
        Entropy regularisation coefficient.
    update_every:
        Number of steps between PPO gradient updates.
    seed:
        Random seed.
    """

    def __init__(
        self,
        simulator: Any,
        capital: float,
        initial_capital: float | None = None,
        hidden_dim: int = 32,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        update_every: int = 32,
        seed: int = 42,
    ):
        super().__init__(simulator, capital)
        self._initial_capital = float(initial_capital if initial_capital is not None else capital)
        self._prev_portfolio_value = self._initial_capital

        self.agent = PPOAgent(
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            update_every=update_every,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def execute(self, date: Any, price: Any, pred_return: Any, actual_return: Any) -> tuple[float, Any, Any, float]:
        state = self._build_state(pred_return, price)
        action, log_prob = self.agent.select_action(state)

        # Execute the chosen action
        if action == _ACTION_BUY and self.position is None:
            self.buy(date, price)
        elif action == _ACTION_SELL and self.position == "long":
            self.sell(date, price, pred_return)
        # _ACTION_HOLD: do nothing

        # Compute reward as the change in portfolio value
        current_portfolio = self.shares * float(price) if self.position == "long" else self.capital
        reward = (current_portfolio - self._prev_portfolio_value) / (self._initial_capital + 1e-12)
        self._prev_portfolio_value = current_portfolio

        # Store transition and maybe update
        self.agent.store(state, action, log_prob, reward, done=False)
        self.agent.maybe_update()

        return self.capital, self.entry_price, self.position, self.shares

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_state(self, pred_return: Any, price: Any) -> np.ndarray:
        """Construct a normalised observation vector.

        Observation components
        ----------------------
        0. ``pred_return``         – raw ML prediction (clipped to ±1)
        1. ``position_flag``       – 1.0 if in a long position, else 0.0
        2. ``portfolio_ratio``     – current value / initial capital (clipped)
        3. ``days_held_norm``      – ``hold_counter / 30`` (clipped to [0, 1])
        4. ``abs_pred_return``     – |pred_return| as a confidence proxy
        """
        pred = float(np.clip(pred_return, -1.0, 1.0))
        position_flag = 1.0 if self.position == "long" else 0.0
        current_value = self.shares * float(price) if self.position == "long" else self.capital
        portfolio_ratio = float(np.clip(current_value / (self._initial_capital + 1e-12), 0.0, 3.0))
        days_held = float(np.clip(self.hold_counter / 30.0, 0.0, 1.0))
        abs_pred = abs(pred)
        return np.array([pred, position_flag, portfolio_ratio, days_held, abs_pred], dtype=np.float32)

    @staticmethod
    def get_extra_params(price_series: pd.Series) -> Dict[str, Any]:  # noqa: ARG004
        return {}

    @staticmethod
    def get_minimum_data_points() -> int:
        return 10
