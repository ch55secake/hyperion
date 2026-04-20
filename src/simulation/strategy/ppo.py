"""PPO (Proximal Policy Optimisation) trading strategy using stable-baselines3.

The :class:`PPOAgent` wraps a ``stable_baselines3.PPO`` model to provide
online PPO learning within a single simulation run.  A minimal
:class:`_TradingEnv` satisfies the ``gymnasium.Env`` interface required by
SB3 for model initialisation; the actual environment interaction happens
step-by-step through :meth:`PPOAgent.store` and :meth:`PPOAgent.maybe_update`.

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

from typing import Any, Dict

import numpy as np
import pandas as pd

import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.utils import configure_logger

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
# Minimal gymnasium environment for SB3 PPO initialisation
# ---------------------------------------------------------------------------


class _TradingEnv(gym.Env):
    """Minimal trading environment used to initialise the SB3 PPO model.

    The observation space is a 5-dimensional continuous vector and the action
    space is a discrete set of three actions (HOLD, BUY, SELL).  During actual
    trading the environment interaction is driven step-by-step via
    :class:`PPOAgent` rather than through this class's ``step`` method.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # rendering not supported

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(_STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(_ACTION_DIM)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return np.zeros(_STATE_DIM, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.zeros(_STATE_DIM, dtype=np.float32), 0.0, False, False, {}


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


class PPOAgent:
    """PPO agent backed by ``stable_baselines3.PPO``.

    Transitions are collected manually via :meth:`store` and fed into SB3's
    :class:`~stable_baselines3.common.buffers.RolloutBuffer` when
    :meth:`maybe_update` triggers a gradient update.  This preserves the same
    online-learning interaction pattern as the original implementation while
    delegating all neural-network and optimisation logic to SB3.

    Parameters
    ----------
    hidden_dim:
        Number of hidden units in each network layer.
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
        self._update_every = update_every

        # On Apple Silicon the Accelerate/vecLib BLAS routines used by PyTorch's
        # CPU backend are not thread-safe in all call patterns.  Pinning intra-op
        # parallelism to a single thread eliminates the SIGSEGV that otherwise
        # appears in softmax / orthogonal_ during SB3 policy initialisation and
        # the first gradient update.  This has no meaningful performance impact for
        # the small networks used here.
        torch.set_num_threads(1)

        env = _TradingEnv()
        # Force CPU device to avoid MPS/CUDA issues in multiprocessing
        self.model = SB3PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            clip_range=clip_eps,
            ent_coef=entropy_coef,
            vf_coef=value_coef,
            n_epochs=update_epochs,
            n_steps=update_every,
            batch_size=update_every,
            seed=seed,
            verbose=0,
            device="cpu",  # Force CPU to ensure compatibility with ProcessPoolExecutor
            policy_kwargs={
                "net_arch": [hidden_dim, hidden_dim],
                # Disable orthogonal weight initialisation: SB3's default ortho_init
                # calls torch.nn.init.orthogonal_() which segfaults on Apple Silicon
                # (MPS / Accelerate BLAS) even when the device is forced to CPU.
                "ortho_init": False,
            },
        )
        self.model.set_logger(configure_logger(0, None, ""))

        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._log_probs: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(state: np.ndarray) -> torch.Tensor:
        """Convert a state array to a batched float tensor for the SB3 policy."""
        return torch.as_tensor(state[None], dtype=torch.float32)

    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        """Sample an action from the current policy.

        Returns ``(action, log_prob)`` where *action* is an integer
        (0 = HOLD, 1 = BUY, 2 = SELL).
        """
        obs = self._to_tensor(state)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs)
            # dist.distribution is typed as Distribution | list[Distribution];
            # for a discrete action space it is always a single Distribution.
            action_dist = dist.distribution
            assert not isinstance(action_dist, list), "Expected scalar distribution for discrete action space"
            action_tensor = action_dist.sample()
            log_prob = action_dist.log_prob(action_tensor)
        return int(action_tensor.item()), float(log_prob.item())

    def store(self, state: np.ndarray, action: int, log_prob: float, reward: float, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self._states.append(state.copy())
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._dones.append(done)

    def maybe_update(self) -> None:
        """Run a PPO update if enough steps have been collected."""
        if len(self._states) >= self._update_every:
            self._ppo_update()
            self._states.clear()
            self._actions.clear()
            self._log_probs.clear()
            self._rewards.clear()
            self._dones.clear()

    def value(self, state: np.ndarray) -> float:
        """Estimate the value of *state*."""
        with torch.no_grad():
            val = self.model.policy.predict_values(self._to_tensor(state))
        return float(val.item())

    # ------------------------------------------------------------------
    # PPO update via SB3 rollout buffer
    # ------------------------------------------------------------------

    def _ppo_update(self) -> None:
        """Populate SB3's rollout buffer with stored transitions and run PPO.

        Values for all stored states are computed in a single batched forward
        pass rather than one call per step, which avoids repeated Python/C++
        round-trip overhead for the full rollout.
        """
        buffer = self.model.rollout_buffer
        buffer.reset()

        # Batch all value predictions in one forward pass instead of N serial calls.
        states_arr = np.stack(self._states)  # (N, state_dim)
        obs_batch = torch.as_tensor(states_arr, dtype=torch.float32)
        with torch.no_grad():
            all_values = self.model.policy.predict_values(obs_batch)  # (N, 1)

        for i, (state, action, log_prob, reward) in enumerate(
            zip(self._states, self._actions, self._log_probs, self._rewards)
        ):
            obs = state.reshape(1, -1)
            act = np.array([[action]])
            rew = np.array([reward])
            episode_start = np.array([False])
            val = all_values[i : i + 1]
            lp = torch.tensor([[log_prob]], dtype=torch.float32)
            buffer.add(obs, act, rew, episode_start, val, lp)

        with torch.no_grad():
            last_val = self.model.policy.predict_values(self._to_tensor(self._states[-1]))

        buffer.compute_returns_and_advantage(last_values=last_val, dones=np.array([False]))
        self.model.train()

    @property
    def buffer_size(self) -> int:
        """Number of transitions currently stored in the replay buffer."""
        return len(self._states)


# ---------------------------------------------------------------------------
# PPO Strategy
# ---------------------------------------------------------------------------


@register_strategy("ppo")
class PPOStrategy(Strategy):
    """Trading strategy driven by a ``stable_baselines3`` PPO agent.

    The agent observes both raw market data *and* the ML model's predicted
    return, then decides whether to buy, sell, or hold.  Over time it
    learns when the ML predictions are reliable and how to size positions
    accordingly.  All neural-network and optimisation logic is delegated to
    ``stable_baselines3.PPO``.

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
        Number of hidden units in each PPO network layer.
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
    def get_extra_params(prices_series: pd.Series) -> Dict[str, Any]:  # noqa: ARG004
        return {}

    @staticmethod
    def get_minimum_data_points() -> int:
        return 10
