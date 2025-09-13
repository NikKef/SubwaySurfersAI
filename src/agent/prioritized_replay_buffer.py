"""Prioritized replay buffer for proportional prioritization.

This implementation extends Stable-Baselines3's :class:`ReplayBuffer`
with proportional prioritized sampling following
`Schaul et al. (2016) <https://arxiv.org/abs/1511.05952>`_.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from typing import NamedTuple

from stable_baselines3.common.type_aliases import ReplayBufferSamples

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from stable_baselines3.common.vec_env import VecNormalize


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with proportional prioritization."""

    def __init__(
        self,
        *args,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        """Create prioritized replay buffer.

        Parameters
        ----------
        alpha:
            Exponent determining how much prioritization is used (0 means
            uniform sampling).
        beta:
            Initial value of beta for importance-sampling weights.
        eps:
            Small constant added to priorities to avoid zero probability.
        """

        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Add a new transition and assign it maximum priority."""
        super().add(*args, **kwargs)
        idx = (self.pos - 1) % self.buffer_size
        self.priorities[idx] = self.max_priority

    def _get_probabilities(self) -> np.ndarray:
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos]
        scaled = priorities**self.alpha
        return scaled / scaled.sum()

    def sample(self, batch_size: int, env: Optional["VecNormalize"] = None):  # type: ignore[override]
        """Sample a batch of experiences."""
        probs = self._get_probabilities()
        indices = np.random.choice(len(probs), batch_size, p=probs)
        samples: ReplayBufferSamples = super()._get_samples(indices, env=env)
        total = len(probs)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        w = th.as_tensor(weights, device=self.device).reshape(-1, 1)
        return PrioritizedReplayBufferSamples(
            samples.observations,
            samples.actions,
            samples.next_observations,
            samples.dones,
            samples.rewards,
            w,
            indices,
            samples.discounts,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        priorities = np.abs(priorities) + self.eps
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


class PrioritizedReplayBufferSamples(NamedTuple):
    """Samples from :class:`PrioritizedReplayBuffer` including weights."""

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor
    indices: np.ndarray
    discounts: Optional[th.Tensor]
