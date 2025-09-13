"""DQN variant with prioritized experience replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN

from .prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
    PrioritizedReplayBufferSamples,
)


class PrioritizedDQN(DQN):
    """Deep Q-Network with prioritized replay buffer."""

    def __init__(
        self,
        *args: Any,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        priority_beta_increment: float = 1e-3,
        priority_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.priority_beta_increment = priority_beta_increment
        replay_buffer_kwargs = kwargs.pop("replay_buffer_kwargs", {})
        replay_buffer_kwargs.update(
            dict(alpha=priority_alpha, beta=priority_beta, eps=priority_eps)
        )
        super().__init__(
            *args,
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            **kwargs,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # type: ignore[override]
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data: PrioritizedReplayBufferSamples = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            td_errors = F.smooth_l1_loss(
                current_q_values, target_q_values, reduction="none"
            )
            loss = (td_errors * replay_data.weights).mean()
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            new_priorities = td_errors.detach().cpu().numpy().squeeze()
            self.replay_buffer.update_priorities(replay_data.indices, new_priorities)
            # Anneal beta towards 1.0
            self.replay_buffer.beta = min(
                1.0, self.replay_buffer.beta + self.priority_beta_increment
            )

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
