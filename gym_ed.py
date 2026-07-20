"""
Gymnasium-compatible wrapper around the mask-aware EDEnv core.

Exposes:
  * a proper `gymnasium.Env` (Discrete actions, Box observations)
  * `action_masks()` so `sb3_contrib.MaskablePPO` (via `ActionMasker`) can
    read the per-step boolean action mask.

The underlying `EDEnv` already returns a gymnasium-style 5-tuple from `step`
and `(obs, info)` from `reset`; this wrapper just declares the spaces and
plumbs the mask through.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C
from ed_env import EDEnv


class GymED(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_days: int = C.EPISODE_DAYS, seed: int = C.SEED):
        super().__init__()
        self.core = EDEnv(seed=seed, episode_days=episode_days)
        self.action_space = spaces.Discrete(self.core.n_actions)
        # Observations are mostly normalized to ~[-1, 1], but a few census
        # ratios can spike above 1 under heavy load — use a generous finite box
        # rather than clipping information away.
        self.observation_space = spaces.Box(
            low=-10.0, high=100.0, shape=(self.core.obs_dim,), dtype=np.float32
        )
        # EDSim's state attributes only exist after a reset; establish a valid
        # initial mask so wrappers can query it before the first reset() call.
        _, info = self.core.reset(seed)
        self._last_mask = info["action_mask"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.core.reset(seed)
        self._last_mask = info["action_mask"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.core.step(int(action))
        self._last_mask = info["action_mask"]
        return obs, reward, terminated, truncated, info

    # sb3-contrib looks for this method (directly or via ActionMasker).
    def action_masks(self) -> np.ndarray:
        return self._last_mask
