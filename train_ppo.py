"""
Train a MaskablePPO agent on the ED flow environment and benchmark it against
the charge-nurse heuristic from run_demo.py.

Usage:
    python train_ppo.py                 # train with defaults, then evaluate
    python train_ppo.py --timesteps 500000
    python train_ppo.py --eval-only     # load ppo_ed.zip and just evaluate

The environment is heavily action-masked, so we use MaskablePPO (sb3-contrib)
with an ActionMasker wrapper that surfaces env.action_masks().
"""

from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import config as C
from gym_ed import GymED
from run_demo import charge_nurse_policy

MODEL_PATH = "ppo_ed"


class RewardStatsCallback(BaseCallback):
    """Log the reward_stats breakdown at the end of each training episode."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "stats" in info:
                for k, v in info["stats"].items():
                    if k.endswith("reward"):            # was endswitch
                        self.logger.record(f"reward/{k}", v)
                if self.verbose:
                    print(f"ep end @ {self.num_timesteps:,} steps  "  # added spacing
                          f"total_reward={info['stats']['total_reward']:,.0f}")
        return True


def mask_fn(env) -> np.ndarray:
    return env.action_masks()


def make_env(seed: int, days: int):
    def _init():
        env = GymED(episode_days=days, seed=seed)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def _tb_log_dir():
    """Only enable tensorboard logging if the package is actually installed."""
    try:
        import tensorboard  # noqa: F401
        return "./tb_ed"
    except ImportError:
        return None


def build_model(days: int, seed: int, n_envs: int) -> MaskablePPO:
    venv = DummyVecEnv([make_env(seed + i, days) for i in range(n_envs)])
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        seed=seed,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,          # long episodes (~3 sim days of decision points)
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        ##0.0003
        clip_range=0.2,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=_tb_log_dir(),
    )
    return model


def run_heuristic(seed: int, days: int) -> float:
    """Total reward of the charge-nurse baseline on the same seed."""
    env = GymED(episode_days=days, seed=seed)
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    total = 0.0
    done = False
    while not done:
        a = charge_nurse_policy(env.core, obs, info, rng)
        obs, r, done, _, info = env.step(a)
        total += r
    return total


def evaluate(model: MaskablePPO, days: int, seeds=range(5)) -> None:
    print("\n=== Evaluation (MaskablePPO vs charge-nurse heuristic) ===")
    ppo_scores, heur_scores = [], []
    for s in seeds:
        env = GymED(episode_days=days, seed=int(s))
        obs, info = env.reset(seed=int(s))
        total = 0.0
        done = False
        while not done:
            masks = get_action_masks(env)
            a, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, r, done, _, info = env.step(int(a))
            total += r
        ppo_scores.append(total)
        heur_scores.append(run_heuristic(int(s), days))
        print(f"  seed {s}:  PPO {total:>12,.0f}   heuristic {heur_scores[-1]:>12,.0f}")
    print(f"\n  mean PPO       {np.mean(ppo_scores):>12,.0f}")
    print(f"  mean heuristic {np.mean(heur_scores):>12,.0f}")
    print("  last-episode PPO stats:")
    for k, v in info["stats"].items():
        print(f"    {k:26s} {v}")
    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--days", type=int, default=C.EPISODE_DAYS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    if args.eval_only:
        model = MaskablePPO.load(MODEL_PATH)
        evaluate(model, args.days)
        return

    model = build_model(args.days, args.seed, args.n_envs)
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=RewardStatsCallback(verbose=1))
    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}.zip")
    evaluate(model, args.days)


if __name__ == "__main__":
    main()
