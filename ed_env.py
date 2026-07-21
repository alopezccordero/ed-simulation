"""
Gym-style RL environment over EDSim.

No gymnasium dependency required here — the API matches gymnasium's, so on your
own machine you can do:

    import gymnasium as gym
    from gymnasium import spaces

    class GymED(gym.Env):
        def __init__(self):
            self.core = EDEnv()
            self.action_space = spaces.Discrete(self.core.n_actions)
            self.observation_space = spaces.Box(-1, 10, (self.core.obs_dim,), np.float32)
        def reset(self, seed=None, options=None): return self.core.reset(seed)
        def step(self, a): return self.core.step(a)

Action space (Discrete, flattened):
    0                                   -> NO-OP (advance time)
    1                                   -> CALL EXTRA NURSE (costly, arrives later)
    2 + k*NUM_ROOMS + (r-1)             -> place waiting candidate k into room r

Invalid actions are masked (info["action_mask"]) and treated as NO-OP if taken.
"""

from __future__ import annotations
import numpy as np

import config as C
from ed_sim import EDSim, PHASE_IDX


class EDEnv:
    K = EDSim.K_CANDIDATES
    N_SPECIAL = 2  # noop, call extra nurse

    def __init__(self, seed: int = C.SEED, episode_days: int = C.EPISODE_DAYS):
        self.sim = EDSim(seed=seed, episode_days=episode_days)
        self.n_actions = self.N_SPECIAL + self.K * C.NUM_ROOMS
        self.obs_dim = len(self._obs_zeros())

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None):
        self.sim.reset(seed)
        self.sim.pop_reward()  # discard warmup reward
        return self._obs(), {"action_mask": self.action_mask()}

    def step(self, action: int):
        sim = self.sim
        acted = False
        if action == 1:
            sim.agent_call_extra_nurse()
            acted = True
        elif action >= self.N_SPECIAL:
            k, r = divmod(action - self.N_SPECIAL, C.NUM_ROOMS)
            r += 1
            cands = sim.waiting_candidates()
            if k < len(cands) and sim.can_place(cands[k], r):
                sim.apply_place(cands[k], r)
                acted = True
        # if the agent placed someone and more moves are possible, let it act
        # again immediately; otherwise advance the world to the next decision
        if not (acted and sim.waiting and sim.free_rooms()):
            sim.run_until_decision()
        reward = sim.pop_reward() * C.REWARD_SCALE
        terminated = sim.done
        obs = self._obs()
        info = {"action_mask": self.action_mask(), "stats": sim.summary()}
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------ masking
    def action_mask(self) -> np.ndarray:
        sim = self.sim
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True #what are these indexes for?
        mask[1] = True
        cands = sim.waiting_candidates()
        free = set(sim.free_rooms())
        for k, p in enumerate(cands):
            for r in free:
                if sim.can_place(p, r):
                    mask[self.N_SPECIAL + k * C.NUM_ROOMS + (r - 1)] = True
        return mask

    # ------------------------------------------------------------ observation
    def _obs_zeros(self):
        return self._build_obs(zeros=True)

    def _obs(self):
        return self._build_obs(zeros=False)

    def _build_obs(self, zeros=False) -> np.ndarray:
        sim = self.sim
        v = []
        now = 0.0 if zeros else sim.now
        # time of day (sin/cos), day progress
        tod = (now % 1440) / 1440 * 2 * np.pi
        v += [np.sin(tod), np.cos(tod), (now % 1440) / 1440]
        if zeros:
            v += [0.0] * 14
        else:
            wr_by_ac = [sum(1 for p in sim.waiting if p.acuity == a) for a in range(1, 6)]
            max_wait = max([now - p.t_arrival for p in sim.waiting], default=0.0)
            v += [x / 5.0 for x in wr_by_ac]
            v += [
                min(1.0, max_wait / 240.0),
                len(sim.chairs) / 5.0,
                len(sim.triage_queue) / 8.0,
                sim.hosp_occ,
                sum(1 for p in map(sim.patients.get, sim.active)
                    if p.phase == "boarding") / 8.0,
                (sim.na_total - sim.na_sitting) / max(1, sim.na_total),
                sim.doctor.qlen() / 8.0,
                sim.np_.qlen() / 8.0,
                (sim.lab.qlen() + sim.rad.qlen() + sim.ct.qlen()) / 10.0,
            ]
        # zones: staffed?, census/6, charge covering flag, extra nurses
        for z in C.ZONES:
            if zeros:
                v += [0.0, 0.0, 0.0]
            else:
                v += [1.0 if sim._zone_staffed(z) else 0.0,
                      sim.zone_census(z) / 6.0,
                      1.0 if sim.charge_covering == z else 0.0]
        v += [0.0 if zeros else min(1.0, sim.extra_nurses / 2.0)]
        # rooms: 17 x 7
        for r in range(1, C.NUM_ROOMS + 1):
            if zeros or sim.rooms.get(r) is None:
                blocked = 0.0 if zeros else (1.0 if sim.room_blocked[r] else 0.0)
                v += [0.0, blocked, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                p = sim.patients[sim.rooms[r]]
                v += [1.0, 0.0,
                      p.acuity / 5.0,
                      PHASE_IDX[p.phase] / (len(PHASE_IDX) - 1),
                      min(1.0, (now - (p.t_roomed or now)) / 360.0),
                      1.0 if p.psych else 0.0,
                      1.0 if (p.needs_sitter and not p.sitter) else 0.0]
        # waiting candidates: K x 12
        cands = [] if zeros else sim.waiting_candidates()
        for k in range(self.K):
            if k >= len(cands):
                v += [0.0] * 12
            else:
                p = cands[k]
                onehot = [1.0 if p.acuity == a else 0.0 for a in range(1, 6)]
                v += [1.0] + onehot + [
                    min(1.0, (now - p.t_arrival) / 240.0),
                    p.vitals_abnormality(sim.rng),
                    1.0 if p.psych else 0.0,
                    1.0 if p.needs_sitter else 0.0,
                    1.0 if p.triage_suggestion else 0.0,
                    1.0 if p.mode == "ambulance" else 0.0,
                ]
        return np.asarray(v, dtype=np.float32)
