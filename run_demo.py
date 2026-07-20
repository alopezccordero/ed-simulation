"""
Demo: run the ED for 3 simulated days under two policies and compare.

  1. random-valid  : picks any valid action (chaos baseline)
  2. charge-nurse  : heuristic close to real practice —
       - ESI-1/2 first, trauma rooms for ESI-1
       - honor triage-nurse "get them back" flags
       - fast track (rooms 1-2) reserved for ESI-4/5
       - psych needing sitter only bedded when an NA is free (else wait, unless
         they're deteriorating)
       - longest-waiting wins ties

Use the same structure to plug in an RL policy later.
"""

import numpy as np

import config as C
from ed_env import EDEnv


def random_policy(env, obs, info, rng):
    valid = np.flatnonzero(info["action_mask"])
    return int(rng.choice(valid))


def charge_nurse_policy(env, obs, info, rng):
    sim = env.sim
    cands = sim.waiting_candidates()
    free = sim.free_rooms()
    if not cands or not free:
        return 0
    ft = [r for r in free if r in C.FAST_TRACK_ROOMS]
    trauma = [r for r in free if r in C.TRAUMA_ROOMS]
    main = [r for r in free if r not in C.FAST_TRACK_ROOMS]

    order = sorted(range(len(cands)),
                   key=lambda k: (cands[k].acuity,
                                  not cands[k].triage_suggestion,
                                  -(sim.now - cands[k].t_arrival)))
    for k in order:
        p = cands[k]
        # hold sitter-requiring psych until an NA frees up (unless they're hot)
        if p.needs_sitter and sim.na_sitting >= sim.na_total \
                and p.true_sev < 0.6 and p.agitation < 0.8:
            continue
        if p.acuity == 1:
            pool = trauma or main or ft
        elif p.acuity >= 4 and not p.triage_suggestion:
            pool = ft or main
        else:
            pool = main or ft
        for r in pool:
            if sim.can_place(p, r):
                return env.N_SPECIAL + k * C.NUM_ROOMS + (r - 1)
    return 0


def run(policy, name, seed=7, days=3):
    env = EDEnv(seed=seed, episode_days=days)
    obs, info = env.reset()
    rng = np.random.default_rng(0)
    total_r, steps = 0.0, 0
    done = False
    while not done:
        a = policy(env, obs, info, rng)
        obs, r, done, _, info = env.step(a)
        total_r += r
        steps += 1
    print(f"\n=== {name} ({days} days, seed {seed}) ===")
    print(f"steps: {steps}   total reward: {total_r:,.0f}")
    for k, v in info["stats"].items():
        print(f"  {k:26s} {v}")
    return total_r


if __name__ == "__main__":
    run(random_policy, "RANDOM-VALID policy")
    run(charge_nurse_policy, "CHARGE-NURSE heuristic")
