# ED Flow Simulator — RL-ready Emergency Department environment

An event-driven simulation of a 17-room emergency department, built to train an
RL agent that optimizes length of stay, patient distribution, and resource
allocation under acuity constraints. Pure Python + numpy — no SimPy required.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every tunable parameter: layout, staffing, arrival curves, acuity mixes, psych rules, service times, vitals dynamics, reward weights |
| `ed_sim.py` | Core discrete-event engine (patients, staff task servers, vitals, boarding, surges, staffing dynamics) |
| `ed_env.py` | Gym-style wrapper: `reset()` / `step(action)` with action masking |
| `run_demo.py` | Runs 3 simulated days under a random policy vs a charge-nurse heuristic |

Run: `python3 run_demo.py`

## What's modeled

**Layout & staffing (exactly as specified):**
- Rooms 1–17. Fast track = rooms 1–2. Trauma = rooms 14–17.
- Zone nurses: A (1–5, incl. fast track), B (6–9), C (10–13), D (14–17 trauma).
- 1 triage nurse, 1 charge nurse (floats; covers a zone on callout or helps a
  drowning zone), 1 doctor, 1 NP, 1 lab tech, 1 rad tech, 1 CT tech, 2 nurse
  assistants.
- Zone nurses slow down when carrying > `NURSE_SAFE_LOAD` patients.
- Dynamic staffing: shift changes at 07:00/19:00, random callouts, delayed
  backfill, and an agent action to call in an extra nurse (costly, 90-min lag).

**Patient flow:** arrival (walk-in or ambulance; ambulances jump the triage
line) → triage → waiting room or chairs → room placement (**the agent's core
decision**) → zone-nurse assessment → provider (MD takes ESI 1–2, NP takes 4–5,
ESI-3 load-balanced) → labs/x-ray/CT by acuity-dependent probability → result
review → disposition → discharge, admission (boarding in the room until an
inpatient bed opens; boarding time scales with a stochastic hospital-occupancy
process), or psych placement. Rooms get a 10-min turnover after vacating.

**Volatility:** time-of-day arrival curve, day-to-day volume multiplier,
random crowding waves (1.4–2.2× for ~2 h), and rare multi-casualty ambulance
surges of 3–7 simultaneous high-acuity arrivals.

**Acuity & vitals:** each patient carries a latent severity that drives
observable vitals (HR, SBP, RR, SpO2, temp). Untreated patients drift worse;
treatment pulls severity down. Crossing thresholds triggers acuity upgrades
(deterioration penalty) or a code — with higher mortality if it happens in the
waiting room. Low-acuity patients renege (LWBS) if they wait too long.

**Behavioral health:**
- Medical clearances start on a "chairs" track (no bed): triage nurse draws
  labs, provider evaluates chairside — *unless* agitation exceeds a threshold,
  in which case they demand a bed (agitation grows while waiting).
- Sections / suicidal / homicidal patients get a bed and consume one nurse
  assistant as a 1:1 sitter until placement; a sitter-less SI/HI patient in a
  room accrues a safety penalty.

**Triage nurse discretion:** at triage, the nurse's gestalt (noisy read of
hidden severity) can flag a patient for a room even when the assigned ESI
wouldn't justify it — the `triage_suggestion` bit is visible to the agent, and
the underlying signal is real, so learning to respect it pays off.

## RL interface

```python
from ed_env import EDEnv
env = EDEnv(seed=0, episode_days=3)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
mask = info["action_mask"]   # boolean over the whole action space
```

**Actions** (`Discrete(2 + 8*17 = 138)`):
- `0` no-op (advance time to next decision point)
- `1` call in an extra nurse (reward cost, arrives after a delay)
- `2 + k*17 + (r-1)` place waiting candidate `k` (of the top-8, ranked by
  acuity/flags/wait) into room `r`

Invalid actions are masked; taking one acts as a no-op. After a successful
placement the agent may act again immediately (batch placements), so a single
free-bed wave can be fully allocated before time advances.

**Observation** (float vector, ~250 dims): time-of-day encoding; waiting-room
census by acuity; max wait; triage/chairs/boarding counts; hospital occupancy;
free NAs; provider and tech queue depths; per-zone staffing/census/coverage;
per-room state (occupancy, acuity, care phase, time in room, psych/sitter
flags); and per-candidate features (acuity one-hot, wait, observed vitals
abnormality, psych flags, sitter need, triage suggestion, arrival mode). The
agent sees *observed* vitals abnormality, not the latent severity — it must
learn to act under the same uncertainty as a real charge nurse.

**Reward:** per-minute waiting costs weighted by acuity, boarding and LOS
costs, sitter-gap penalties, large penalties for LWBS (acuity-scaled),
deterioration, codes, and deaths, positive rewards for discharges and admit
handoffs, and a cost for calling in staff. All weights in `config.REWARD`.

## Training recipe (on your machine)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ed_env import EDEnv

class GymED(gym.Env):
    def __init__(self, days=3):
        self.core = EDEnv(episode_days=days)
        self.action_space = spaces.Discrete(self.core.n_actions)
        self.observation_space = spaces.Box(-1, 10, (self.core.obs_dim,), np.float32)
    def reset(self, seed=None, options=None):
        return self.core.reset(seed)
    def step(self, a):
        return self.core.step(int(a))
```

Because actions are heavily masked, use a mask-aware algorithm —
**MaskablePPO** from `sb3-contrib` is the natural fit (wrap with
`ActionMasker` returning `info["action_mask"]`). RLlib and CleanRL also
support masking. Benchmark against `charge_nurse_policy` in `run_demo.py`;
beating it on total reward, LWBS, deaths, and LOS simultaneously is the goal.

## Design notes & fidelity knobs

- Exogenous randomness (arrivals, patient attributes, waves, callouts) uses a
  **separate RNG stream**, so different policies on the same seed face the
  identical patient load — clean comparisons and lower-variance evaluation.
- `ARRIVAL_SCALE` controls stress. 0.75 (~95/day, default) is stressed but
  winnable; 1.0 (~125/day) overwhelms this footprint.
- Everything is parameterized: fit `config.py` to your department's real data
  (arrival curves, ESI mix, service-time lognormals, admit rates, boarding
  distributions) for site-specific realism.
- Deliberate simplifications (extend as needed): one task per care step rather
  than granular nursing tasks; CT transport not separately modeled; providers
  don't round on boarders; no bed-manager negotiation; deaths only via codes.
- Possible extensions: multi-discrete actions for provider prioritization,
  agent-controlled charge-nurse zone assignment, observation/hallway beds,
  diversion status, and per-shift staffing decisions as a hierarchical policy.
