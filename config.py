"""
Configuration for the ED simulation.
All times are in MINUTES. All rates are per-minute unless noted.
Tune everything here to match your department's real data.
"""

# ---------------------------------------------------------------- layout ----
NUM_ROOMS = 17

# Zones exactly as described:
#   Nurse "FT/A"  -> fast track rooms 1-2 plus rooms 3-5
#   Nurse "B"     -> rooms 6-9
#   Nurse "C"     -> rooms 10-13
#   Nurse "D"     -> rooms 14-17 (trauma rooms)
ZONES = {
    "A": [1, 2, 3, 4, 5],       # includes fast track (1, 2)
    "B": [6, 7, 8, 9],
    "C": [10, 11, 12, 13],
    "D": [14, 15, 16, 17],      # trauma
}
FAST_TRACK_ROOMS = [1, 2]
TRAUMA_ROOMS = [14, 15, 16, 17]

# --------------------------------------------------------------- staffing ---
BASE_STAFF = {
    "triage_nurse": 1,
    "zone_nurses": 4,        # one per zone A-D
    "charge_nurse": 1,       # floats; can cover a zone on callout / surge
    "doctor": 1,
    "np": 1,
    "lab_tech": 1,
    "rad_tech": 1,           # x-ray
    "ct_tech": 1,
    "nurse_assistant": 2,    # can be pulled as 1:1 sitters
}

SHIFT_CHANGE_HOURS = [7, 19]          # 07:00 and 19:00
NURSE_CALLOUT_PROB = 0.08             # per nurse per shift
CALLOUT_BACKFILL_DELAY_MEAN = 180.0   # minutes until agency/callback nurse arrives (if at all)
CALLOUT_BACKFILL_PROB = 0.5
EXTRA_NURSE_CALLIN_DELAY = 90.0       # delay when the AGENT requests an extra nurse
EXTRA_NURSE_COST = 40.0               # reward penalty for calling someone in

# Max concurrent patients a zone nurse can safely carry before care slows down
NURSE_SAFE_LOAD = 4
NURSE_OVERLOAD_SLOWDOWN = 0.35        # +35% task time per patient above safe load

# ---------------------------------------------------------------- arrivals --
# Global scale on all arrivals (1.0 ~ 124 pts/day: overwhelming for this footprint;
# 0.75 ~ 93/day: stressed but winnable — good RL regime)
ARRIVAL_SCALE = 0.75
# Base arrival rate (patients/hour) by hour of day (0-23) — typical bimodal ED curve
HOURLY_ARRIVALS = [
    3.0, 2.5, 2.0, 1.8, 1.8, 2.0, 2.5, 3.5,      # 00-07
    5.0, 6.5, 7.5, 8.0, 7.5, 7.0, 6.5, 6.5,      # 08-15
    7.0, 7.5, 8.0, 7.5, 6.5, 5.5, 4.5, 3.5,      # 16-23
]
# Day-to-day volatility: each day gets a multiplier ~ lognormal(0, sigma)
DAILY_VOLUME_SIGMA = 0.18
# Short "crowding waves" (e.g., nursing-home dump, weather): random multiplier bursts
WAVE_RATE_PER_DAY = 1.5           # expected waves per day
WAVE_DURATION_MEAN = 120.0        # minutes
WAVE_MULTIPLIER = (1.4, 2.2)      # uniform range

# Multi-casualty / EMS surge events (rare): burst of ambulances at once
SURGE_RATE_PER_DAY = 0.25
SURGE_SIZE = (3, 7)               # simultaneous ambulance arrivals
SURGE_HIGH_ACUITY = True

AMBULANCE_FRACTION = 0.22         # of all arrivals

# ------------------------------------------------------------------ acuity --
# ESI 1 (critical) .. 5 (minor). Walk-in vs ambulance mixes differ.
ACUITY_DIST_WALKIN = {1: 0.005, 2: 0.10, 3: 0.45, 4: 0.32, 5: 0.125}
ACUITY_DIST_AMBO = {1: 0.05, 2: 0.30, 3: 0.45, 4: 0.17, 5: 0.03}

# -------------------------------------------------------------- psychiatry --
PSYCH_FRACTION = 0.10                     # of arrivals are behavioral-health
# Among psych arrivals:
PSYCH_TYPES = {
    "medical_clearance": 0.55,            # may not need a bed...
    "section": 0.20,                      # involuntary hold -> bed + sitter
    "suicidal": 0.17,                     # -> bed + sitter
    "homicidal": 0.08,                    # -> bed + sitter
}
# Medical-clearance agitation: if agitation exceeds threshold they get a bed anyway
AGITATION_BED_THRESHOLD = 0.65
PSYCH_DISPO_TIME_MEAN = 300.0             # placement search after clearance (long tail)
PSYCH_DISPO_TIME_SIGMA = 0.8

# ------------------------------------------------------------ triage logic --
TRIAGE_TIME_MEAN = 5.0
TRIAGE_TIME_SIGMA = 0.35
# Probability the triage nurse's gestalt flags a patient for a room even when
# acuity alone wouldn't justify it. Driven by hidden severity + noise.
TRIAGE_OVERRIDE_NOISE = 0.15
TRIAGE_OVERRIDE_BASE = 0.06

# ---------------------------------------------------------- service times ---
# Lognormal (mean minutes, sigma) keyed by acuity where relevant
NURSE_ASSESS_TIME = {1: (10, .3), 2: (12, .3), 3: (10, .3), 4: (7, .3), 5: (5, .3)}
PROVIDER_EVAL_TIME = {1: (25, .4), 2: (20, .4), 3: (15, .4), 4: (10, .4), 5: (7, .4)}
NURSE_TASK_TIME = (8, 0.4)          # meds/IV/recheck bundles
DISPO_DECISION_TIME = (6, 0.3)
DISCHARGE_PROCESS_TIME = (12, 0.3)

# Diagnostics: probability ordered by acuity, and process times
LAB_PROB = {1: 0.98, 2: 0.90, 3: 0.75, 4: 0.35, 5: 0.10}
XRAY_PROB = {1: 0.50, 2: 0.45, 3: 0.40, 4: 0.35, 5: 0.15}
CT_PROB = {1: 0.70, 2: 0.45, 3: 0.25, 4: 0.08, 5: 0.01}
LAB_DRAW_TIME = (7, 0.3)
LAB_RESULT_TIME = (45, 0.35)         # analyzer turnaround after draw
XRAY_TIME = (15, 0.3)
CT_TIME = (25, 0.3)
RESULT_REVIEW_TIME = (5, 0.3)

# ----------------------------------------------------------------- boarding -
ADMIT_PROB = {1: 0.90, 2: 0.55, 3: 0.28, 4: 0.06, 5: 0.01}
# Hospital occupancy is a mean-reverting process in [0,1]; boarding time scales with it
HOSPITAL_OCC_MEAN = 0.85
HOSPITAL_OCC_VOL = 0.03
BOARDING_TIME_BASE = 150.0           # minutes at low occupancy
BOARDING_TIME_MAX_EXTRA = 600.0      # added as occupancy -> 1

# ------------------------------------------------------------------- vitals -
# Latent severity s in [0,1]. Vitals are emitted from s. OU-style dynamics.
SEV_INIT_BY_ACUITY = {1: 0.85, 2: 0.62, 3: 0.40, 4: 0.22, 5: 0.10}
SEV_INIT_NOISE = 0.08
SEV_MEANREV = 0.004                  # pull toward trajectory target per minute
SEV_NOISE = 0.010                    # brownian noise per sqrt(minute)
SEV_UNTREATED_DRIFT = 0.0012         # upward drift per minute before provider seen
SEV_TREATED_TARGET = 0.08            # severity target once treatment underway
DETERIORATION_THRESHOLD = 0.80       # crossing this -> acuity upgrade event
CODE_THRESHOLD = 0.97                # crossing this -> code/arrest
REASSESS_INTERVAL_ROOM = 30.0
REASSESS_INTERVAL_WR = 45.0          # waiting-room recheck cadence (when triage nurse free)

# ------------------------------------------------------------------ reneging -
# Left-without-being-seen: patience (minutes) lognormal by acuity
LWBS_PATIENCE = {1: (1e9, .1), 2: (600, .5), 3: (240, .5), 4: (150, .6), 5: (110, .6)}

# ------------------------------------------------------------------- reward --
REWARD = {
    "wait_per_min_by_acuity": {1: -2.0, 2: -0.8, 3: -0.30, 4: -0.12, 5: -0.08},
    "boarding_per_min": -0.05,
    "los_per_min": -0.02,
    "lwbs": {1: -500, 2: -300, 3: -120, 4: -40, 5: -20},
    "deterioration": -150.0,
    "code": -600.0,
    "death": -2000.0,
    "discharge": +60.0,
    "admit_handoff": +80.0,
    "psych_no_sitter_per_min": -1.5,   # SI/HI/section in a room without a sitter
    "extra_nurse": -EXTRA_NURSE_COST, # 40.0
    
}
#wait_per_mi_acuity reward is -2.0 for acuity level 1? (almost death
#but lwbs is -500?
"""
problems with reward configuration
code? if coding patient - critical patient arrives, randomness will penalize the 
system?
psych_no_sitter_per_min has to be a bigger reward - 
#wait per min acuity needs a bigger reward 

"""
EPISODE_DAYS = 3
SEED = 42
