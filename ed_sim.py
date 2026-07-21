"""
Event-driven Emergency Department simulation.

Pure Python + numpy (no SimPy needed). Time unit = minutes.
Designed to be driven by an RL agent through decision epochs:
    sim.reset() -> runs until first decision point
    sim.apply_action(...) / sim.run_until_decision()

The Gym-style wrapper lives in ed_env.py.
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import config as C


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
#rng expects a random number generator
#mean expects a float, sigma expects a float
#mean is desired average
#sigma is variance of distribution 
def lognorm(rng: np.random.Generator, mean: float, sigma: float) -> float:
    """Lognormal draw with the given (approximate) mean."""
    mu = math.log(max(mean, 1e-6)) - 0.5 * sigma * sigma
    return float(rng.lognormal(mu, sigma))


def sample_categorical(rng, dist: dict):
    keys = list(dist.keys())
    probs = np.array([dist[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return keys[rng.choice(len(keys), p=probs)]


# --------------------------------------------------------------------------
# patient
# --------------------------------------------------------------------------
PHASES = [
    "pre_triage", "in_triage", "waiting_room", "chairs",
    "roomed_wait_nurse", "nurse_assess", "wait_provider", "provider",
    "testing", "wait_results", "wait_dispo", "boarding",
    "psych_dispo", "discharge_process", "done",
]

PHASE_IDX = {p: i for i, p in enumerate(PHASES)}
#assignation of indexes for each phase 

@dataclass
class Patient:
    #variables for patient information 
    pid: int 
    t_arrival: float
    mode: str                      # "walkin" | "ambulance"
    acuity: int                    # observed ESI after triage (1..5)
    true_sev: float                # latent severity in [0,1]
    psych: Optional[str] = None    # None | medical_clearance | section | suicidal | homicidal
    agitation: float = 0.0

    #process information 
    phase: str = "pre_triage" #phase starts at pretriage
    room: Optional[int] = None 
    needs_bed: bool = True #starts at true 
    sitter: bool = False           # NA currently sitting 1:1
    needs_sitter: bool = False
    triage_suggestion: bool = False  # triage nurse gestalt: "get this one back"
    treated: bool = False          # provider has initiated care
    upgraded: int = 0              # times acuity was upgraded from deterioration
    coded: bool = False
    admit: Optional[bool] = None

    #time information 
    t_triaged: Optional[float] = None
    t_roomed: Optional[float] = None
    t_provider: Optional[float] = None
    t_dispo: Optional[float] = None
    t_depart: Optional[float] = None
    outcome: Optional[str] = None  # discharged | admitted | lwbs | death | psych_placed

    sev_last_t: float = 0.0 #?? what is this? 
    pending_tests: int = 0
    lwbs_deadline: float = 1e18 # if not treated patient leaves after 1e18
    patience: float = 1e18 #?? i dont understand this one 

    def wait_time(self, now):
        return now - self.t_arrival # now - arrival time 

    # ---- latent severity / vitals ------------------------------------
    def update_sev(self, now: float, rng: np.random.Generator):
        dt = now - self.sev_last_t # change in time - why sev_last_t?
        #still not understand variable sev_last_t
        if dt <= 0: #if the time has not changed 
            return # 
        target = C.SEV_TREATED_TARGET if self.treated else self.true_sev
        #if patient was treated, target becomes severity_treated_target. 
        #severity will drift towards target
        #if not treated, target will become true severity
        # patient will deteriorate 
        drift = 0.0 if self.treated else C.SEV_UNTREATED_DRIFT * (0.5 + self.true_sev)
        #if treated, patient wont drift
        #if treated, patient will drift at rate sev_untreated_drift 
        s = self.true_sev
        #s starts with true severity
        s += C.SEV_MEANREV * (target - s) * dt
        #moves severity towards target 
        s += drift * dt 
        #adds drift to the severity multiplied by change in time
        s += float(rng.normal(0, C.SEV_NOISE * math.sqrt(dt)))
        #creates a random number between severity noise multiplied by
        #change in time squared 
        self.true_sev = min(1.0, max(0.0, s))
        #clamps updated severity to stay between 1.0 and s

        # psych agitation grows while waiting, cools once roomed/treated
        if self.psych:
            g = 0.0016 if self.phase in ("waiting_room", "chairs") else -0.0025
            #what does g represents? agitation?
            self.agitation = min(1.0, max(0.0, self.agitation + g * dt
                                          + float(rng.normal(0, 0.01 * math.sqrt(dt)))))
            #if patient is psych clamps agitation between 1.0 and (agitation + g*dt + a random number
            #generated between 0.01 and dt squared)
            #if psych is in waiting room or chairs agitation becomes worst
        self.sev_last_t = now
        #still dont understand sev_last_t

    def vitals(self, rng: np.random.Generator) -> dict:
        """Observable vitals emitted from latent severity (with sensor noise)."""
        s = self.true_sev #severity 
        n = lambda sd: float(rng.normal(0, sd)) # noise generation
        return {
            "hr":   72 + 60 * s + n(4),
            "sbp":  120 - 45 * s + n(5),
            "rr":   14 + 16 * s + n(1.5),
            "spo2": 98 - 12 * s + n(0.8),
            "temp": 36.8 + 1.8 * s * (0.5 + n(0.2)),
        } #returns a dictionary of vitals of a patient
    #based on their severity and with random noise 

    def vitals_abnormality(self, rng) -> float:
        """Scalar 0..1 summary of how deranged the vitals look (what staff see)."""
        v = self.vitals(rng)
        score = 0.0
        score += max(0, (v["hr"] - 100) / 60) + max(0, (60 - v["hr"]) / 30)
        score += max(0, (100 - v["sbp"]) / 40)
        score += max(0, (v["rr"] - 20) / 15)
        score += max(0, (94 - v["spo2"]) / 10)
        return float(min(1.0, score / 3.0))


# --------------------------------------------------------------------------
# task servers (nurses, providers, techs)
# --------------------------------------------------------------------------
class TaskServer:
    """Single server with a priority queue of tasks."""

    def __init__(self, sim: "EDSim", name: str):
        self.sim = sim
        self.name = name
        self.queue: list = []           # (priority, seq, duration_fn, on_done)
        self.busy = False
        self.present = True             # staffing toggle (callouts)
        self._seq = 0

    def qlen(self):
        return len(self.queue) + (1 if self.busy else 0)

    def submit(self, priority: float, duration_fn: Callable[[], float],
               on_done: Callable[[], None]):
        heapq.heappush(self.queue, (priority, self._seq, duration_fn, on_done))
        self._seq += 1
        self._try_start()

    def _try_start(self):
        if self.busy or not self.present or not self.queue:
            return
        prio, _, dur_fn, on_done = heapq.heappop(self.queue)
        self.busy = True
        dur = max(0.5, dur_fn())
        def finish():
            self.busy = False
            try:
                on_done()
            finally:
                self._try_start()
        self.sim.schedule(self.sim.now + dur, finish)

    def set_present(self, present: bool):
        self.present = present
        if present:
            self._try_start()


# --------------------------------------------------------------------------
# main simulation
# --------------------------------------------------------------------------
class EDSim:
    K_CANDIDATES = 8    # waiting patients visible to the agent per decision

    def __init__(self, seed: int = C.SEED, episode_days: int = C.EPISODE_DAYS):
        self.seed = seed
        self.episode_days = episode_days

    # ------------------------------------------------------------ lifecycle
    def reset(self, seed: Optional[int] = None):

        if seed is None:
            s = int(np.random.randint(0, 2**32, dtype=np.uint32))
        else:
            s = seed


        self.rng = np.random.default_rng(s)
        self.rng_x = np.random.default_rng(s * 1_000_003 + 17)  # exogenous stream:
        # arrivals, patient attributes, waves, callouts — identical across policies
        self.now = 0.0
        self.t_end = self.episode_days * 24 * 60.0
        self._heap: list = []
        self._seq = 0
        self._pid = 0

        # rooms: rid -> patient or None ; "turnover" blocks briefly after vacate
        self.rooms = {r: None for r in range(1, C.NUM_ROOMS + 1)}
        self.room_blocked = {r: False for r in self.rooms}
        self.zone_of = {r: z for z, rooms in C.ZONES.items() for r in rooms}

        # staff
        self.zone_nurse = {z: TaskServer(self, f"nurse_{z}") for z in C.ZONES}
        self.triage_nurse = TaskServer(self, "triage")
        self.charge = TaskServer(self, "charge")
        self.doctor = TaskServer(self, "doctor")
        self.np_ = TaskServer(self, "np")
        self.lab = TaskServer(self, "lab")
        self.rad = TaskServer(self, "rad")
        self.ct = TaskServer(self, "ct")
        self.na_total = C.BASE_STAFF["nurse_assistant"]
        self.na_sitting = 0            # NAs consumed as 1:1 sitters
        self.charge_covering: Optional[str] = None   # zone charge nurse covers
        self.extra_nurses = 0          # agent-called float nurses on the floor

        # queues
        self.triage_queue: list[Patient] = []
        self.waiting: list[Patient] = []     # triaged, want a bed
        self.chairs: list[Patient] = []      # med-clearance being worked up chairside
        self.patients: dict[int, Patient] = {}
        self.active: set[int] = set()

        # env / exogenous state
        self.hosp_occ = C.HOSPITAL_OCC_MEAN
        self.day_mult = float(np.exp(self.rng_x.normal(0, C.DAILY_VOLUME_SIGMA)))
        self.wave_mult = 1.0
        self.max_rate = max(C.HOURLY_ARRIVALS) * C.ARRIVAL_SCALE / 60.0 * 2.5

        # bookkeeping
        self.reward_acc = 0.0 # resets reward to 0 
        self.last_accrual = 0.0
        self.decision_needed = True
        self.stats = {
            "arrivals": 0, "discharged": 0, "admitted": 0, "lwbs": 0,
            "deaths": 0, "codes": 0, "deteriorations": 0, "psych_placed": 0,
            "los_sum": 0.0, "los_n": 0, "door_to_room_sum": 0.0, "door_to_room_n": 0,
            "door_to_provider_sum": 0.0, "door_to_provider_n": 0,
            "boarding_min": 0.0,
        }
        self.reward_stats = {
            "wait_per_min_acuity_reward": 0.00,
            "boarding_per_minute_reward": 0.00,
            "los_per_min_reward": 0.00,
            "lwbs_reward": 0.00,
            "deterioration_reward": 0.00,
            "code_reward": 0.00,
            "death_reward": 0.00,
            "discharge_reward": 0.00,
            "admit_handoff_reward": 0.00,
            "psych_no_sitter_per_min_reward": 0.00,
            "extra_nurse_reward": 0.00
        }
        
        


        # kick things off
        self._schedule_next_arrival()
        self._schedule_daily_events()
        self.schedule(15.0, self._tick)
        for h in C.SHIFT_CHANGE_HOURS:
            self.schedule(h * 60.0, self._shift_change)
        self.run_until_decision()
        return self

    # ------------------------------------------------------------ event core
    def schedule(self, t: float, fn: Callable, *args):
        heapq.heappush(self._heap, (t, self._seq, fn, args))
        self._seq += 1

    def run_until_decision(self):
        self.decision_needed = False
        while self._heap and not self.decision_needed:
            t, _, fn, args = heapq.heappop(self._heap)
            if t > self.t_end:
                self.now = self.t_end
                self._accrue()
                return False           # episode over
            self._accrue_to(t)
            self.now = t
            fn(*args)
        return self.now < self.t_end

    def _request_decision(self):
        self.decision_needed = True

    @property
    def done(self):
        return self.now >= self.t_end

    # ------------------------------------------------------------ reward accrual
    def _accrue_to(self, t):
        self.now = max(self.now, self.now)  # no-op guard
        dt = t - self.last_accrual #change in time?
        if dt <= 0:
            return # no time passed 

        reward_by_acuity_waiting = 0.0
        reward_by_acuity_chairs = 0.0
        reward_boarding_per_min = 0.0
        reward_los_per_min_room = 0.0
        reward_no_sitter = 0.0
        for p in self.waiting: 
            # wait per min * change in time 
            reward_by_acuity_waiting += C.REWARD["wait_per_min_by_acuity"][p.acuity] * dt 
        for p in self.chairs:
            # wait per mi * change in time  
            reward_by_acuity_chairs += 0.5 * C.REWARD["wait_per_min_by_acuity"][p.acuity] * dt
        for pid in self.active:
            p = self.patients[pid]
            if p.phase == "boarding":
                # boarding per min * change in time
                reward_boarding_per_min += C.REWARD["boarding_per_min"] * dt
                self.stats["boarding_min"] += dt
            if p.room is not None: # if patient has a room
                # los per minute * change in time 
                reward_los_per_min_room += C.REWARD["los_per_min"] * dt
                if p.needs_sitter and not p.sitter:
                    # change in time 
                    
                    reward_no_sitter += C.REWARD["psych_no_sitter_per_min"] * dt
    
        self.reward_acc += reward_by_acuity_waiting + reward_by_acuity_chairs + reward_boarding_per_min + reward_los_per_min_room + reward_no_sitter
        self.reward_stats["wait_per_min_acuity_reward"] += reward_by_acuity_chairs + reward_by_acuity_waiting
        self.reward_stats["boarding_per_minute_reward"] += reward_boarding_per_min
        self.reward_stats["los_per_min_reward"] += reward_los_per_min_room #for patients that have room
        self.reward_stats["psych_no_sitter_per_min_reward"] += reward_no_sitter

        #rew_acc is the total reward for people waiting
        #just a sumation

        self.last_accrual = t

        #need to make a reward penalization for overcrowding nurses and rooms


    def _accrue(self):
        self._accrue_to(self.now)

    def pop_reward(self):
        self._accrue()
        r = self.reward_acc
        self.reward_acc = 0.0
        return r

    # ------------------------------------------------------------ arrivals
    def _rate_now(self):
        hour = int(self.now // 60) % 24
        return C.HOURLY_ARRIVALS[hour] * C.ARRIVAL_SCALE / 60.0 * self.day_mult * self.wave_mult

    def _schedule_next_arrival(self):
        # thinning against max rate
        t = self.now
        while True:
            t += float(self.rng_x.exponential(1.0 / self.max_rate))
            if self.rng_x.random() < min(1.0, (C.HOURLY_ARRIVALS[int(t // 60) % 24] * C.ARRIVAL_SCALE / 60.0
                                             * self.day_mult * self.wave_mult) / self.max_rate):
                break
        self.schedule(t, self._arrival)

    def _schedule_daily_events(self):
        for day in range(self.episode_days):
            base = day * 1440.0
            # crowding waves
            n_waves = self.rng_x.poisson(C.WAVE_RATE_PER_DAY)
            for _ in range(n_waves):
                t0 = base + self.rng_x.uniform(0, 1440)
                dur = lognorm(self.rng_x, C.WAVE_DURATION_MEAN, 0.4)
                mult = self.rng_x.uniform(*C.WAVE_MULTIPLIER)
                self.schedule(t0, self._set_wave, mult)
                self.schedule(t0 + dur, self._set_wave, 1.0)
            # EMS surges
            n_surge = self.rng_x.poisson(C.SURGE_RATE_PER_DAY)
            for _ in range(n_surge):
                t0 = base + self.rng_x.uniform(0, 1440)
                size = int(self.rng_x.integers(*C.SURGE_SIZE))
                self.schedule(t0, self._surge, size)
            # fresh day volume multiplier
            self.schedule(base, self._new_day)

    def _new_day(self):
        self.day_mult = float(np.exp(self.rng_x.normal(0, C.DAILY_VOLUME_SIGMA)))

    def _set_wave(self, mult):
        self.wave_mult = mult

    def _surge(self, size):
        for _ in range(size):
            self._arrival(force_mode="ambulance",
                          force_high_acuity=C.SURGE_HIGH_ACUITY, chained=True)

    def _arrival(self, force_mode=None, force_high_acuity=False, chained=False):
        if not chained:
            self._schedule_next_arrival()
        self._pid += 1
        mode = force_mode or ("ambulance" if self.rng_x.random() < C.AMBULANCE_FRACTION
                              else "walkin")
        dist = C.ACUITY_DIST_AMBO if mode == "ambulance" else C.ACUITY_DIST_WALKIN
        acuity = sample_categorical(self.rng_x, dist)
        if force_high_acuity:
            acuity = min(acuity, int(self.rng_x.integers(1, 3)))
        sev = C.SEV_INIT_BY_ACUITY[acuity] + float(self.rng_x.normal(0, C.SEV_INIT_NOISE))
        p = Patient(pid=self._pid, t_arrival=self.now, mode=mode, acuity=acuity,
                    true_sev=min(1.0, max(0.02, sev)))
        p.sev_last_t = self.now
        mean_pt, sig_pt = C.LWBS_PATIENCE[acuity]
        p.patience = lognorm(self.rng_x, mean_pt, sig_pt)
        if self.rng_x.random() < C.PSYCH_FRACTION:
            p.psych = sample_categorical(self.rng_x, C.PSYCH_TYPES)
            p.agitation = float(self.rng_x.uniform(0.1, 0.6))
            p.needs_sitter = p.psych in ("section", "suicidal", "homicidal")
        self.patients[p.pid] = p
        self.stats["arrivals"] += 1
        # triage queue (ambulances jump ahead of walk-ins)
        p.phase = "pre_triage"
        self.triage_queue.append(p)
        self.triage_queue.sort(key=lambda q: (0 if q.mode == "ambulance" else 1,
                                              q.t_arrival))
        self._pump_triage()

    # ------------------------------------------------------------ triage
    def _pump_triage(self):
        if not self.triage_queue:
            return
        if self.triage_nurse.busy:
            return
        p = self.triage_queue.pop(0)
        p.phase = "in_triage"
        dur = lambda: lognorm(self.rng, C.TRIAGE_TIME_MEAN, C.TRIAGE_TIME_SIGMA)
        self.triage_nurse.submit(0, dur, lambda: self._triage_done(p))

    def _triage_done(self, p: Patient):
        p.update_sev(self.now, self.rng)
        p.t_triaged = self.now
        # triage nurse gestalt: hidden severity leaks into a "room this one" flag
        gestalt = p.true_sev + self.rng.normal(0, C.TRIAGE_OVERRIDE_NOISE)
        expected = C.SEV_INIT_BY_ACUITY[p.acuity]
        p.triage_suggestion = (gestalt > expected + 0.15) or \
                              (self.rng.random() < C.TRIAGE_OVERRIDE_BASE)
        self.active.add(p.pid)

        # psych medical clearance: chairs track unless too agitated
        if p.psych == "medical_clearance" and p.agitation < C.AGITATION_BED_THRESHOLD:
            p.needs_bed = False
            p.phase = "chairs"
            self.chairs.append(p)
            self._start_chairs_workup(p)
        else:
            p.phase = "waiting_room"
            self.waiting.append(p)
            # LWBS clock (high acuity effectively never leaves)
            if p.psych is None:
                p.lwbs_deadline = self.now + p.patience
                self.schedule(p.lwbs_deadline, self._maybe_lwbs, p)
        # ESI-1: crash straight back if a trauma room is open (bypasses agent)
        if p.acuity == 1:
            r = self._free_room(prefer=C.TRAUMA_ROOMS)
            if r is not None:
                self._place(p, r)
        self._pump_triage()
        self._request_decision()

    def _maybe_lwbs(self, p: Patient):
        if p.phase == "waiting_room" and p.outcome is None \
                and self.now >= p.lwbs_deadline - 1e-6:
            self._depart(p, "lwbs")
            lwbs_reward = C.REWARD["lwbs"][p.acuity]
            self.reward_acc += lwbs_reward
            self.reward_stats["lwbs_reward"] += lwbs_reward
            self.stats["lwbs"] += 1
            self._request_decision()

    # ------------------------------------------------------------ chairs (med clearance)
    def _start_chairs_workup(self, p: Patient):
        # triage nurse draws clearance labs when free (low priority)
        def labs_done():
            self.schedule(self.now + lognorm(self.rng, *C.LAB_RESULT_TIME),
                          self._chairs_provider, p)
        self.triage_nurse.submit(5, lambda: lognorm(self.rng, *C.LAB_DRAW_TIME),
                                 labs_done)

    def _chairs_provider(self, p: Patient):
        if p.phase != "chairs":
            return
        srv = self.np_ if self.np_.qlen() <= self.doctor.qlen() else self.doctor
        def evaluated():
            if p.phase != "chairs":
                return
            p.treated = True
            if p.t_provider is None:
                p.t_provider = self.now
                self._rec_provider(p)
            dispo = lognorm(self.rng, C.PSYCH_DISPO_TIME_MEAN, C.PSYCH_DISPO_TIME_SIGMA)
            self.schedule(self.now + dispo, self._psych_leave, p)
        srv.submit(p.acuity, lambda: lognorm(self.rng, *C.PROVIDER_EVAL_TIME[p.acuity]),
                   evaluated)

    def _psych_leave(self, p: Patient):
        # only valid for patients still on the chairs track; if they were roomed
        # in the meantime, the roomed pathway (_dispo -> psych_dispo) handles exit
        if p.outcome is not None or p.phase != "chairs":
            return
        self._depart(p, "psych_placed")
        self.stats["psych_placed"] += 1
        discharge_reward = C.REWARD["discharge"]
        self.reward_acc += discharge_reward
        self.reward_stats["discharge_reward"] += discharge_reward

        self._request_decision()



    # ------------------------------------------------------------ rooming
    def _free_room(self, prefer=None):
        pool = (prefer or []) + [r for r in self.rooms if not prefer or r not in prefer]
        for r in pool:
            if self.rooms[r] is None and not self.room_blocked[r] \
                    and self._zone_staffed(self.zone_of[r]):
                return r
        return None

    def _zone_staffed(self, z): #changes in staff 
        return self.zone_nurse[z].present or self.charge_covering == z

    def zone_census(self, z):
        return sum(1 for pid in self.active
                   if (room := self.patients[pid].room) is not None
                   and self.zone_of[room] == z)
    #zone census  - very important for overcrowding reward 

    def _nurse_dur(self, z, base_mean, base_sig):
        def f():
            load = self.zone_census(z)
            over = max(0, load - C.NURSE_SAFE_LOAD)
            slow = 1.0 + C.NURSE_OVERLOAD_SLOWDOWN * over
            if self.extra_nurses > 0:
                slow = max(1.0, slow - 0.3 * self.extra_nurses)
            return lognorm(self.rng, base_mean, base_sig) * slow
        return f

    def _zone_server(self, z) -> TaskServer:
        ns = self.zone_nurse[z]
        if not ns.present and self.charge_covering == z:
            return self.charge
        # charge helps out when a staffed zone is drowning
        if ns.qlen() >= 3 and not self.charge.busy and self.charge_covering is None:
            return self.charge
        return ns

    def can_place(self, p: Patient, r: int) -> bool:
        if p.phase not in ("waiting_room", "chairs"):
            return False
        if self.rooms[r] is not None or self.room_blocked[r]:
            return False
        if not self._zone_staffed(self.zone_of[r]):
            return False
        return True

    def _place(self, p: Patient, r: int):
        """Move a waiting patient into room r and start the roomed pathway."""
        if p in self.waiting:
            self.waiting.remove(p)
        if p in self.chairs:
            self.chairs.remove(p)
        self.rooms[r] = p.pid
        p.room = r
        p.t_roomed = self.now
        p.phase = "roomed_wait_nurse"
        self.stats["door_to_room_sum"] += p.wait_time(self.now)
        self.stats["door_to_room_n"] += 1
        # 1:1 sitter for section / SI / HI
        if p.needs_sitter and self.na_sitting < self.na_total:
            self.na_sitting += 1
            p.sitter = True
        z = self.zone_of[r]
        self._zone_server(z).submit(
            p.acuity,
            self._nurse_dur(z, *C.NURSE_ASSESS_TIME[p.acuity]),
            lambda: self._nurse_assess_done(p))

    def _nurse_assess_done(self, p: Patient):
        if p.outcome is not None:
            return
        p.phase = "wait_provider"
        # provider routing: MD takes 1-2, NP takes 4-5, ESI-3 to shorter queue
        if p.acuity <= 2: #if patient is dying or deteriorating, server is doctor 
            srv = self.doctor
        elif p.acuity >= 4:
            srv = self.np_ if self.np_.qlen() <= self.doctor.qlen() + 2 else self.doctor
        else:
            srv = self.np_ if self.np_.qlen() < self.doctor.qlen() else self.doctor
        def seen():
            self._provider_done(p)
        srv.submit(p.acuity, lambda: lognorm(self.rng, *C.PROVIDER_EVAL_TIME[p.acuity]),
                   seen)

    def _rec_provider(self, p):
        self.stats["door_to_provider_sum"] += p.t_provider - p.t_arrival
        self.stats["door_to_provider_n"] += 1

    def _provider_done(self, p: Patient):
        if p.outcome is not None:
            return
        p.update_sev(self.now, self.rng)
        p.treated = True
        if p.t_provider is None:
            p.t_provider = self.now
            self._rec_provider(p)
        p.phase = "testing"
        # order diagnostics
        tests = []
        if self.rng.random() < C.LAB_PROB[p.acuity]:
            tests.append("lab")
        if self.rng.random() < C.XRAY_PROB[p.acuity]:
            tests.append("xray")
        if self.rng.random() < C.CT_PROB[p.acuity]:
            tests.append("ct")
        p.pending_tests = len(tests)
        if not tests:
            self._to_dispo(p)
            return
        for t in tests:
            if t == "lab":
                def draw_done(p=p):
                    self.schedule(self.now + lognorm(self.rng, *C.LAB_RESULT_TIME),
                                  self._test_done, p)
                self.lab.submit(p.acuity, lambda: lognorm(self.rng, *C.LAB_DRAW_TIME),
                                draw_done)
            elif t == "xray":
                self.rad.submit(p.acuity, lambda: lognorm(self.rng, *C.XRAY_TIME),
                                lambda p=p: self._test_done(p))
            else:
                self.ct.submit(p.acuity, lambda: lognorm(self.rng, *C.CT_TIME),
                               lambda p=p: self._test_done(p))

    def _test_done(self, p: Patient):
        if p.outcome is not None:
            return
        p.pending_tests -= 1
        if p.pending_tests <= 0:
            p.phase = "wait_results"
            srv = self.doctor if p.acuity <= 2 else \
                (self.np_ if self.np_.qlen() < self.doctor.qlen() else self.doctor)
            srv.submit(p.acuity + 0.5,
                       lambda: lognorm(self.rng, *C.RESULT_REVIEW_TIME),
                       lambda: self._to_dispo(p))

    def _to_dispo(self, p: Patient):
        if p.outcome is not None:
            return
        p.phase = "wait_dispo"
        srv = self.doctor if p.acuity <= 2 else \
            (self.np_ if self.np_.qlen() < self.doctor.qlen() else self.doctor)
        srv.submit(p.acuity + 1,
                   lambda: lognorm(self.rng, *C.DISPO_DECISION_TIME),
                   lambda: self._dispo(p))

    def _dispo(self, p: Patient):
        if p.outcome is not None:
            return
        p.t_dispo = self.now
        # psych bedded patients: wait for placement, then leave
        if p.psych in ("section", "suicidal", "homicidal") or \
                (p.psych == "medical_clearance" and p.room is not None):
            p.phase = "psych_dispo"
            dispo = lognorm(self.rng, C.PSYCH_DISPO_TIME_MEAN, C.PSYCH_DISPO_TIME_SIGMA)
            self.schedule(self.now + dispo, self._finish_psych_bed, p)
            return
        p.admit = self.rng.random() < C.ADMIT_PROB[p.acuity] or p.coded
        if p.admit:
            p.phase = "boarding"
            extra = C.BOARDING_TIME_MAX_EXTRA * max(0.0, (self.hosp_occ - 0.7)) / 0.3
            bt = lognorm(self.rng, C.BOARDING_TIME_BASE + extra, 0.5)
            self.schedule(self.now + bt, self._bed_ready, p)
        else:
            p.phase = "discharge_process"
            z = self.zone_of[p.room] if p.room else "A"
            self._zone_server(z).submit(
                p.acuity + 2,
                self._nurse_dur(z, *C.DISCHARGE_PROCESS_TIME),
                lambda: self._discharged(p))

    def _finish_psych_bed(self, p: Patient):
        if p.outcome is not None:
            return
        self._vacate(p, "psych_placed")
        self.stats["psych_placed"] += 1
        discharge_reward = C.REWARD["discharge"]
        self.reward_acc += discharge_reward #add discharge reward to reward_acc 
        self.reward_stats["discharge_reward"] += discharge_reward



    def _bed_ready(self, p: Patient):
        if p.outcome is not None:
            return
        self._vacate(p, "admitted")
        self.stats["admitted"] += 1
        admit_handoff_reward = C.REWARD["admit_handoff"]
        self.reward_acc += admit_handoff_reward #add admitted reward to reward_acc
        self.reward_stats["admit_handoff_reward"] += admit_handoff_reward
    def _discharged(self, p: Patient):
        if p.outcome is not None:
            return
        self._vacate(p, "discharged")
        self.stats["discharged"] += 1
        discharge_reward = C.REWARD["discharge"]
        self.reward_acc += discharge_reward #add discharge reward to reward_acc 
        self.reward_stats["discharge_reward"] += discharge_reward
    # ------------------------------------------------------------ departures
    def _vacate(self, p: Patient, outcome: str):
        r = p.room
        self._depart(p, outcome)
        if r is not None:
            self.rooms[r] = None
            self.room_blocked[r] = True
            self.schedule(self.now + 10.0, self._turnover_done, r)  # clean/turnover

    def _turnover_done(self, r):
        self.room_blocked[r] = False
        self._request_decision()

    def _depart(self, p: Patient, outcome: str):
        p.update_sev(self.now, self.rng)
        p.outcome = outcome
        p.t_depart = self.now
        p.phase = "done"
        if p.sitter:
            self.na_sitting -= 1
            p.sitter = False
        self.active.discard(p.pid)
        if p in self.waiting:
            self.waiting.remove(p)
        if p in self.chairs:
            self.chairs.remove(p)
        if outcome in ("discharged", "admitted", "psych_placed"):
            self.stats["los_sum"] += p.t_depart - p.t_arrival
            self.stats["los_n"] += 1

    # ------------------------------------------------------------ ticks / dynamics
    def _tick(self):
        self.schedule(self.now + 15.0, self._tick)
        # hospital occupancy random walk (mean-reverting)
        self.hosp_occ += 0.05 * (C.HOSPITAL_OCC_MEAN - self.hosp_occ) \
            + float(self.rng.normal(0, C.HOSPITAL_OCC_VOL))
        self.hosp_occ = min(0.995, max(0.5, self.hosp_occ))
        # update severities; check deterioration / codes / agitation escalation
        for pid in list(self.active):
            p = self.patients[pid]
            p.update_sev(self.now, self.rng)
            # med-clearance patient boils over -> now needs a bed
            if p.phase == "chairs" and p.agitation >= C.AGITATION_BED_THRESHOLD:
                p.needs_bed = True #if psych patient gets agitated 
                p.phase = "waiting_room"
                self.chairs.remove(p)
                self.waiting.append(p)
            if p.true_sev >= C.CODE_THRESHOLD and not p.coded:
                self._code(p) #if true severity is > code_treshold 
            elif p.true_sev >= C.DETERIORATION_THRESHOLD and p.acuity > 1 \
                    and p.true_sev > C.SEV_INIT_BY_ACUITY[p.acuity] + 0.15:
                p.acuity = max(1, p.acuity - 1)
                p.upgraded += 1
                self.stats["deteriorations"] += 1
                deterioration_reward = C.REWARD["deterioration"]
                self.reward_acc += deterioration_reward #reward = reward plus deterioration reward
                self.reward_stats["deterioration_reward"] += deterioration_reward
        
        self._request_decision()

    def _code(self, p: Patient):
        p.coded = True
        self.stats["codes"] += 1

        code_reward = C.REWARD["code"]
        self.reward_stats["code_reward"] += code_reward
        self.reward_acc += code_reward
        death_p = 0.35 if p.room is None else 0.12
        if self.rng.random() < death_p:
            was_roomed = p.room is not None
            r = p.room
            self._depart(p, "death")
            if was_roomed:
                self.rooms[r] = None
                self.room_blocked[r] = True
                self.schedule(self.now + 30.0, self._turnover_done, r)
            death_reward = C.REWARD["death"]
            self.stats["deaths"] += 1
            self.reward_acc += death_reward # reward = reward + death reward 
            self.reward_stats["death_reward"] += death_reward

            return
        # survived: emergency rooming if in WR
        p.true_sev = 0.70
        p.acuity = 1
        p.treated = True
        if p.room is None:
            r = self._free_room(prefer=C.TRAUMA_ROOMS)
            if r is not None:
                self._place(p, r)
        p.admit = True

    # ------------------------------------------------------------ staffing dynamics
    def _shift_change(self):
        self.schedule(self.now + 12 * 60.0, self._shift_change)
        self.charge_covering = None
        self.charge.set_present(True)
        for z, ns in self.zone_nurse.items():
            ns.set_present(True)
        # callouts
        called_out = [z for z in C.ZONES if self.rng_x.random() < C.NURSE_CALLOUT_PROB]
        for z in called_out:
            self.zone_nurse[z].set_present(False)
            if self.charge_covering is None:
                self.charge_covering = z     # charge covers first gap
            if self.rng_x.random() < C.CALLOUT_BACKFILL_PROB:
                delay = lognorm(self.rng_x, C.CALLOUT_BACKFILL_DELAY_MEAN, 0.4)
                self.schedule(self.now + delay, self._backfill, z)
        self.extra_nurses = 0
        self._request_decision()

    def _backfill(self, z):
        self.zone_nurse[z].set_present(True)
        if self.charge_covering == z:
            self.charge_covering = None

    def agent_call_extra_nurse(self):
        extra_nurse_reward = C.REWARD["extra_nurse"]
        self.reward_acc += extra_nurse_reward # reward = reward + reward for calling an extra nurse 
        self.reward_stats["extra_nurse_reward"] += extra_nurse_reward
        self.schedule(self.now + C.EXTRA_NURSE_CALLIN_DELAY, self._extra_arrives)



    def _extra_arrives(self):
        self.extra_nurses += 1
        # extra nurse also re-opens the worst uncovered zone if any
        for z, ns in self.zone_nurse.items():
            if not ns.present and self.charge_covering != z:
                ns.set_present(True)
                self.extra_nurses -= 1
                break
        self._request_decision()

    # ------------------------------------------------------------ agent interface
    def waiting_candidates(self) -> list[Patient]:
        """Top-K waiting patients ranked by (acuity, triage flag, wait)."""
        pool = sorted(self.waiting,
                      key=lambda p: (p.acuity, not p.triage_suggestion,
                                     -(self.now - p.t_arrival)))
        return pool[: self.K_CANDIDATES]

    def free_rooms(self) -> list[int]:
        return [r for r in self.rooms
                if self.rooms[r] is None and not self.room_blocked[r]
                and self._zone_staffed(self.zone_of[r])]
    #returns list of all rooms if room is free and zone is staffed 

    def apply_place(self, p: Patient, r: int) -> bool: #add a patient to a room
        if self.can_place(p, r):
            self._place(p, r)
            return True
        return False

    def summary(self):
        s = self.stats
        return {
            "arrivals": s["arrivals"],
            "discharged": s["discharged"],
            "admitted": s["admitted"],
            "psych_placed": s["psych_placed"],
            "lwbs": s["lwbs"],
            "lwbs_rate_%": round(100 * s["lwbs"] / max(1, s["arrivals"]), 1),
            "deaths": s["deaths"],
            "codes": s["codes"],
            "deteriorations": s["deteriorations"],
            "avg_LOS_min": round(s["los_sum"] / max(1, s["los_n"]), 1),
            "avg_door_to_room_min": round(s["door_to_room_sum"] / max(1, s["door_to_room_n"]), 1),
            "avg_door_to_provider_min": round(s["door_to_provider_sum"] / max(1, s["door_to_provider_n"]), 1),
            "boarding_hours": round(s["boarding_min"] / 60.0, 1),
            "wait_per_min_by_acuity_reward": round(self.reward_stats["wait_per_min_acuity_reward"] , 2),
            "boarding_per_minute_reward": round(self.reward_stats["boarding_per_minute_reward"], 2),
            "los_per_minute_reward": round(self.reward_stats["los_per_min_reward"], 2),
            "lwbs_reward": round(self.reward_stats["lwbs_reward"], 2),
            "deterioration_reward": round(self.reward_stats["deterioration_reward"], 2),
            "code_reward": round(self.reward_stats["code_reward"], 2),
            "death_reward": round(self.reward_stats["death_reward"], 2),
            "discharge_reward": round(self.reward_stats["discharge_reward"], 2),
            "admin_handoff_reward": round(self.reward_stats["admit_handoff_reward"], 2),
            "psych_no_sitter_reward": round(self.reward_stats["psych_no_sitter_per_min_reward"], 2),
            "extra_nurse_reward": round(self.reward_stats["extra_nurse_reward"], 2),
            "total_reward": round(sum(self.reward_stats.values()), 2)
        }

