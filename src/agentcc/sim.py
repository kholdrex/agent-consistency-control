"""Discrete-event simulation of feedback requests served by language-model agents under three consistency disciplines.

C0: C1 without deduplication of response events (diagnostic only).
C1: learner state checked; item and evidence versions are not rechecked before delivery.
C2: optimistic control: item and evidence versions rechecked at delivery, one regeneration, then abstention.
C3: pessimistic control: shared locks on the item and evidence from retrieval to delivery; revisions wait for the locks.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

CONDITIONS = ("C0", "C1", "C2", "C3")
GRID = np.arange(-4.0, 4.0 + 1e-9, 0.05)
LOG_PRIOR = -0.5 * GRID**2

T_BANK, T_THETA, T_START, T_RT, T_CORRECT, T_RESP_FAULT, T_REVISION, T_SERVICE, T_PASS, T_GATE_FAULT = range(10)
R_REVISION, R_RELEASE, R_COMMIT, R_GATE, R_WORKER, R_REQUEST = range(6)


def rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *key]))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Config:
    condition: str
    seed: int
    bank_seed: int = 2026
    world: str = "W1"
    fault: float = 0.3
    rho_item: float = 0.02
    rho_passage: float = 0.05
    p_invalidating: float = 0.5
    n_learners: int = 30
    n_decisions: int = 30
    start_window: float = 600.0
    n_items: int = 200
    n_passages: int = 300
    passages_per_item: int = 3
    zipf: float = 1.1
    rt_median: float = 45.0
    rt_sigma: float = 0.5
    deadline: float = 60.0
    drain: float = 300.0
    workers: int = 4
    max_calls: int = 6
    target: float = 0.65
    dup_offset: float = 10.0
    delay_mean: float = 10.0
    core_time: float = 0.01
    repair_pass_prob: float | None = None


@dataclass(frozen=True)
class Timing:
    name: str
    tutor: tuple[float, ...]
    validator: tuple[float, ...]
    pass_prob: float

    @staticmethod
    def from_profile(path):
        import json

        d = json.loads(open(path).read())
        return Timing(d["backend"], tuple(d["tutor"]), tuple(d["validator"]), float(d["pass_prob"]))

    @staticmethod
    def synthetic(name="dev", tutor_median=4.0, validator_median=2.0, sigma=0.4, pass_prob=0.9, n=200, seed=1):
        g = np.random.default_rng(seed)
        return Timing(
            name,
            tuple(np.round(tutor_median * np.exp(sigma * g.standard_normal(n)), 3)),
            tuple(np.round(validator_median * np.exp(sigma * g.standard_normal(n)), 3)),
            pass_prob,
        )


@dataclass
class Bank:
    b: np.ndarray
    a: np.ndarray
    deps: list[tuple[int, ...]]


def make_bank(cfg: Config, item_deps=None) -> Bank:
    """item_deps: passage IDs per item from the evidence corpus; without them, a synthetic Zipf graph (development)."""
    g = rng(cfg.bank_seed, T_BANK)
    b = g.uniform(-2.0, 2.0, cfg.n_items)
    a = g.uniform(0.5, 2.0, cfg.n_items)
    if item_deps is not None:
        return Bank(b, a, [(k, *sorted(cfg.n_items + int(p) for p in item_deps[k])) for k in range(cfg.n_items)])
    w = 1.0 / np.arange(1, cfg.n_passages + 1) ** cfg.zipf
    w /= w.sum()
    deps = []
    for k in range(cfg.n_items):
        ps = g.choice(cfg.n_passages, size=cfg.passages_per_item, replace=False, p=w)
        deps.append((k, *sorted(cfg.n_items + int(p) for p in ps)))
    return Bank(b, a, deps)


def true_prob(cfg: Config, bank: Bank, theta: float, items) -> np.ndarray:
    b = bank.b[items]
    if cfg.world == "W1":
        return sigmoid(theta - b)
    return 0.2 + 0.75 * sigmoid(bank.a[items] * (theta - b))


def response_fault(cfg: Config, u: int, d: int) -> tuple[str, float]:
    g = rng(cfg.seed, T_RESP_FAULT, u, d)
    x, y = g.random(), g.random()
    if x < cfg.fault / 2:
        return "dup", y * cfg.dup_offset
    if x < cfg.fault:
        return "delay", -cfg.delay_mean * np.log1p(-y)
    return "none", 0.0


def learner_timeline(cfg: Config):
    """Presentation and commit times do not depend on the condition: they use only per-(learner, decision) draws."""
    starts, ends = [], []
    for u in range(cfg.n_learners):
        t = rng(cfg.seed, T_START, u).uniform(0.0, cfg.start_window)
        starts.append(t)
        for d in range(cfg.n_decisions):
            t += response_time(cfg, u, d)
            kind, extra = response_fault(cfg, u, d)
            t += (extra if kind == "delay" else 0.0) + cfg.core_time
        ends.append(t)
    return starts, ends


def response_time(cfg: Config, u: int, d: int) -> float:
    return float(cfg.rt_median * np.exp(cfg.rt_sigma * rng(cfg.seed, T_RT, u, d).standard_normal()))


def revision_ledger(cfg: Config, t_end: float, n_objects: int, active=None):
    ledger = []
    for o in range(n_objects):
        rate = (cfg.rho_item if o < cfg.n_items else cfg.rho_passage) / 3600.0
        if active is not None and o not in active:
            rate = 0.0
        g = rng(cfg.seed, T_REVISION, o)
        t, revs = 0.0, []
        while rate > 0:
            t += g.exponential(1.0 / rate)
            if t > t_end:
                break
            revs.append((t, bool(g.random() < cfg.p_invalidating)))
        ledger.append(revs)
    return ledger


@dataclass
class Request:
    id: int
    learner: int
    decision: int
    item: int
    deps: tuple[int, ...]
    arrival: float
    attempt: int = 0
    calls: int = 0
    started: int = 0
    completed_calls: int = 0
    repaired: bool = False
    snapshot: dict = field(default_factory=dict)
    held: list = field(default_factory=list)
    lock_pos: int = 0
    regenerations: int = 0
    unchanged_regeneration: bool = False
    end: float | None = None
    deliver_step: int | None = None
    delivered: bool = False
    fresh: bool | None = None
    valid: bool | None = None
    pending_exposure: bool | None = None
    outcome: str = "late"


class Simulation:
    def __init__(self, cfg: Config, timing: Timing, item_deps=None):
        if cfg.condition not in CONDITIONS:
            raise ValueError(cfg.condition)
        self.cfg, self.timing = cfg, timing
        self.bank = make_bank(cfg, item_deps)
        self.n_objects = cfg.n_items + cfg.n_passages
        starts, ends = learner_timeline(cfg)
        self.starts = starts
        self.horizon = max(ends) + cfg.drain
        active = {o for d in self.bank.deps for o in d}
        self.ledger = revision_ledger(cfg, self.horizon, self.n_objects, active)
        self.version = [0] * self.n_objects
        self.applied_at = [[] for _ in range(self.n_objects)]
        self.applied_step = [[] for _ in range(self.n_objects)]
        self.step = 0
        self.readers = [set() for _ in range(self.n_objects)]
        self.writers = [deque() for _ in range(self.n_objects)]
        self.waiting = [[] for _ in range(self.n_objects)]
        self.free = cfg.workers
        self.jobs = deque()
        self.heap, self.seq = [], 0
        g = [rng(cfg.seed, T_THETA, u) for u in range(cfg.n_learners)]
        bands = [(-2.0, -2.0 / 3), (-2.0 / 3, 2.0 / 3), (2.0 / 3, 2.0)]
        self.theta = [g[u].uniform(*bands[u * 3 // cfg.n_learners]) for u in range(cfg.n_learners)]
        self.logpost = [LOG_PRIOR.copy() for _ in range(cfg.n_learners)]
        self.presented = [np.zeros(cfg.n_items, bool) for _ in range(cfg.n_learners)]
        self.current_item = [None] * cfg.n_learners
        self.committed = [set() for _ in range(cfg.n_learners)]
        self.commit_count = [0] * cfg.n_learners
        self.regret = []
        self.requests: list[Request] = []

    def push(self, t, rank, fn, *args):
        self.seq += 1
        heapq.heappush(self.heap, (t, rank, self.seq, fn, args))

    def run(self):
        for u, t in enumerate(self.starts):
            self.push(t, R_COMMIT, self.present, u, 0)
        for o, revs in enumerate(self.ledger):
            for j, (t, _) in enumerate(revs):
                self.push(t, R_REVISION, self.revision_created, o, j)
        while self.heap:
            t, _, _, fn, args = heapq.heappop(self.heap)
            if t > self.horizon:
                break
            self.step += 1
            fn(t, *args)
        return self.summary_records()

    # learners and decision core

    def present(self, t, u, d):
        cfg, bank = self.cfg, self.bank
        post = np.exp(self.logpost[u] - self.logpost[u].max())
        post /= post.sum()
        eligible = np.flatnonzero(~self.presented[u])
        pred = (post[:, None] * sigmoid(GRID[:, None] - bank.b[eligible][None, :])).sum(axis=0)
        k = int(eligible[np.argmin(np.abs(pred - cfg.target))])
        p_true = true_prob(cfg, bank, self.theta[u], eligible)
        best = np.min((p_true - cfg.target) ** 2)
        mine = (true_prob(cfg, bank, self.theta[u], [k])[0] - cfg.target) ** 2
        self.regret.append(float(mine - best))
        self.presented[u][k] = True
        self.current_item[u] = k
        correct = rng(cfg.seed, T_CORRECT, u, d).random() < true_prob(cfg, bank, self.theta[u], [k])[0]
        sent = t + response_time(cfg, u, d)
        kind, extra = response_fault(cfg, u, d)
        arrive = sent + (extra if kind == "delay" else 0.0)
        self.push(arrive, R_COMMIT, self.commit, u, d, k, bool(correct), False)
        if kind == "dup":
            self.push(arrive + extra, R_COMMIT, self.commit, u, d, k, bool(correct), True)

    def commit(self, t, u, d, k, correct, is_copy):
        cfg = self.cfg
        if (u, d) in self.committed[u] and cfg.condition != "C0":
            return
        self.committed[u].add((u, d))
        self.commit_count[u] += 1
        p = sigmoid(GRID - self.bank.b[k])
        self.logpost[u] += np.log(p if correct else 1.0 - p)
        if is_copy:
            return
        r = Request(u * cfg.n_decisions + d, u, d, k, self.bank.deps[k], t)
        self.requests.append(r)
        self.push(t + cfg.core_time, R_REQUEST, self.start_request, r)
        if d + 1 < cfg.n_decisions:
            self.push(t + cfg.core_time, R_COMMIT, self.present, u, d + 1)

    # content revisions

    def revision_created(self, t, o, j):
        if self.cfg.condition != "C3":
            self.apply_revision(t, o)
            return
        self.writers[o].append(j)
        self.drain_writers(t, o)

    def apply_revision(self, t, o):
        self.version[o] += 1
        self.applied_at[o].append(t)
        self.applied_step[o].append(self.step)

    def drain_writers(self, t, o):
        while self.writers[o] and not self.readers[o]:
            self.writers[o].popleft()
            self.apply_revision(t, o)
        if not self.writers[o] and self.waiting[o]:
            waiting, self.waiting[o] = self.waiting[o], []
            for r in waiting:
                self.acquire(t, r)

    # request lifecycle

    def start_request(self, t, r):
        r.snapshot = {o: self.version[o] for o in r.deps}
        r.repaired = False
        if self.cfg.condition == "C3":
            r.lock_pos, r.held = 0, []
            self.acquire(t, r)
        else:
            self.call(t, r, "tutor")

    def acquire(self, t, r):
        while r.lock_pos < len(r.deps):
            o = r.deps[r.lock_pos]
            if self.writers[o]:
                self.waiting[o].append(r)
                return
            self.readers[o].add(r.id)
            r.held.append(o)
            r.lock_pos += 1
        r.snapshot = {o: self.version[o] for o in r.deps}
        self.call(t, r, "tutor")

    def release(self, t, r):
        held, r.held = r.held, []
        for o in held:
            self.readers[o].discard(r.id)
            if self.writers[o] and not self.readers[o]:
                self.push(t, R_RELEASE, self.drain_writers, o)

    def call(self, t, r, role):
        if r.calls >= self.cfg.max_calls:
            self.finish(t, r, "abstain_ceiling")
            return
        r.calls += 1
        if self.free > 0:
            self.free -= 1
            self.begin(t, r, role, r.calls)
        else:
            self.jobs.append((r, role, r.calls))

    def begin(self, t, r, role, n):
        r.started += 1
        pool = self.timing.tutor if role == "tutor" else self.timing.validator
        g = rng(self.cfg.seed, T_SERVICE, r.id, n)
        self.push(t + pool[int(g.integers(len(pool)))], R_WORKER, self.done, r, role, n)

    def done(self, t, r, role, n):
        r.completed_calls += 1
        if self.jobs:
            nr, nrole, nn = self.jobs.popleft()
            self.begin(t, nr, nrole, nn)
        else:
            self.free += 1
        if role == "tutor":
            self.call(t, r, "validator")
            return
        p_pass = self.timing.pass_prob
        if r.repaired and self.cfg.repair_pass_prob is not None:
            p_pass = self.cfg.repair_pass_prob
        if rng(self.cfg.seed, T_PASS, r.id, n).random() < p_pass:
            g = rng(self.cfg.seed, T_GATE_FAULT, r.id, n)
            x, y = g.random(), g.random()
            delay = -self.cfg.delay_mean * np.log1p(-y) if x < self.cfg.fault / 2 else 0.0
            self.push(t + delay, R_GATE, self.gate, r)
        elif not r.repaired:
            r.repaired = True
            self.call(t, r, "tutor")
        else:
            self.finish(t, r, "abstain_validator")

    def gate(self, t, r):
        stale = any(self.version[o] != r.snapshot[o] for o in r.deps)
        if self.cfg.condition == "C2" and stale:
            if r.attempt == 0:
                r.attempt, r.regenerations = 1, 1
                r.unchanged_regeneration = not self.unreflected_invalidating(t, r)
                self.start_request(t, r)
            else:
                self.finish(t, r, "abstain_stale")
            return
        r.delivered = True
        r.deliver_step = self.step
        r.fresh = not stale
        unreflected = self.unreflected(t, r)
        r.valid = not any(inv for _, inv in unreflected)
        r.pending_exposure = any(inv for _, inv in unreflected) and not stale
        self.finish(t, r, "delivered")

    def unreflected(self, t, r):
        out = []
        for o in r.deps:
            out += [rev for rev in self.ledger[o][r.snapshot[o]:] if rev[0] <= t]
        return out

    def unreflected_invalidating(self, t, r):
        return any(inv for _, inv in self.unreflected(t, r))

    def finish(self, t, r, outcome):
        r.end, r.outcome = t, outcome
        if self.cfg.condition == "C3":
            self.release(t, r)

    # results

    def summary_records(self):
        cfg = self.cfg
        rows = []
        for r in self.requests:
            finished = r.end is not None
            rows.append({
                "id": r.id, "learner": r.learner, "decision": r.decision, "item": r.item,
                "arrival": r.arrival, "end": r.end, "latency": r.end - r.arrival if finished else None,
                "censored": not finished, "outcome": r.outcome if finished else "unfinished",
                "delivered": r.delivered, "fresh": r.fresh, "valid": r.valid, "pending_exposure": r.pending_exposure,
                "timely": r.delivered and r.end - r.arrival <= cfg.deadline,
                "budget_calls": r.calls, "started_calls": r.started, "completed_calls": r.completed_calls,
                "regenerations": r.regenerations, "unchanged_regeneration": r.unchanged_regeneration,
                "snapshot": [[o, r.snapshot[o]] for o in r.deps] if r.delivered else None,
                "deliver_step": r.deliver_step,
            })
        first = min(r.arrival for r in self.requests)
        last = max(r.arrival for r in self.requests)
        cohort = []
        for o, revs in enumerate(self.ledger):
            for j, (created, inv) in enumerate(revs):
                if first <= created <= last:
                    applied = self.applied_at[o][j] if j < len(self.applied_at[o]) else None
                    cohort.append({"object": o, "created": created, "invalidating": inv, "applied": applied})
        rmse = float(np.sqrt(np.mean([
            ((np.exp(lp - lp.max()) / np.exp(lp - lp.max()).sum()) @ GRID - th) ** 2
            for lp, th in zip(self.logpost, self.theta, strict=True)
        ])))
        extra = {
            "regret": float(np.mean(self.regret)), "rmse": rmse,
            "double_commits": int(sum(self.commit_count) - sum(len(c) for c in self.committed)),
            "revision_cohort": cohort, "revisions": sum(len(v) for v in self.ledger), "horizon": self.horizon,
            "ledger": self.ledger, "applied_at": self.applied_at, "applied_step": self.applied_step,
        }
        return rows, extra


def simulate(cfg: Config, timing: Timing, item_deps=None):
    return Simulation(cfg, timing, item_deps).run()
