"""Experiment 1: the simulation grid and per-replication summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audit import check
from .sim import Config, Timing, simulate

LAMBDAS = (0.0, 0.05, 0.5, 5.0, 50.0)
LOADS = {0.5: 35, 0.85: 60}
SEEDS = tuple(range(1001, 1021))
WORKERS = 4


def corpus_deps(corpus):
    return [b["deps"] for b in corpus["bank"]]


def summarise(rows, extra):
    n = len(rows)
    done = [r for r in rows if not r["censored"]]
    lat = [r["latency"] for r in done]
    regen = sum(r["regenerations"] for r in rows)
    cohort = extra["revision_cohort"]
    delays = [c["applied"] - c["created"] for c in cohort if c["applied"] is not None]
    return {
        "requests": n,
        "tvf": sum(r["delivered"] and r["timely"] and r["valid"] for r in rows) / n,
        "invalid_per_1000": 1000 * sum(r["delivered"] and not r["valid"] for r in rows) / n,
        "stale_applied_per_1000": 1000 * sum(r["delivered"] and not r["fresh"] for r in rows) / n,
        "abstained": sum(r["outcome"].startswith("abstain") for r in rows) / n,
        "censored": sum(r["censored"] for r in rows) / n,
        "latency_median": float(np.median(lat)) if lat else None,
        "latency_p95": float(np.percentile(lat, 95)) if lat else None,
        "calls_per_request": sum(r["started_calls"] for r in rows) / n,
        "regenerations_per_request": regen / n,
        "harmless_regeneration_share": (sum(r["unchanged_regeneration"] for r in rows) / regen) if regen else None,
        "pending_exposure_per_1000": 1000 * sum(bool(r["pending_exposure"]) for r in rows) / n,
        "corrections": len(cohort),
        "correction_delay_median": float(np.median(delays)) if delays else None,
        "correction_delay_p95": float(np.percentile(delays, 95)) if delays else None,
        "corrections_censored": (len(cohort) - len(delays)) / len(cohort) if cohort else None,
        "double_commits": extra["double_commits"], "regret": extra["regret"], "rmse": extra["rmse"],
    }


def run_grid(corpus, timing: Timing, out: Path, world="W1", conditions=("C1", "C2", "C3"), seeds=SEEDS):
    deps = corpus_deps(corpus)
    n_passages = len(corpus["passages"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for load, learners in LOADS.items():
        for lam in LAMBDAS:
            conds = conditions + (("C0",) if lam == 0 and "C0" not in conditions else ())
            for c in conds:
                for s in seeds:
                    cfg = Config(condition=c, seed=s, world=world, rho_item=lam, rho_passage=lam,
                                 n_learners=learners, n_items=len(deps), n_passages=n_passages, workers=WORKERS)
                    req, extra = simulate(cfg, timing, deps)
                    receipt = check(req, extra, c)
                    rows.append({"timing": timing.name, "world": world, "load": load, "learners": learners,
                                 "lambda": lam, "condition": c, "seed": s, **summarise(req, extra),
                                 **{f"audit_{k}": v for k, v in receipt.items()}})
    path = out / f"grid_{timing.name}_{world}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path
