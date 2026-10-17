"""Analysis of the simulation grid (primary contrast, cell table, checks) and of Experiment 2b."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

T975_19 = 2.093024054408263  # two-sided 95 % quantile of Student's t with 19 degrees of freedom
BOOT = 10000


def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def interval(values):
    v = np.asarray(values, float)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    return [float(m), float(m - T975_19 * se), float(m + T975_19 * se)]


def boot_interval(values, seed=1):
    v = np.asarray(values, float)
    g = np.random.default_rng(seed)
    means = v[g.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
    return [float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def index(rows):
    cell = {}
    for r in rows:
        cell[(r["load"], r["lambda"], r["condition"], r["seed"])] = r
    return cell


def primary(rows):
    cell = index(rows)
    seeds = sorted({r["seed"] for r in rows})
    loads = sorted({r["load"] for r in rows})
    lams = sorted(x for x in {r["lambda"] for r in rows} if x > 0)
    per_seed = [np.mean([cell[(ld, lam, "C2", s)]["tvf"] - cell[(ld, lam, "C3", s)]["tvf"]
                         for ld in loads for lam in lams]) for s in seeds]
    return {"estimand": "mean over 8 cells (lambda > 0 x 2 loads) of TVF(C2) - TVF(C3)", "replications": len(seeds),
            "t_interval": interval(per_seed), "bootstrap_interval": boot_interval(per_seed), "per_seed": per_seed}


MEASURES = ("tvf", "invalid_per_1000", "abstained", "censored", "latency_median", "latency_p95", "calls_per_request",
            "regenerations_per_request", "pending_exposure_per_1000", "correction_delay_median", "correction_delay_p95",
            "corrections_censored", "corrections")


def cells(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["load"], r["lambda"], r["condition"])].append(r)
    out = []
    for (ld, lam, c), rs in sorted(groups.items()):
        row = {"load": ld, "learners": rs[0]["learners"], "lambda": lam, "condition": c, "n": len(rs)}
        for m in MEASURES:
            vals = [r[m] for r in rs if r[m] is not None]
            row[m] = interval(vals) if len(vals) > 1 else (vals[0] if vals else None)
        out.append(row)
    return out


def contrasts(rows):
    cell = index(rows)
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for ld in sorted({r["load"] for r in rows}):
        for lam in sorted({r["lambda"] for r in rows}):
            entry = {"load": ld, "lambda": lam}
            for a, b in (("C2", "C1"), ("C2", "C3"), ("C3", "C1")):
                entry[f"tvf_{a}_minus_{b}"] = interval([cell[(ld, lam, a, s)]["tvf"] - cell[(ld, lam, b, s)]["tvf"]
                                                        for s in seeds])
            out.append(entry)
    return out


def checks(rows):
    audit = {k: sum(r[k] for r in rows) for k in rows[0] if k.startswith("audit_")}
    cell = index(rows)
    seeds = sorted({r["seed"] for r in rows})
    keys = ("tvf", "invalid_per_1000", "latency_p95", "calls_per_request")
    identical = all(cell[(ld, 0.0, "C1", s)][k] == cell[(ld, 0.0, c, s)][k]
                    for ld in sorted({r["load"] for r in rows}) for s in seeds for c in ("C2", "C3") for k in keys)
    return {"audit_mismatches": audit, "lambda0_conditions_identical": identical}


def quality(path):
    rows = load_rows(path)
    n = len(rows)
    return {
        "items": n,
        "final_answer_accuracy": sum(r["final"] == r["key"] for r in rows) / n,
        "final_answer_missing": sum(r["final"] is None for r in rows) / n,
        "validator_pass": sum(r["validator_pass"] for r in rows) / n,
        "explanations_citing_none": sum(not r["cited"] for r in rows) / n,
        "citations_outside_supplied": sum(bool(set(r["cited"]) - set(r["supplied"])) for r in rows) / n,
        "pass_given_correct": float(np.mean([r["validator_pass"] for r in rows if r["final"] == r["key"]])),
        "pass_given_wrong": float(np.mean([r["validator_pass"] for r in rows if r["final"] != r["key"]])),
        "tutor_seconds_median": float(np.median([r["tutor"]["seconds"] for r in rows])),
        "cost_usd": float(sum(r["tutor"]["cost"] + r["validator"]["cost"] for r in rows)),
    }


def figure(cell_rows, path):
    """Fig. 1 of the paper at print width (17 cm): TVF (a, b) and 95th percentile latency (c, d) by correction rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    styles = {"C1": ("o", "-", "0.0"), "C2": ("s", "--", "0.3"), "C3": ("^", ":", "0.45")}
    names = {"C1": "C1, state only", "C2": "C2, optimistic", "C3": "C3, pessimistic"}
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(17 / 2.54, 11.5 / 2.54), sharex=True)
    plain = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for j, ld in enumerate((0.5, 0.85)):
        for c, (mk, ls, col) in styles.items():
            rs = sorted((r for r in cell_rows if r["load"] == ld and r["condition"] == c and r["lambda"] > 0),
                        key=lambda r: r["lambda"])
            x = [r["lambda"] for r in rs]
            axes[0, j].plot(x, [100 * r["tvf"][0] for r in rs], marker=mk, ls=ls, color=col, label=names[c], ms=5,
                            lw=1.4)
            axes[1, j].plot(x, [r["latency_p95"][0] for r in rs], marker=mk, ls=ls, color=col, ms=5, lw=1.4)
        axes[0, j].set_title(f"({'ab'[j]}) utilisation {ld}", fontsize=11)
        axes[1, j].set_title(f"({'cd'[j]}) utilisation {ld}", fontsize=11)
        axes[1, j].set_xlabel("Corrections per object-hour")
        if ld == 0.85:
            axes[1, j].set_yscale("log")
        axes[1, j].yaxis.set_major_formatter(plain)
        axes[1, j].yaxis.set_minor_formatter(ticker.NullFormatter())
        for i in range(2):
            axes[i, j].set_xscale("log")
            axes[i, j].set_xticks([0.05, 0.5, 5, 50])
            axes[i, j].xaxis.set_major_formatter(plain)
            axes[i, j].xaxis.set_minor_formatter(ticker.NullFormatter())
            axes[i, j].grid(alpha=0.3, lw=0.5)
    axes[0, 0].set_ylabel("Timely valid feedback, %")
    axes[1, 0].set_ylabel("Latency p95, s")
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=600)


def report(root: Path):
    w1 = load_rows(root / "results/sim/grid_api_W1.jsonl")
    out = {"primary": primary(w1), "checks": checks(w1), "cells": cells(w1), "contrasts": contrasts(w1)}
    w2_path = root / "results/sim/grid_api_W2.jsonl"
    if w2_path.exists():
        w2 = load_rows(w2_path)
        out["w2"] = {"primary": primary(w2), "checks": checks(w2), "contrasts": contrasts(w2)}
    s_path = root / "results/sim/grid_api_W1_repair0.jsonl"
    if s_path.exists():
        s = load_rows(s_path)
        out["sensitivity_repair0"] = {"primary": primary(s), "checks": checks(s), "cells": cells(s),
                                      "contrasts": contrasts(s)}
    q = root / "results/quality/api.jsonl"
    if q.exists():
        out["quality"] = quality(q)
    (root / "results/summary.json").write_text(json.dumps(out, indent=1))
    (root / "figures").mkdir(exist_ok=True)
    figure(out["cells"], root / "figures/operating_map.png")
    return out
