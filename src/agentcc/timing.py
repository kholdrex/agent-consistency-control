"""Experiment 2a: service times of the tutor and validator roles, validator pass rate and effective parallelism."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .agents import tutor, validator
from .data import LETTERS, item_text, load

WARMUP, N_SEQ, N_CONC, CONC = 3, 30, 32, 4
LEARNER_SEED = 4242


def dev_cases(corpus):
    test = load("test.json")
    passages = {p["pid"]: p for p in corpus["passages"]}
    g = np.random.default_rng(LEARNER_SEED)
    cases = []
    for it in corpus["dev"]:
        item = test[it["test_index"]]
        if g.random() < 0.6:
            answer = item["correct"]
        else:
            answer = str(g.choice([x for x in LETTERS if x != item["correct"]]))
        cases.append((item_text(item), answer, [passages[p] for p in it["deps"]]))
    return cases


def measure(backend, corpus, out_dir: Path):
    cases = dev_cases(corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / f"{backend.name}.jsonl").open("w", encoding="utf-8")

    def record(phase, i, res):
        row = {"phase": phase, "case": i, **{k: v for k, v in res.items()}}
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        return row

    for i in range(WARMUP):
        text, ans, ps = cases[i]
        record("warmup", i, tutor(backend, text, ans, ps))
    seq = []
    for n in range(N_SEQ):
        text, ans, ps = cases[n % len(cases)]
        t = record("sequential", n, tutor(backend, text, ans, ps))
        v = record("sequential", n, validator(backend, text, ps, t["text"]))
        seq.append((t, v))

    def one(n):
        text, ans, ps = cases[n % len(cases)]
        return tutor(backend, text, ans, ps)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(CONC) as ex:
        conc = list(ex.map(one, range(N_CONC)))
    wall = time.perf_counter() - t0
    for n, res in enumerate(conc):
        record("concurrent", n, res)
    log.close()

    tut = np.array([t["seconds"] for t, _ in seq])
    val = np.array([v["seconds"] for _, v in seq])
    seq_rate = 1.0 / tut.mean()
    conc_rate = N_CONC / wall
    profile = {
        "backend": backend.name, "model": getattr(backend, "model", None),
        "tutor": [round(float(x), 3) for x in tut], "validator": [round(float(x), 3) for x in val],
        "pass_prob": float(np.mean([v["pass"] for _, v in seq])),
        "sequential_tutor_rate": seq_rate, "concurrent_tutor_rate": conc_rate,
        "effective_parallelism": conc_rate / seq_rate, "concurrency": CONC,
        "cost_usd": float(sum(r.get("cost", 0.0) for pair in seq for r in pair)
                          + sum(r.get("cost", 0.0) for r in conc)),
        "median_tutor_output_tokens": float(np.median([t["output_tokens"] or 0 for t, _ in seq])),
    }
    (out_dir / f"{backend.name}_profile.json").write_text(json.dumps(profile, indent=1))
    return profile


def rescore(corpus, out_dir: Path, name: str):
    """Recompute the validator pass rate from the logged texts with the current parser (no new model calls)."""
    from .agents import parse_verdict, schema_ok

    cases = dev_cases(corpus)
    rows = [json.loads(x) for x in (out_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    seq = [r for r in rows if r["phase"] == "sequential"]
    tut = {r["case"]: r for r in seq if r["role"] == "tutor"}
    val = {r["case"]: r for r in seq if r["role"] == "validator"}
    passes = [parse_verdict(val[n]["text"]) and schema_ok(tut[n]["text"], cases[n % len(cases)][2])
              for n in sorted(tut)]
    path = out_dir / f"{name}_profile.json"
    profile = json.loads(path.read_text())
    profile["pass_prob_initial_parser"] = profile.get("pass_prob_initial_parser", profile["pass_prob"])
    profile["pass_prob"] = float(np.mean(passes))
    path.write_text(json.dumps(profile, indent=1))
    return profile["pass_prob_initial_parser"], profile["pass_prob"]
