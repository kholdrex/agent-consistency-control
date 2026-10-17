"""Experiment 2b: descriptive quality of the API tutor and validator on the 200 evaluation items."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .agents import cited, final_letter, tutor, validator
from .data import LETTERS, item_text, load

LEARNER_SEED = 4243


def run(backend, corpus, out: Path):
    test = load("test.json")
    passages = {p["pid"]: p for p in corpus["passages"]}
    g = np.random.default_rng(LEARNER_SEED)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["bank_id"] for x in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a", encoding="utf-8") as f:
        for b in corpus["bank"]:
            item = test[b["test_index"]]
            answer = item["correct"] if g.random() < 0.6 else str(g.choice([x for x in LETTERS if x != item["correct"]]))
            if b["bank_id"] in done:
                continue
            ps = [passages[p] for p in b["deps"]]
            t = tutor(backend, item_text(item), answer, ps)
            v = validator(backend, item_text(item), ps, t["text"])
            f.write(json.dumps({
                "bank_id": b["bank_id"], "learner_answer": answer, "key": item["correct"],
                "final": final_letter(t["text"]), "cited": cited(t["text"]), "supplied": b["deps"],
                "validator_pass": v["pass"], "tutor": t, "validator": v,
            }, ensure_ascii=False) + "\n")
            f.flush()
    return out
