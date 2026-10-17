"""AQuA-RAT items, audit, splits and the versioned evidence corpus with BM25 dependency sets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

AQUA_COMMIT = "26b64ed1f22208c6742f47df547dfd65088ec30d"
AQUA_URL = "https://raw.githubusercontent.com/google-deepmind/AQuA/{commit}/{name}"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SPLIT_SEED = 20261017
POOL_SIZE = 20000
TOP_K = 3
N_DEV, N_EVAL = 20, 200
LETTERS = "ABCDE"


def load(name):
    path = RAW / name
    if not path.exists():
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(AQUA_URL.format(commit=AQUA_COMMIT, name=name), timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text, n=5):
    t = tokens(text)
    return {tuple(t[i:i + n]) for i in range(max(len(t) - n + 1, 1))}


def stated_letter(rationale):
    tail = rationale.strip().splitlines()[-1] if rationale.strip() else ""
    m = re.search(r"(?:answer|ans)\s*(?:is|:|-|=)?\s*\(?([A-E])\)?\s*\.?\s*$", tail, re.I)
    if m:
        return m.group(1).upper()
    m = re.fullmatch(r"\s*\(?([A-E])\)?\s*\.?\s*", tail)
    return m.group(1).upper() if m else None


def audit_reason(item):
    opts = item.get("options", [])
    if len(opts) != 5 or [o[:1].upper() for o in opts] != list(LETTERS):
        return "options"
    if item.get("correct") not in LETTERS:
        return "key"
    if len(item.get("question", "").strip()) < 20:
        return "short"
    letter = stated_letter(item.get("rationale", ""))
    if letter is not None and letter != item["correct"]:
        return "rationale_key_conflict"
    return None


def build():
    """Deterministic construction of items, splits, evidence corpus and dependency sets."""
    test, train = load("test.json"), load("train.json")
    audit, kept, seen = [], [], set()
    for i, it in enumerate(test):
        reason = audit_reason(it)
        key = " ".join(tokens(it["question"]))
        if reason is None and key in seen:
            reason = "duplicate"
        seen.add(key)
        audit.append({"test_index": i, "excluded": reason})
        if reason is None:
            kept.append(i)
    g = np.random.default_rng(SPLIT_SEED)
    order = [kept[j] for j in g.permutation(len(kept))]
    dev, ev = order[:N_DEV], order[N_DEV:N_DEV + N_EVAL]
    items = {i: test[i] for i in dev + ev}

    target_sh = {i: shingles(items[i]["question"]) for i in items}
    index = {}
    for i, sh in target_sh.items():
        for s in sh:
            index.setdefault(s, set()).add(i)
    pool, dropped = [], 0
    for j in g.permutation(len(train)):
        if len(pool) >= POOL_SIZE:
            break
        it = train[int(j)]
        if audit_reason(it) is not None or not (150 <= len(it["rationale"]) <= 900):
            continue
        sh = shingles(it["question"])
        hits = {}
        for s in sh:
            for i in index.get(s, ()):
                hits[i] = hits.get(i, 0) + 1
        if any(c / max(len(sh | target_sh[i]), 1) >= 0.3 for i, c in hits.items()):
            dropped += 1
            continue
        pool.append(int(j))

    docs = [f"Problem: {train[j]['question']}\nSolution: {train[j]['rationale']}" for j in pool]
    bm25 = BM25Okapi([tokens(d) for d in docs], k1=1.5, b=0.75)
    retrieved = {}
    for i in dev + ev:
        scores = bm25.get_scores(tokens(items[i]["question"]))
        top = np.argsort(-scores, kind="stable")[:TOP_K]
        retrieved[i] = [pool[int(t)] for t in top]
    used = sorted({j for v in retrieved.values() for j in v})
    pid = {j: n for n, j in enumerate(used)}
    passages = [
        {"pid": pid[j], "train_index": j, "text": f"Problem: {train[j]['question']}\nSolution: {train[j]['rationale']}"}
        for j in used
    ]
    bank = []
    for n, i in enumerate(ev):
        bank.append({"bank_id": n, "test_index": i, "deps": [pid[j] for j in retrieved[i]]})
    dev_items = [{"dev_id": n, "test_index": i, "deps": [pid[j] for j in retrieved[i]]} for n, i in enumerate(dev)]
    return {
        "aqua_commit": AQUA_COMMIT, "split_seed": SPLIT_SEED, "pool_size": len(pool),
        "pool_dropped_near_duplicates": dropped,
        "audit": audit, "dev": dev_items, "bank": bank, "passages": passages,
    }


def item_text(item):
    return item["question"].strip() + "\n" + "\n".join(o.strip() for o in item["options"])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
