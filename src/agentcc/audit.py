"""Independent reconstruction of freshness and validity from snapshots, delivery times and the revision ledger."""

from __future__ import annotations

from bisect import bisect_right


def check(rows, extra, condition):
    ledger, applied = extra["ledger"], extra["applied_step"]
    mismatches = {"fresh": 0, "valid": 0, "pending": 0, "stale_c2_c3": 0, "invalid_c2": 0}
    for r in rows:
        if not r["delivered"]:
            continue
        t, step = r["end"], r["deliver_step"]
        fresh = valid = True
        pending = False
        for o, v in r["snapshot"]:
            if bisect_right(applied[o], step) != v:
                fresh = False
            unreflected = [inv for created, inv in ledger[o][v:] if created <= t]
            if any(unreflected):
                valid = False
                if bisect_right(applied[o], step) == v:
                    pending = True
        mismatches["fresh"] += fresh != r["fresh"]
        mismatches["valid"] += valid != r["valid"]
        mismatches["pending"] += (pending and fresh) != bool(r["pending_exposure"])
        if condition in ("C2", "C3") and not fresh:
            mismatches["stale_c2_c3"] += 1
        if condition == "C2" and not valid:
            mismatches["invalid_c2"] += 1
    return mismatches
