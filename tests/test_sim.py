from agentcc.sim import Config, Timing, simulate

TIMING = Timing.synthetic()


def run(condition, **kw):
    kw.setdefault("n_learners", 12)
    kw.setdefault("n_decisions", 10)
    return simulate(Config(condition=condition, seed=7, **kw), TIMING)


def test_deterministic():
    a, _ = run("C2", rho_item=5, rho_passage=5)
    b, _ = run("C2", rho_item=5, rho_passage=5)
    assert a == b


def test_no_revisions_c1_c2_c3_identical():
    rows = [run(c, rho_item=0, rho_passage=0)[0] for c in ("C1", "C2", "C3")]
    assert rows[0] == rows[1] == rows[2]


def test_optimistic_and_pessimistic_never_deliver_stale():
    for c in ("C2", "C3"):
        rows, _ = run(c, rho_item=50, rho_passage=50)
        assert all(r["fresh"] for r in rows if r["delivered"])


def test_state_only_delivers_stale_under_heavy_churn():
    rows, _ = run("C1", rho_item=50, rho_passage=50)
    assert any(r["delivered"] and not r["fresh"] for r in rows)


def test_deduplication_and_double_commits():
    _, c0 = run("C0", fault=0.6)
    _, c1 = run("C1", fault=0.6)
    assert c0["double_commits"] > 0
    assert c1["double_commits"] == 0


def test_validity_counts_pending_corrections_in_c3():
    rows, _ = run("C3", rho_item=50, rho_passage=50)
    for r in rows:
        if r["delivered"] and r["pending_exposure"]:
            assert r["fresh"] and not r["valid"]


def test_call_budget_respected():
    rows, _ = run("C2", rho_item=50, rho_passage=50)
    assert max(r["budget_calls"] for r in rows) <= 6
    assert all(r["started_calls"] <= r["budget_calls"] for r in rows)


def test_same_request_stream_across_conditions():
    ids = [[(r["id"], r["arrival"]) for r in run(c, rho_item=5, rho_passage=5)[0]] for c in ("C1", "C2", "C3")]
    assert ids[0] == ids[1] == ids[2]
