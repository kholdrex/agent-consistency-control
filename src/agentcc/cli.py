"""Command line: rebuild the corpus, measure timing, run the grid and the quality check, write the summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentcc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus", help="rebuild data/corpus.json from AQuA-RAT")
    t = sub.add_parser("timing", help="measure agent timing on the development items")
    t.add_argument("backend", choices=["api", "local"])
    g = sub.add_parser("grid", help="run the simulation grid with the API timing profile")
    g.add_argument("--world", default="W1", choices=["W1", "W2"])
    g.add_argument("--repair-pass-prob", type=float, default=None)
    sub.add_parser("quality", help="run tutor and validator once on the 200 evaluation items (API)")
    sub.add_parser("report", help="write results/summary.json and figures/operating_map.png")
    a = ap.parse_args(argv)

    if a.cmd == "corpus":
        from .data import build

        (ROOT / "data/corpus.json").write_text(json.dumps(build(), ensure_ascii=False, indent=0))
        return
    corpus = json.loads((ROOT / "data/corpus.json").read_text())
    if a.cmd == "timing":
        from .agents import Ollama, OpenRouter
        from .timing import measure

        measure(OpenRouter() if a.backend == "api" else Ollama(), corpus, ROOT / "results/timing")
    elif a.cmd == "grid":
        from .experiment import run_grid
        from .sim import Timing

        tag = "" if a.repair_pass_prob is None else f"_repair{a.repair_pass_prob:g}"
        run_grid(corpus, Timing.from_profile(ROOT / "results/timing/api_profile.json"), ROOT / "results/sim",
                 world=a.world, repair_pass_prob=a.repair_pass_prob, tag=tag)
    elif a.cmd == "quality":
        from .agents import OpenRouter
        from .quality import run

        run(OpenRouter(budget_usd=2.0), corpus, ROOT / "results/quality/api.jsonl")
    elif a.cmd == "report":
        from .analysis import report

        report(ROOT)


if __name__ == "__main__":
    main()
