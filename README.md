# Consistency control for language-model agents in adaptive assessment

Code, data selection, measurements and results for the study *Consistency control for language-model agents in adaptive
assessment* (O. Kholodniak).

Tutor and validator agents prepare feedback on a learner's answer to an algebra item while the items and the reference
passages they depend on are being corrected. Three disciplines are compared on one request stream: checking only the learner
state (C1), rechecking the versions of the item and the passages at delivery with one regeneration (C2, optimistic) and locking
them while the agents work (C3, pessimistic). A discrete-event simulation uses agent response times measured with a cloud model
(Gemini 3.1 Flash-Lite through OpenRouter) and covers 0-50 corrections per object-hour at two loads with 20 replications;
an independent checker reconstructs every delivery.

## Layout

```
PROTOCOL.md          study protocol, frozen before the evaluation runs (tag protocol-freeze), with the deviations
freeze/manifest.json SHA-256 of the frozen files
src/agentcc/         data.py (AQuA-RAT items, audit, evidence corpus, BM25 dependency sets), agents.py (prompts, Ollama and
                     OpenRouter clients), timing.py (timing measurement), sim.py (simulation), audit.py (independent
                     checker), experiment.py (grid), quality.py (agent output quality), analysis.py (summary, figure), cli.py
data/corpus.json     audit, splits, evidence passages and dependency sets (built by `agentcc corpus`)
results/timing/      logged calls and timing profiles of both backends; OpenRouter's model record
results/sim/         one line per replication, condition and cell: grid_api_W1.jsonl (primary), grid_api_W2.jsonl,
                     grid_api_W1_repair0.jsonl (sensitivity: repeated validation always fails)
results/quality/     tutor and validator outputs on the 200 evaluation items
results/summary.json primary contrast, cell table, checks, agent quality
figures/             operating_map.png
```

The grid files hold per-replication aggregates and the checker's mismatch counts; delivery-level histories are not stored and
are reproduced by rerunning the simulator, which is deterministic.

## Usage

Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```
uv venv && uv pip install -e ".[dev]"
agentcc corpus          # downloads AQuA-RAT at the pinned commit into data/raw (not redistributed)
agentcc grid            # about 6 minutes on an Apple M1 Pro
agentcc grid --world W2
agentcc report
pytest
```

`agentcc timing api` and `agentcc quality` call the OpenRouter API (key file ~/.config/qos-llm/openrouter.key or the path in
OPENROUTER_KEY_FILE) and cost a few cents; `agentcc timing local` needs a running Ollama with qwen2.5:7b-instruct.

## Data and licences

AQuA-RAT (Apache 2.0), commit 26b64ed1, by Ling et al. (2017). Code: MIT. Results and figures: CC BY 4.0.
