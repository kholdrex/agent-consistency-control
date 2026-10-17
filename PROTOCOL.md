# Protocol: consistency control for language-model agents that prepare feedback in adaptive assessment

Version 4, written before the evaluation runs. Earlier drafts (v1-v3) and their reviews are kept outside this repository.

## 1. Question and scope

Language-model agents prepare feedback on a learner's answer while the item bank and the evidence corpus they rely on are being corrected,
and while messages between components are delayed or duplicated. Three disciplines are compared on one request stream:

- C1 state-only: duplicate response events are ignored; item and evidence versions are not rechecked before delivery.
- C2 optimistic: at delivery the gate rechecks the versions of the item and the cited evidence; stale work is regenerated once with the
  current versions, then abandoned.
- C3 pessimistic: shared locks on the item and its evidence from retrieval to delivery; corrections wait until the locks are released.
- C0 (diagnostic only): C1 without deduplication of response events.

The question is how the share of feedback requests that end with timely and valid feedback, the latency, the model calls per request and
the delay of corrections depend on the discipline, the intensity of corrections and the load. Scope: feedback consistency under an
exogenous request stream. Next-item selection is an atomic step of the decision core; assessment quality (regret, ability error) is
reported only as a diagnostic and is identical for C1-C3 by construction.

## 2. Materials

- Items: AQuA-RAT (Apache 2.0) at commit 26b64ed, test split; automatic audit (five options A-E, valid key, question of at least 20
  characters, no duplicate question, no conflict between the key and a letter stated at the end of the rationale): 251 of 254 kept; a
  seeded permutation (seed 20261017) gives 20 development and 200 evaluation items (the bank).
- Evidence: 20,000 training-split problems with rationales of 150-900 characters, drawn by the same permutation after removing 18 near
  duplicates of the 220 selected items (5-word shingle Jaccard >= 0.3); BM25 (k1 = 1.5, b = 0.75) over "Problem + Solution" texts; the
  top 3 passages per item form its evidence; the union is the versioned corpus (625 passages for the 220 items; the 200 evaluation items
  reference 568 of them, 28 passages by more than one item). Dependency contract: feedback depends on the item and on all three supplied
  passages, whether or not the explanation cites them (a conservative contract; in development 9 of 30 explanations cited none). Simulated
  invalidity is defined by this contract, not by demonstrated semantic dependence.
- Files: data/corpus.json (audit, splits, passages, dependency sets), built by agentcc.data.build().

## 3. Agents and measured timing (Experiment 2a, development items only)

- Tutor and validator prompts in src/agentcc/agents.py; agents never see the key. The validator passes an explanation if its JSON verdict
  is PASS and the explanation states a final letter and cites only provided passages.
- Backends: API model alias google/gemini-3.1-flash-lite via OpenRouter (provider reported: Google; OpenRouter's model record retrieved
  2026-10-17 is saved in results/timing/openrouter_model.json; the alias does not pin a dated snapshot, which limits reproducibility),
  temperature 0, reasoning effort minimal, max 400 output tokens; local qwen2.5:7b-instruct (Ollama 0.34.3, Q4_K_M, digest recorded in
  results/timing/local_profile.json), temperature 0, max 400 output tokens.
- For each backend: 3 warm-up calls, then 30 sequential tutor + validator calls, then 32 tutor calls at concurrency 4.
- API profile (measured 2026-10-17): tutor mean 1.49 s (median 1.53 s), validator mean 1.00 s (median 1.02 s), validator pass rate 0.80
  (unchanged after the final-answer parser was made strict; pass rates are recomputed from the logged texts, no new calls), throughput ratio
  at concurrency 4 = 4.10 (no measurable contention), cost USD 0.04. Local profile: tutor mean 12.77 s, validator mean 5.36 s, validator pass
  rate 0.77 with the strict parser (0.83 with the initial one), throughput ratio 1.12 (results/timing/local_profile.json).
- The simulator draws service times from the 30 measured sequential values of each role; injected transport delays are added separately.

## 4. Simulation (Experiment 1)

Implementation: src/agentcc/sim.py (the code is part of this protocol). Main settings:

- Bank: the 200 evaluation items with their real dependency sets; difficulties b ~ U(-2, 2) and discriminations a ~ U(0.5, 2) drawn once
  with bank seed 2026 (synthetic stress-test parameters, unrelated to item content).
- Learners: abilities in three equal bands U(-2, -2/3), U(-2/3, 2/3), U(2/3, 2); 30 decisions each; start times U(0, 600) s; answer time
  LogNormal(median 45 s, sigma 0.5); response world W1 Rasch (primary) and W2 = 0.2 + 0.75 sigmoid(a (theta - b)) (descriptive).
- Decision core: grid posterior (-4..4, step 0.05, prior N(0, 1)); next item = unseen item with predicted success probability closest to
  0.65; it is the only writer of the learner record.
- Feedback request: retrieve (version snapshot) -> tutor -> validator -> [one repair: tutor -> validator] -> transport -> gate; at most
  6 model calls; FIFO pool of K = 4 workers; a request waiting for a lock holds no worker.
- Faults (reduced model, fixed for all cells): a response message is duplicated (copy after U(0, 10) s) with probability 0.15 or delayed
  (Exp, mean 10 s) with probability 0.15; an agent result is delayed (Exp, mean 10 s) with probability 0.15; consumers are idempotent.
- Corrections: Poisson process per item and per passage with rate lambda per object-hour; each correction invalidating with probability 0.5
  (it makes earlier text wrong) or harmless; corrections do not change difficulty or retrieval sets. C1/C2 apply a correction at its
  creation; C3 applies it when no lock is held on the object; waiting writers block new readers of that object.
- Deadline: feedback is timely if delivered within 60 s of the request. Valid: no invalidating correction of any dependency was created
  before delivery and not reflected in the delivered content (applied or pending). Horizon: last session end + 300 s; requests unfinished
  at the horizon count as not timely; their latency is censored.

Grid: lambda in {0, 0.05, 0.5, 5, 50}; load in {0.5, 0.85} nominal worker utilisation with the API profile = 35 and 60 learners per
replication with K = 4 (offered demand (1.4935 + 1.0005) x 1.2 = 2.993 worker-seconds per request including repairs; mean interval between
requests of one learner 45 exp(0.125) + 0.15 x 10 + 0.01 = 52.50 s; utilisation 0.499 and 0.855); conditions
C1, C2, C3 (C0 at lambda = 0 only); world W1; 20 replications with seeds 1001-1020 (development used seeds 1-99 and synthetic timing). The
same replication seed gives the same learners, answers, faults and corrections in every condition. Corrections are generated only for
objects referenced by evaluation items (200 items, 568 passages). Descriptive extension: W2. The local profile is not simulated: measured
on the laptop, it serves requests essentially one at a time (throughput ratio 1.12 at concurrency 4) with a mean demand of
(12.77 + 5.36) x 1.2 = 21.8 s per request, i.e. about 1.2 and 2.0 learners at utilisation 0.5 and 0.85; this capacity is reported
descriptively.

## 5. Measures (per replication, ratios of totals over all feedback requests)

TVF: delivered, timely and valid / all requests (primary outcome). Invalid deliveries per 1000 requests. Abstentions. Latency: median and
95th percentile over completed requests, with the censored share. Started model calls per request. Regenerations per request and the
share of regenerations caused only by harmless corrections. Correction delay: for corrections created between the first and last request
arrival, time to application; median and 95th percentile among corrections applied before the horizon, reported next to the censored share
(not full-cohort quantiles). Pending exposure (C3): deliveries with an invalidating correction
created but not yet applied. Diagnostics: double commits (C0), regret and ability RMSE.

## 6. Analysis

Primary: D = TVF(C2) - TVF(C3), averaged with equal weights over the 8 cells with lambda > 0 (4 lambda x 2 loads), computed within each
replication; mean over 20 replications with a two-sided 95 % t interval. All other contrasts, cells and measures are descriptive, reported
cell by cell with replication-level 95 % intervals. A contrast is called practically negligible only if its whole 95 % interval lies within
[-1, +1] percentage points of TVF (about 0.3 feedback events in a 30-decision session); an interval crossing either bound is inconclusive
about practical importance. Sanity checks: lambda = 0 gives identical C1-C3 results; an independent checker (src/agentcc/audit.py)
reconstructs freshness, validity and pending exposure for every delivery from the retained snapshots, event order and the correction
ledger; its mismatch counts are stored with each grid row and must be zero, and C2/C3 must have no stale deliveries. Development results led to the change from fixed
low correction rates to the lambda grid; this is reported in the paper.

## 7. Experiment 2b (descriptive agent quality)

On the 200 evaluation items, the API backend runs tutor and validator once per item with the learner answer drawn as in 2a (seed 4243):
final-answer accuracy against the key, validator pass rate, citation validity, tokens and cost. Budget cap USD 2.

## 8. Provenance and independence

All artefacts are new for this study: prompts, evidence corpus, simulator, timing measurements, figures and logs. Not used: the author's
FAHRAI platform, its forgetting and readiness models, simulators, item banks or results; recalibration, evidence regimes and invocation
allocation methods of the author's other studies; routing, placement, retrieval-ranking or caching contributions; GSM8K/GSM-Plus, XES3G5M,
ZNO and WebFAQ data; earlier measurements.

## 9. Deviations

Any change after the first evaluation run is listed here with its date and reason.

- 2026-10-17: src/agentcc/analysis.py (primary contrast, cell table, checks, figure) was written after the freeze and before any
  evaluation result was read; it implements section 6 as written. src/agentcc/quality.py is part of the frozen state (tag
  protocol-freeze).
- 2026-10-17, after an external review of the results: an exploratory sensitivity analysis assumes that the repeated validation after a
  repair always fails (Config.repair_pass_prob = 0), because a model run at temperature 0 tends to repeat its output; the main analysis
  is unchanged. Results: results/sim/grid_api_W1_repair0.jsonl.
