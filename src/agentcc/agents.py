"""Tutor and validator agents and the two model backends (local Ollama, OpenRouter)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

LOCAL_MODEL = "qwen2.5:7b-instruct"
API_MODEL = "google/gemini-3.1-flash-lite"
MAX_TOKENS = 400
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = Path(os.environ.get("OPENROUTER_KEY_FILE", Path.home() / ".config" / "qos-llm" / "openrouter.key"))

TUTOR_SYSTEM = (
    "You are a mathematics tutor in an adaptive assessment system. You never see the answer key."
)
TUTOR_USER = """Problem:
{item}

The learner chose: {answer}

Reference passages:
{passages}

Write a short explanation (at most 120 words) of how to solve the problem and whether the learner's choice is right.
Cite a reference passage as [P<number>] only where you use it. The last line must be: Final answer: <letter A-E>"""

VALIDATOR_SYSTEM = "You check explanations written by a tutor agent. Reply with JSON only."
VALIDATOR_USER = """Problem:
{item}

Reference passages:
{passages}

Explanation to check:
{explanation}

Reply with JSON {{"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}}. PASS only if the explanation is internally consistent,
its arithmetic is correct and its final answer follows from it."""


def format_passages(passages):
    return "\n\n".join(f"[P{p['pid']}] {p['text'][:900]}" for p in passages)


def final_letter(text):
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    m = re.fullmatch(r"\**final answer\s*:\s*\**\s*\(?([A-E])\)?\s*\.?\**", lines[-1], re.I) if lines else None
    return m.group(1).upper() if m else None


def cited(text):
    return sorted({int(x) for x in re.findall(r"\[P(\d+)\]", text)})


def schema_ok(explanation, passages):
    allowed = {p["pid"] for p in passages}
    return final_letter(explanation) is not None and set(cited(explanation)) <= allowed


def parse_verdict(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        v = {}
    return str(v.get("verdict", "")).upper() == "PASS"


class Ollama:
    name = "local"

    def __init__(self, model=LOCAL_MODEL):
        self.model = model

    def chat(self, system, user):
        t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_predict": MAX_TOKENS},
        }, timeout=600)
        r.raise_for_status()
        d = r.json()
        return {
            "text": d["message"]["content"], "seconds": time.perf_counter() - t0,
            "prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"), "cost": 0.0,
            "provider": "ollama", "model": self.model,
        }

    def digest(self):
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30).json()
        return next((m["digest"] for m in r["models"] if m["name"] == self.model), None)


class OpenRouter:
    name = "api"

    def __init__(self, model=API_MODEL, budget_usd=2.0):
        self.model, self.budget, self.spent = model, budget_usd, 0.0
        self.key = KEY_FILE.read_text().strip()

    def chat(self, system, user):
        if self.spent >= self.budget:
            raise RuntimeError(f"API budget of {self.budget} USD reached")
        t0 = time.perf_counter()
        r = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {self.key}"}, json={
            "model": self.model, "temperature": 0, "max_tokens": MAX_TOKENS, "seed": 0,
            "reasoning": {"effort": "minimal"}, "usage": {"include": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, timeout=600)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {})
        cost = float(u.get("cost") or 0.0)
        self.spent += cost
        return {
            "text": d["choices"][0]["message"]["content"] or "", "seconds": time.perf_counter() - t0,
            "prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": cost, "provider": d.get("provider"), "model": d.get("model"),
        }


def tutor(backend, item_text, answer, passages):
    out = backend.chat(TUTOR_SYSTEM, TUTOR_USER.format(item=item_text, answer=answer, passages=format_passages(passages)))
    out["role"] = "tutor"
    return out


def validator(backend, item_text, passages, explanation):
    out = backend.chat(VALIDATOR_SYSTEM, VALIDATOR_USER.format(
        item=item_text, passages=format_passages(passages), explanation=explanation))
    out["role"] = "validator"
    out["pass"] = parse_verdict(out["text"]) and schema_ok(explanation, passages)
    return out
