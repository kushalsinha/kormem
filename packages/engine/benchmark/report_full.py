"""Report a full LongMemEval run: accuracy, retrieval recall, context size, latency, cost and storage.

Reads the answer results, the official-judge file, the cached contexts and the
usage log of the run's ``KORMEM_HOME``. Phase times are epoch seconds (the
``*.start`` files a run script writes).

Usage:
    python -m benchmark.report_full --results out.json --official out.official_gpt-4o-2024-08-06.json \\
        --contexts ctx/ --home <KORMEM_HOME> --build-start T --answer-start T --judge-start T --end T
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path
from typing import Dict, List

from benchmark.common import datasetPath
from kormem.core.usage import summarize

# $ per million tokens: (input, cached input, output); embeddings are billed on characters / 4.
PRICE = {"gpt-6-luna": (0.10, 0.01, 0.50), "gpt-6.1-sol": (2.0, 2.0, 10.0),
         "gpt-4o-2024-08-06": (2.5, 1.25, 10.0), "gemini-embedding-2": (0.20, 0, 0)}
KS = (1, 3, 5, 10)


def cost(since: float, until: float, path: Path) -> Dict[str, Dict]:
    a, b = summarize(since, path), summarize(until, path)
    out = {}
    for k, v in a.items():
        d = {x: v.get(x, 0) - b.get(k, {}).get(x, 0) for x in v}
        if not d.get("calls"):
            continue
        model = k.split("|")[1]
        pi, pc, po = PRICE.get(model, (0, 0, 0))
        usd = (d.get("chars", 0) / 4 / 1e6 * pi if "embedding" in model else
               ((d["input"] - d["cachedInput"]) * pi + d["cachedInput"] * pc + d["output"] * po) / 1e6)
        out[k] = {**d, "usd": round(usd, 4)}
    return out


def pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("results", "official", "contexts", "home"):
        ap.add_argument(f"--{a}", required=True)
    for a in ("build-start", "answer-start", "judge-start", "end"):
        ap.add_argument(f"--{a}", type=float, required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    items = {it["question_id"].lower(): it for it in json.load(open(datasetPath(args.subset)))}
    results = {r["questionId"]: r for r in json.load(open(args.results))["results"]}
    official = json.load(open(args.official))
    verdict = {r["questionId"]: r["official"] for r in official["results"]}
    rep: Dict = {"n": len(results)}

    # ---- accuracy (official judge)
    rep["accuracy"] = {k: official[k] for k in ("accuracy", "correct", "total", "taskAveragedAccuracy",
                                                 "abstention", "byType")}

    # ---- retrieval: ranked candidates and the context sent to the answer model
    hits = {f"{m}@{k}": [] for m in ("any", "all") for k in KS}
    ctxAny, ctxAll, ctxSessions, recallSec, rounds, skipped = [], [], [], [], [], 0
    byType: Dict[str, List[bool]] = {}
    for qid, it in items.items():
        ctx = json.loads((Path(args.contexts) / f"{qid}.json").read_text())
        gold = set(it["answer_session_ids"])
        ranked = [re.sub(r"^chat ", "", c["title"]) for c in ctx["debug"].get("candidates", [])]
        for k in KS:
            hits[f"any@{k}"].append(bool(gold & set(ranked[:k])))
            hits[f"all@{k}"].append(gold <= set(ranked[:k]))
        inCtx = {re.sub(r"^chat ", "", s["title"]) for s in ctx["sources"]}
        ctxAny.append(bool(gold & inCtx))
        ctxAll.append(gold <= inCtx)
        byType.setdefault(it["question_type"], []).append(gold <= inCtx)
        ctxSessions.append(len(ctx["sources"]))
        if "recallSec" in ctx["debug"]:
            recallSec.append(ctx["debug"]["recallSec"])
        agent = ctx["debug"].get("agent") or {}
        if agent.get("skipped"):
            skipped += 1
        elif agent.get("rounds"):
            rounds.append(agent["rounds"])
    n = len(items)
    rep["retrieval"] = {
        **{k: round(sum(v) / n, 4) for k, v in hits.items()},
        "goldAnyInContext": round(sum(ctxAny) / n, 4), "goldAllInContext": round(sum(ctxAll) / n, 4),
        "goldAllInContextByType": {t: round(sum(v) / len(v), 4) for t, v in sorted(byType.items())},
        "contextSessionsMean": round(st.mean(ctxSessions), 2), "contextSessionsMedian": st.median(ctxSessions),
        "agentSkipped": skipped, "agentRoundsMean": round(st.mean(rounds), 2) if rounds else None,
    }

    # ---- latency
    total = [float(r["latencySec"]) for r in results.values() if "latencySec" in r]
    rep["latencySec"] = {
        "recall": {"n": len(recallSec), "p50": pct(recallSec, .5), "p90": pct(recallSec, .9),
                   "mean": round(st.mean(recallSec), 2) if recallSec else None},
        "recallPlusAnswer": {"n": len(total), "p50": round(pct(total, .5), 2), "p90": round(pct(total, .9), 2)},
    }
    rep["answerPromptChars"] = {"mean": round(st.mean(int(r["promptLen"]) for r in results.values()))}

    # ---- cost and time per phase
    log = Path(args.home) / "logs" / "usage.jsonl"
    phases = {"build": (args.build_start, args.answer_start), "answer": (args.answer_start, args.judge_start),
              "judge": (args.judge_start, args.end)}
    rep["cost"] = {}
    for name, (a, b) in phases.items():
        c = cost(a, b, log)
        rep["cost"][name] = {"usd": round(sum(v["usd"] for v in c.values()), 2), "wallMin": round((b - a) / 60, 1),
                             "byModel": c}
    ans = rep["cost"]["answer"]["byModel"].get("bench.answer|gpt-6.1-sol")
    if ans:
        rep["answerInputTokensMean"] = round(ans["input"] / ans["calls"])

    # ---- storage
    home = Path(args.home)
    vault = [p for p in (home / "vault").glob("lme_*") if p.is_dir()]
    vaultBytes = sum(f.stat().st_size for p in vault for f in p.rglob("*") if f.is_file())
    rep["storage"] = {"namespaces": len(vault), "vaultMBPerUser": round(vaultBytes / max(1, len(vault)) / 1e6, 3),
                      "indexFileGB": round((home / "index.sqlite").stat().st_size / 1e9, 2)}

    text = json.dumps(rep, indent=1)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
