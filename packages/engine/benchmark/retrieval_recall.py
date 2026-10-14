"""Session-level retrieval recall (R@k) from cached LongMemEval contexts.

Reads the session ranking each cached context stores (``debug.candidates``,
the top 30 by session score) and reports, over all 500 questions:

- ``any@k``: at least one gold session in the top k, LongMemEval's
  ``recall_any``, which is what MemPalace and others publish as R@5;
- ``all@k``: every gold session in the top k (``recall_all``).

Candidates are identified by timestamp. When two sessions share one, the
ranking is ambiguous, so both bounds are reported: ``low`` resolves every tie
against the gold session, ``high`` in its favour. Quote ``low``.

Usage:
    python -m benchmark.retrieval_recall path/to/contexts [--subset longmemeval_s_cleaned.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmark.common import datasetPath

KS = (1, 3, 5, 10)


def _parse(value):
    if not isinstance(value, str):
        return value
    return eval(value, {"__builtins__": {}}, {"nan": float("nan"), "None": None, "True": True, "False": False})


def ranking(item: dict, candidates: list, favourGold: bool) -> list:
    gold = set(item["answer_session_ids"])
    byDate = defaultdict(list)
    for sid, date in zip(item["haystack_session_ids"], item["haystack_dates"]):
        byDate[date].append(sid)
    used, ranked = defaultdict(set), []
    for c in candidates:
        pool = [s for s in byDate[c["header"]] if s not in used[c["header"]]]
        pool.sort(key=lambda s: (s not in gold) if favourGold else (s in gold))
        used[c["header"]].add(pool[0])
        ranked.append(pool[0])
    return ranked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("contexts")
    ap.add_argument("--subset", default=None)
    args = ap.parse_args()

    with open(datasetPath(args.subset), encoding="utf-8") as f:
        items = {it["question_id"].lower(): it for it in json.load(f)}
    hits = defaultdict(int)
    files = sorted(Path(args.contexts).glob("*.json"))
    for path in files:
        item = items[path.stem.lower()]
        gold = set(item["answer_session_ids"])
        candidates = _parse(json.loads(path.read_text())["debug"]["candidates"])
        for bound, favour in (("low", False), ("high", True)):
            ranked = ranking(item, candidates, favour)
            for k in KS:
                top = set(ranked[:k])
                hits[bound, "any", k] += bool(gold & top)
                hits[bound, "all", k] += gold <= top
    n = len(files)
    print(f"{n} questions")
    for k in KS:
        print(f"@{k:<2}  any {100 * hits['low', 'any', k] / n:5.1f}% (high {100 * hits['high', 'any', k] / n:5.1f}%)"
              f"   all {100 * hits['low', 'all', k] / n:5.1f}% (high {100 * hits['high', 'all', k] / n:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
