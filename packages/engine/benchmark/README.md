# LongMemEval benchmark harness

Results and methodology: [docs/benchmarks/longmemeval.md](../../../docs/benchmarks/longmemeval.md).

The harness uses the product path end to end: each question's haystack sessions go in as chat sessions, extraction builds the vault, and `RecallPipeline` (the code behind `/recall`) produces the context. Nothing is benchmark-specific except the answer prompt and the judge.

| Script | Purpose |
|---|---|
| `ingest.py` | Build each question's namespace (`lme_<qid>`): sessions → chat sessions → extraction (vault linker + persona). |
| `run_subset.py` | Recall, answer and judge each question in-process. |
| `official_judge.py` | Re-score a results file with the official LongMemEval protocol: `gpt-4o-2024-08-06` and the verbatim `evaluate_qa.py` prompts. |
| `retrieval_recall.py` | Session-level R@k (`recall_any` and `recall_all`) from a folder of cached contexts. |
| `judge.py`, `config.py`, `common.py` | Kormem's own judge, model settings, shared helpers. |

```bash
export LONGMEMEVAL_DATASET=/path/to/longmemeval_s_cleaned.json
python -m benchmark.ingest --subset questions.json --concurrency 12
KORMEM_CONTEXT_CACHE=contexts python -m benchmark.run_subset --subset questions.json \
  --concurrency 8 --results-jsonl out.jsonl --resume --output out.json
python -m benchmark.official_judge out.json
python -m benchmark.retrieval_recall contexts
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `KORMEM_ANSWER_MODEL` | `gpt-6.1-sol` | Answer model. Names starting `gpt-`, `gemini-`, `grok-` or `claude-` go to that provider's API. |
| `KORMEM_ANSWER_REASONING_EFFORT` | `medium` | Reasoning effort; for Claude it maps to an extended-thinking budget. |
| `KORMEM_RECALL_EFFORT` | `high` | `high` runs the recall agent (the published runs); `low` is the default for the CLI and MCP: Jev-ranked sources, no agent. |
| `KORMEM_ANSWER_RANKS` | `1` | Show each session's rank and Jev relevance in the answer prompt (contexts saved before 0.1.3 have none). `0` turns it off. |
| `KORMEM_EXTRACTION_REASONING_EFFORT` | `low` | Reasoning effort for extraction calls. |
| `KORMEM_CONTEXT_CACHE` | — | Save and reuse each question's recall output, so answer models see identical input. |

API keys are read from the repo-root `.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `TYPESAFE_API_KEY` (Jev, for recall), `XAI_API_KEY`, and `ANTHROPIC_API_KEY` (plus `ANTHROPIC_WORKSPACE_ID` if your key isn't scoped to a workspace).
