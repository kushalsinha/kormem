# LongMemEval results

Kormem scores **94.4% (472/500) on [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)** [7]. That uses Claude Sonnet 5.5 as the answer model and the official GPT-4o judge. Each question gets a single-shot answer from a curated context of about 19k tokens, roughly 1/6 of the conversation history.

Run date: 2026-10-12. Dataset: `longmemeval_s_cleaned.json`, all 500 questions, including the 30 abstention questions.

## Results by answer model

The memory, retrieval and prompt are identical in every row; only the answer model changes. Gemini, Grok and Sonnet answered from byte-identical cached contexts. Sol used an earlier retrieval pass of the same pipeline.

| Answer model | Official judge | Task-averaged | Abstention | Multi-session | Temporal | Knowledge update | Preference |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | **94.4%** | 94.6% | 28/30 | **120**/133 | 126/133 | **77**/78 | 27/30 |
| GPT-6.1 Sol | 94.0% | **94.8%** | 26/30 | 117/133 | 126/133 | 76/78 | **28**/30 |
| Gemini 3.8 Flash | 93.6% | 93.6% | 27/30 | 118/133 | 126/133 | 76/78 | 26/30 |
| Grok 4.7 | 88.0% | 89.8% | 28/30 | 98/133 | 121/133 | 75/78 | 26/30 |
| *Sol, full history in context (no curation)* | *93.4%* | *92.9%* | *27/30* | | | | |

- **The top three are a statistical tie.** The 95% binomial CI at n=500 is about ±2.1 pts. Re-running the official judge on the same answers also moves a model's score by 1–2 questions.
- **Grok is behind on counting.** It loses 19–22 questions on multi-session sums and counts. It also tends to answer "you didn't mention X" when X is in its context.
- **Curated context matches full history** (94.0% vs 93.4% with the same answer model) at about 14% of the input.

## Comparison with published systems

All rows use LongMemEval-S, 500 questions and a single-shot answer. We exclude agentic multi-pass answering. Retrieval-only metrics (R@k) measure something else, so they have [their own table](#retrieval-recall).

| System | Answer model | Judge | Context per question | Accuracy |
|---|---|---|---|---|
| [Mastra Observational Memory](https://mastra.ai/research/observational-memory) [1] | gpt-5-mini | GPT-4o, LongMemEval prompts | full compressed observation log | 94.9% |
| **Kormem** | **Claude Sonnet 5.5** | **GPT-4o-2024-08-06, official `evaluate_qa.py`** | **~19k tokens** | **94.4%** |
| **Kormem** | **GPT-6.1 Sol** | **same** | **same** | **94.0%** |
| [Mem0 platform v3](https://github.com/mem0ai/memory-benchmarks) [2] | GPT-5 | custom lenient judge (not the official prompts) | top-200 memories, ~7k tokens | 93.4–94.4%¹ |
| [gbrain](https://github.com/garrytan/gbrain) [3] | Claude Sonnet 4.6 | GPT-4o-2024-08-06, official `evaluate_qa.py` | top-5 full sessions | 86.6% |
| [Supermemory](https://supermemory.ai/research/longmembench) [4] | Gemini 3 Pro / GPT-5 | GPT-4o, LongMemEval prompts | top-20 | 85.2% / 84.6%² |
| [Mastra Observational Memory](https://mastra.ai/research/observational-memory) [1] | GPT-4o | GPT-4o, LongMemEval prompts | full compressed observation log | 84.2% |
| [Zep](https://arxiv.org/abs/2501.13956) [5] | GPT-4o | GPT-4o, LongMemEval prompts | ~1.6k tokens | 71.2% |
| Full-context baseline [5] | GPT-4o | GPT-4o, LongMemEval prompts | ~115k tokens | 60.2% |

¹ Mem0's README reports 94.4% (472/500). The committed `results/platform/longmemeval_results.json` says 93.4% (467/500). Their judge is a single unified prompt that explicitly accepts partial abstentions and loose preference matches, so it is more lenient than the official one.
² Supermemory also reports 97% for GPT-4o. Those per-type figures are the same as their Recall@20 retrieval table, so we list only the GPT-5 and Gemini rows. Their ~99% "ASMR" result is agentic multi-pass and excluded.

**What this comparison can and can't tell you:**
- **These are mostly comparisons of answer models.** On identical context, swapping the answer model alone moves Kormem by 6.4 pts (Grok to Sonnet). Older rows with GPT-4o answers can't be separated from the system's own contribution.
- **gbrain is the closest apples-to-apples row:** same judge and prompts, full-session context and a recent Claude model. Kormem leads by 7.8 pts, but a newer answer model (Sonnet 5.5 vs 4.6) accounts for an unknown share of that.
- **Kormem is not a low-token system.** It uses about 19k tokens of context, against roughly 7k for Mem0 and 1.6k for Zep.

## Retrieval recall

R@5 asks whether a gold evidence session is among the top 5 retrieved sessions. It measures retrieval only, with no answer and no judge, so it can't be compared with the accuracy table above. [MemPalace](https://github.com/mempalace/mempalace) [6] publishes only this metric for LongMemEval, so we computed Kormem's from the same cached contexts as the accuracy runs.

Session level, all 500 questions. `any@5` is LongMemEval's `recall_any` (at least one gold session in the top 5), which is the published R@5. `all@5` (every gold session in the top 5) is stricter and matters more for multi-session questions.

| System | Ranking | LLM in ranking | any@5 (R@5) | all@5 |
|---|---|---|---|---|
| [MemPalace](https://github.com/mempalace/mempalace) raw [6] | semantic search | no | 96.6% | 85.0% |
| [MemPalace](https://github.com/mempalace/mempalace) hybrid v4, held-out 450 [6] | keyword, time and preference boosts | no | 98.4% | 88.7% |
| [MemPalace](https://github.com/mempalace/mempalace) hybrid v4 + LLM rerank [6] | top-20 reranked by an LLM | yes | 99.2%¹ | 89.8% |
| **Kormem** | **session scoring (`gpt-6-luna`, Decisions API)** | **yes** | **97.8%** | **76.0%** |

¹ MemPalace headlines "≥99%" for this mode and says part of the last gap was tuned on specific wrong answers.

- **MemPalace numbers are recomputed from the per-question rankings it commits** (`benchmarks/results_mempal_*_session_*.jsonl`, MemPalace commit `d9bf01e`). Our any@5 matches their published figures exactly; all@5 is our computation from the same files.
- **Kormem ranks first with a hit, but MemPalace ranks more complete.** Kormem's any@5 is between MemPalace raw and hybrid. On all@5 it trails all three MemPalace modes by 9–14 pts.
- **Kormem doesn't send the top 5 to the answer model.** It sends a reranked, variable set (mean 4.7 sessions) that holds every gold session for 90.0% of answerable questions (see [Where the errors are](#where-the-errors-are)). Raising all@5 is the clearest retrieval improvement available.
- **Bounds.** Kormem candidates are matched to sessions by timestamp, and 158 of 15,000 candidate slots share one with another session. Resolving every tie against the gold session gives the figures above; in its favour gives 98.4% / 77.6%.

Reproduce: `python -m benchmark.retrieval_recall contexts/` on the release asset's `contexts/` folder.

## Where the errors are

Sonnet, Sol and Gemini each miss 28–32 questions. Measured on the shared contexts:

- **Retrieval puts every gold session in context for 90.0% of answerable questions** (423/470); 93.4% of all gold sessions are retrieved. gbrain's release retrieval reaches 95.5% (449/470) on the same measure, so Kormem is **behind on retrieval**.
- **Most misses are reasoning, not retrieval.** In the Sol run, 21 of 30 misses had every gold session in context. The failures are over- or under-counting, date arithmetic and wrong time windows. Only 5 misses lacked some or all of the gold evidence, and 4 were unanswerable questions the model answered anyway.
- **The four models together answer 489/500 (97.8%)** if you take any model's correct answer. Only 11 questions defeat all four.

## Limitations

1. **Retrieval is LLM-heavy.** Each question makes an OpenAI Decisions call to score sessions and a gpt-6-luna rerank. Latency and cost are higher than embedding-only systems.
2. **Single run per configuration.** We report no confidence intervals beyond the binomial estimate above.
3. **Ingestion provenance.** Chunks for 393 questions came from an earlier ingestion by the same chunker and were re-embedded with the current model. The other 107 were ingested fresh. Memory extraction ran fresh for all 500.

## Setup

| Stage | Configuration |
|---|---|
| Ingestion | One namespace per question; chunked sessions in SQLite with FTS5 |
| Embeddings | `gemini-embedding-2`, 768-d |
| Memory extraction | `gpt-6-luna`, reasoning medium |
| Session scoring | OpenAI Decisions API (`gpt-6-luna`), top-p 0.9 nucleus |
| Rerank | `gpt-6-luna` picks from the top 30. Sessions with P ≥ 0.8 are always kept (max 6). Counting questions keep the whole nucleus. |
| Context sent to the answer model | Relevant memory lines with absolute dates, plus full transcripts of the selected sessions (mean 4.7) in chronological order |
| Answer | Single call: reasoning notes, then an `ANSWER:` line, which is the only part judged. Reasoning effort medium; Claude uses a 6k-token extended-thinking budget. |
| Judge | `gpt-4o-2024-08-06`, temperature 0, verbatim `evaluate_qa.py` prompts, `_abs` questions judged with the abstention prompt |

## Reproduce

Per-question answers, official verdicts and the cached contexts every answer model received are published as the release asset `longmemeval-results-2026-10-12.tar.gz`.

Run from `packages/engine`, with API keys in the repo-root `.env`. Download [`longmemeval_s_cleaned.json`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) first.

```bash
export LONGMEMEVAL_DATASET=/path/to/longmemeval_s_cleaned.json

# 1. Ingest (chunk + embed) and extract memory/persona for every question
python -m benchmark.ingest
python -m benchmark.extract_subset --subset $LONGMEMEVAL_DATASET

# 2. Start six engine servers (each handles one request at a time)
for p in 8787 8788 8789 8790 8791 8792; do PORT=$p benchmark/serve.sh & done

# 3. Answer, then re-score with the official protocol
KORMEM_CONTEXT_MODE=curated KORMEM_CONTEXT_CACHE=contexts \
KORMEM_ANSWER_MODEL=claude-sonnet-5-5 \
KORMEM_API_URLS=$(printf 'http://127.0.0.1:%s,' 8787 8788 8789 8790 8791 8792 | sed 's/,$//') \
python -m benchmark.run_subset --subset $LONGMEMEVAL_DATASET --concurrency 8 \
  --results-jsonl out.jsonl --resume --output out.json
python -m benchmark.official_judge out.json
```

- The published numbers were measured without the recursive recall agent. Set `KORMEM_DEEP_RECALL=0` to reproduce them; the default (`1`) also runs the agent (`deepRecall`), which has not yet been measured on all 500 questions.
- With `KORMEM_CONTEXT_CACHE` set, the first run saves each question's retrieved context to that directory and later runs reuse it. That is how the answer-model rows above share identical input.
- To skip steps 1–2, point `KORMEM_CONTEXT_CACHE` at the `contexts/` folder from the release asset; only the answer model and judge then run.
- Model names starting `gpt-`, `gemini-`, `grok-` or `claude-` are sent to OpenAI, Google, xAI or Anthropic respectively.

## Sources

1. Mastra, *Observational Memory: 95% on LongMemEval* — https://mastra.ai/research/observational-memory
2. Mem0, `memory-benchmarks` — https://github.com/mem0ai/memory-benchmarks
3. gbrain `CHANGELOG.md`, LongMemEval answer-accuracy entry — https://github.com/garrytan/gbrain
4. Supermemory, *LongMemEval eval report* — https://supermemory.ai/research/longmembench
5. Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* (2025) — https://arxiv.org/abs/2501.13956
6. MemPalace, `benchmarks/BENCHMARKS.md` and committed per-question results — https://github.com/mempalace/mempalace/blob/main/benchmarks/BENCHMARKS.md
7. Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* (ICLR 2025) — https://arxiv.org/abs/2410.10813
