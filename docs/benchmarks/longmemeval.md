# LongMemEval results

Kormem scores **94.4% (472/500) on [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)** [7] with its current architecture, Claude Sonnet 5.5 as the answer model and the official GPT-4o judge (GPT-6.1 Sol 93.8%, GPT-6 Astra 93.6%). Retrieval puts every gold session in the answer context for 99.0% of questions.

Dataset: `longmemeval_s_cleaned.json`, all 500 questions, including the 30 abstention questions.

## Current architecture (2026-10-15)

Every session is ingested through the product path: chunks with `gemini-embedding-2` vectors, plus a linked vault (topic notes with dated timelines, memory lines and source notes) written by `gpt-6-luna`. Recall ranks sessions by reciprocal rank fusion of a local cross-encoder with memory-line, chunk-search and note-link signals; the recursive recall agent (`gpt-6-luna`) then walks notes and links. The answer model gets the selected full transcripts (mean 8.0 sessions, ~32k tokens) with the agent's findings, notes and memory lines.

| | Result |
|---|---|
| Official judge | **93.8%** (469/500)¹ |
| Full run before the prompt fix | 92.2% (461/500), task-averaged 93.2% |
| By type (full run) | single-session assistant 56/56, user 68/70, knowledge update 74/78, temporal 123/133, preference 27/30, multi-session 113/133; abstention 26/30 |
| any@5 / all@5 / all@10 | 99.8% / 95.8% / 99.2% |
| All gold sessions in context | 99.0% |
| Latency per question (recall + answer) | p50 35 s, p90 64 s (recall alone p50 31 s) |
| Build cost (500 users, ~24k sessions) | $40.37: extraction $23.58, embeddings $16.78 |
| Recall cost | $0.39 recall agent, $0.08 query embeddings |
| Storage | 0.53 MB vault per user |

¹ The full run scored 461/500. In 38 of its 39 misses every gold session was in context; 12 were the answer model saying "I don't know" about a detail it could have derived (for example, subtracting two stated durations). Rule 11 of the answer prompt now asks it to derive from stated facts before abstaining. Re-answering the 39 misses from the same cached contexts fixed 8 (469/500). The 461 correct answers were not re-answered with the revised prompt; a control sample of 21 of them (13 abstention questions) stayed correct.

### By answer model

Same cached contexts, same judge. Sonnet and Astra answered all 500 with the revised answer prompt in one run each; Sol's figure combines its full run with a re-answer of its misses (¹).

| Answer model | Official judge | Task-averaged | Abstention | Multi-session | Temporal | Knowledge update | Single-session user / assistant / preference |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | **94.4%** (472/500) | **95.6%** | 25/30 | **120**/133 | 124/133 | **75**/78 | 68/70, 56/56, **29**/30 |
| GPT-6.1 Sol | 93.8% (469/500)¹ | 93.2%² | **26**/30 | 113/133² | 123/133² | 74/78² | 68/70, 56/56, 27/30² |
| GPT-6 Astra | 93.6% (468/500) | 93.9% | **26**/30 | 117/133 | **125**/133 | **75**/78 | **69**/70, 56/56, 26/30 |

² Full run before the prompt fix (461/500).

- **The three are a statistical tie** (95% CI about ±2.1 pts at n=500). Together they answer 481/500; Sonnet alone gets 9 that neither GPT model does.
- **Sonnet's lead is in multi-session and preference questions** (120/133 and 29/30). It gives up one abstention.
- **Answer cost and speed.** Sonnet: $53.63 for 500 (26.1M input tokens; its tokenizer counts the same context ~1.6x higher), answer p50 5.4 s. Astra: 16.1M input and 0.07M output tokens, p50 5.9 s. Sol: about $33.

- **The remaining misses are answer reasoning.** Only one of the 39 lacked a gold session (`gpt4_7abb270c`, which needs six sessions; one ranked 14th). The rest are multi-session counting, date arithmetic and over-caution, plus a few likely judge errors (`gpt4_d12ceb0e` answered 59.6, the gold answer, and was marked wrong).
- **Multi-session is within run-to-run noise of the previous architecture** (113 vs 117; two Sol runs on the previous architecture scored 117 and 121). With the revised prompt, 4 of its 7 regressions answer correctly.

## Previous architecture (2026-10-12): results by answer model

Each question got a single-shot answer from a curated context of about 19k tokens, roughly 1/6 of the conversation history.

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
| **Kormem** | **Claude Sonnet 5.5** | **GPT-4o-2024-08-06, official `evaluate_qa.py`** | **~32k tokens (8 full sessions)** | **94.4%** |
| **Kormem** | **GPT-6.1 Sol** | **same** | **same** | **93.8%** |
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
- **Kormem is not a low-token system.** It uses about 32k tokens of context (19k on the previous architecture), against roughly 7k for Mem0 and 1.6k for Zep.

## Retrieval recall

R@5 asks whether a gold evidence session is among the top 5 retrieved sessions. It measures retrieval only, with no answer and no judge, so it can't be compared with the accuracy table above. [MemPalace](https://github.com/mempalace/mempalace) [6] publishes only this metric for LongMemEval, so we computed Kormem's from the cached contexts of the 2026-10-15 accuracy run.

Session level, all 500 questions. `any@5` is LongMemEval's `recall_any` (at least one gold session in the top 5), which is the published R@5. `all@5` (every gold session in the top 5) is stricter and matters more for multi-session questions.

| System | Ranking | LLM in ranking | any@5 (R@5) | all@5 |
|---|---|---|---|---|
| [MemPalace](https://github.com/mempalace/mempalace) raw [6] | semantic search | no | 96.6% | 85.0% |
| [MemPalace](https://github.com/mempalace/mempalace) hybrid v4, held-out 450 [6] | keyword, time and preference boosts | no | 98.4% | 88.7% |
| [MemPalace](https://github.com/mempalace/mempalace) hybrid v4 + LLM rerank [6] | top-20 reranked by an LLM | yes | 99.2%¹ | 89.8% |
| **Kormem** | **cross-encoder + RRF, then `gpt-6-luna` recall agent** | **yes** | **99.8%** | **95.8%** |
| Kormem, previous architecture | session scoring (`gpt-6-luna`, Decisions API) | yes | 97.8% | 76.0% |

¹ MemPalace headlines "≥99%" for this mode and says part of the last gap was tuned on specific wrong answers.

- **MemPalace numbers are recomputed from the per-question rankings it commits** (`benchmarks/results_mempal_*_session_*.jsonl`, MemPalace commit `d9bf01e`). Our any@5 matches their published figures exactly; all@5 is our computation from the same files.
- **Kormem now leads on both measures.** all@5 is 6 pts above MemPalace's best mode (95.8% vs 89.8%); the previous architecture trailed it by 14.
- **Kormem doesn't send the top 5 to the answer model.** It sends about 8 sessions, which hold every gold session for 99.0% of questions.
- **Sessions are matched exactly** by the `chat <session id>` source title. (The previous architecture matched by timestamp; resolving its ties in Kormem's favour gave 98.4% / 77.6%.)

Reproduce: `python -m benchmark.retrieval_recall contexts/` on the release asset's `contexts/` folder.

## Where the errors are (previous architecture)

For the current architecture, see the notes under its [results](#current-architecture-2026-10-15). On the previous architecture, Sonnet, Sol and Gemini each miss 28–32 questions. Measured on the shared contexts:

- **Retrieval puts every gold session in context for 90.0% of answerable questions** (423/470); 93.4% of all gold sessions are retrieved. gbrain's release retrieval reaches 95.5% (449/470) on the same measure, so Kormem is **behind on retrieval**.
- **Most misses are reasoning, not retrieval.** In the Sol run, 21 of 30 misses had every gold session in context. The failures are over- or under-counting, date arithmetic and wrong time windows. Only 5 misses lacked some or all of the gold evidence, and 4 were unanswerable questions the model answered anyway.
- **The four models together answer 489/500 (97.8%)** if you take any model's correct answer. Only 11 questions defeat all four.

## Limitations

1. **Recall is slow.** The recall agent makes several `gpt-6-luna` calls per question (median recall 31 s). It is cheap ($0.39 for 500 questions) but slower than embedding-only systems. The previous architecture used an OpenAI Decisions call and a rerank per question.
2. **Single run per configuration.** We report no confidence intervals beyond the binomial estimate above. The current 469/500 combines a full run with a re-answer of its misses (see the footnote under its results).
3. **Ingestion provenance.** Chunks for 393 questions came from an earlier ingestion by the same chunker and were re-embedded with the current model. The other 107 were ingested fresh. Memory extraction ran fresh for all 500.

## Setup (previous architecture)

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
