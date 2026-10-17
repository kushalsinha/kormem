# Jev in kormem: the research behind the design

This records the tests we ran in October 2026 to decide where [Jev](https://typesafe.ai) (TypeSafe's
decision model) belongs in kormem, what each test showed, and why the architecture ended up as it is.
The README describes the design; this page explains how we chose it.

## The question

Jev answers a typed question about a small `state` with a calibrated probability. It returns a yes/no
score, a label or a pick from up to 255 options, but it never writes text. A call takes about 0.1 s
and costs $0.042 per million input tokens; output is free. We used `jev-1.13.0`, pinned because the
thresholds below are tuned to that version.

Kormem's model work at the time was all luna (`gpt-6-luna`):

| Job | What it writes | Cost |
|---|---|---|
| Extraction (`vault_linker`) | Vault notes, memory lines, timelines | ~$4 per 60 users; most of a ~$40 build of all 500 |
| Recall agent (`recursive_agent`) | Search queries, tool calls, dated findings | ~$0.0012 per question, ~30 s |
| Query rewrite, HyDE, decomposition (`query_optimizer`) | New queries | Very small, but 5.5 s |

Jev can't replace any of these, because each one writes text. It can make the decisions around them:
whether luna runs, where luna's output goes, and how much context reaches the assistant.

## Method

- **Data.** Only LongMemEval-S (`longmemeval_s_cleaned.json`). No real user data was sent to TypeSafe.
- **Sets.** Ideas were screened first on the 60-question check set, spread across all six question types,
  using cached contexts and stored indexes. Anything that looked worth shipping was then measured on all
  500 questions.
- **Answers and judging.** luna answered and the official GPT-4o judge scored. No costly answer models
  (Astra, Sol, Sonnet) were used in this round.
- **Retrieval metric.** "All gold @k" means every session the answer needs is in the top k.
- **Noise.** The same 60 contexts answered twice by luna scored 53 and then 50, so differences of about
  ±3 on 60 questions are noise.
- **Cost.** The whole research round cost a few dollars: about $1.85 for the 60-question round
  ($1.21 Jev), $0.32 for the first gate test, and about $0.44 per 500-question Jev pass.

## Round 1: every candidate decision, 60 questions

### Pruning the answer context

For each source passed to the answer model (about 8 per question), we asked Jev whether it helps answer
the question.

| Drop below | Gold sources kept | Questions with all gold | Answer tokens cut |
|---|---|---|---|
| 0.05 | 121/121 | 60/60 | 5% |
| 0.1 | 120/121 | 59/60 | 33% |
| 0.2 | 118/121 | 57/60 | 56% |
| 0.3 | 114/121 | 53/60 | 65% |

The one gold source lost at 0.1 belonged to an abstention question. In answer runs, pruning at 0.1 cut
the context by 22% (142k to 110k characters) and scored 50 and 52, against 53 and 50 unpruned: no
measurable loss.

### Ranking, and answering without the agent

| Context given to luna | Correct / 60 | Context size |
|---|---|---|
| First pass plus recall agent (run twice) | 53, then 50 | 142k chars |
| Same, pruned below 0.1 (run twice) | 50, then 52 | 110k chars |
| **Jev's top 8 sessions, no agent** | **54** | 140k chars |

Jev ranking put all gold sessions in the top 5 for 59/60 questions, against 57/60 for the first pass.
Skipping the agent removed its ~30 s and left under a second of Jev calls. This test scored every
session (about 129k tokens and $0.005 per question), and that cost grows with a user's history. That's
why the 500-question round scored only the first pass's top 30.

### The other decisions

| Decision | Result | Outcome |
|---|---|---|
| Question routing ("current value", "time", "other") | 46/60 correct, against 35/60 for our regex | Not adopted yet; could route hard questions to `--deep` |
| Skipping the agent for simple lookups | Skips 10, 1 wrongly; the cross-encoder alone would skip 24, 15 wrongly | Moot once the default has no agent |
| PARA placement of new notes | Agrees with luna on 83% of 4,045 notes; 96% when Jev's confidence ≥ 0.8 | Not adopted |
| Merge check for similar notes | Only one pair in 60 users passes the 0.92 cut-off, a bad merge (charity basketball + charity run) that Jev blocks; below the cut-off Jev is too noisy to decide | Not adopted |

When Jev disagreed with luna on PARA with high confidence, it was often right (Fishing and Cognitive
health under Areas). It was wrong on places such as Seoul, which belong in Resources, and it can't
tell when a project is finished.

### The "worth remembering" gate before extraction

We asked Jev, for each of 2,780 sessions, whether it was worth extracting. Sessions that failed weren't
sent to luna, but stayed indexed and searchable.

| Wording | Threshold | Gold kept | Sessions skipped | Tokens skipped |
|---|---|---|---|---|
| A personal fact or request | 0.2 | 121/123 | 20% | 12% |
| A personal fact or request | 0.3 | 121/123 | 24% | 15% |
| A personal fact or request | 0.5 | 120/123 | 29% | 18% |
| + decision, correction, or content the user asked for | 0.5 | 123/123 | 12% | 7% |
| + decision, correction, or content the user asked for | 0.7 | 122/123 | 20% | — |

The first wording lost the "what did you tell me earlier" sessions, such as a list of 100 prompt
parameters, because those chats contain nothing personal. Luna extraction costs about $0.0014 per
session and the Jev check about $0.0001, so the check costs roughly 7% of extraction. The net saving is
about 7% of an already small cost.

LongMemEval's filler is generic chat, which makes a gate look better than it would in real use. Most
real kormem input is coding sessions: project decisions, conventions and fixes, with little personal
content. Those are what users most want remembered, so this is where a gate would do the most harm,
and the benchmark can't measure that wording. **Not adopted.**

## Round 2: recall, all 500 questions

### Ranking the first pass's top 30

Jev scored only the top 30 candidates, using a preview of up to 1,500 characters of each.

| Ranking of the top 30 | All gold @5 | All gold @8 | Jev cost / question |
|---|---|---|---|
| First pass alone | 95.6% | 98.8% | – |
| Jev on previews alone | 95.2% | 97.8% | $0.0009 |
| **First pass + Jev on previews (RRF)** | **97.8%** | **99.0%** | $0.0009 |
| First pass + Jev on full sessions | 97.8% | 99.0% | $0.0036 |

Neither ranking is best on its own: Jev's score is calibrated but sees only a preview, while the first
pass sees the whole chunk index. Fusing their ranks with RRF (k = 60) beats both. Previews cost a
quarter of full sessions and lose nothing. Sources the answer needs scored a median of 0.9, the rest
0.06, and 17 of 948 needed sources scored below 0.1.

The question Jev answers is phrased so partial evidence counts:

> Does `source` contain information that helps answer `question`, even partly: one item of a count,
> one event's date, an earlier or later value, a stated preference, or something the assistant said?

### Answering without the agent

| Context given to luna | Correct | Task-averaged | Recall latency |
|---|---|---|---|
| First pass + agent | 456 / 500 | 92.1% | ~30 s |
| First pass + Jev, top 8, no agent | 449 / 500 | 89.8% | a few seconds |

Most of the 7-answer gap is preference questions (26 vs 23 correct), where the agent writes down the
user's tastes as findings. Per question, the version without the agent fixed 13 and broke 20. This is
why the agent stays available as deep recall rather than being removed.

### The luna query rewrite

| First pass | All gold @8 | Latency p50 |
|---|---|---|
| With the luna rewrite / HyDE / decomposition | 98.8% | 7.1 s |
| Without it | 98.6% | 2.0 s |

The rewrite took 5.5 s of the 7.1 s first pass and made no real difference to recall. **Turned off**
(`queryRewrite: false`). Every 0.1.x `config.yaml` stored `queryOptimize: true`, so the setting was
renamed in order to reach existing installs.

### Trimming after ranking

Using the stored scores, with no answer run:

| Rule | Sources / question | All gold in context |
|---|---|---|
| Top 8, no trim | 8.0 | 495 / 500 |
| **Keep top 5; drop ranks 6–8 with Jev < 0.3** | **5.4** | **492 / 500** |
| Hard cut at top 5 | 5.0 | 489 / 500 |
| Drop any rank with Jev < 0.1 (keep 2) | 5.1 | 486 / 500 |
| Drop any rank with Jev < 0.05 (keep 2) | 7.3 | 495 / 500 |

A probability floor alone either trims too little or drops needed partial evidence from the top ranks.
Protecting the top 5 and trimming only weak tail sources cuts about a third of sources for 3 questions'
evidence.

### Telling the answer model the ranks

About 10% of needed sources score below 0.4, mostly partial pieces of a count or timeline, so low
relevance isn't noise. Sources therefore stay in date order (which time questions need). Each one is
labelled with its rank and relevance, a ranked index is added, and one line says that lower-ranked
sources can still hold one item of a count, an earlier value or a date.

### Answer check of the shipped default

Luna answered all 500 questions from the shipped default (first pass, Jev, top 8 trimmed), through the
product code, once without rank labels and once with them.

| Context | Correct | Task-averaged | Sources / question | Context size | All gold in context |
|---|---|---|---|---|---|
| Agent (deep recall) | 456 / 500 | 92.1% | – | – | – |
| Top 8, no trim, no labels | 449 / 500 | 89.8% | 7.95 | 152k chars | 495 / 500 |
| Default, trimmed, no labels | 447 / 500 | 90.2% | 5.33 | 111k chars | 487 / 500 |
| **Default, trimmed, with rank and relevance labels** | **448 / 500** | **91.6%** | 5.33 | 111k chars | 487 / 500 |

- **Trimming** cut the context by 27% for no measurable loss: 447 against 449, with 14 questions fixed
  and 16 broken, which is within luna's run-to-run swing. It lost the full evidence on 8 more questions
  than the untrimmed top 8 (5 more than the stored-score estimate), but only 3 of the 13 questions
  without full evidence were answered wrong.
- **Labels** left the raw count flat (448 vs 447; 12 fixed, 11 broken) and raised the task-averaged
  score 1.4 points, mostly from preference questions (25 to 28). Temporal questions went from 122 to 119,
  so labels are not a clear win per type, but they cost nothing and point the agent at the right sources.
- The default stays 7–8 answers behind deep recall, as expected; the gap is why `--deep` exists.

## What we decided

| Decision | Why |
|---|---|
| Jev re-ranks the first pass's top 30 previews by default, fused by RRF | +2.2 points all gold @5 for $0.0009 a question and ~0.3 s |
| Default recall runs no agent; `--deep` / `deep: true` runs it | 449 vs 456 correct, but seconds instead of ~30 s and no LLM call at query time |
| Trim ranks 6–8 below Jev 0.3 | A third fewer sources and 27% less context; 447 vs 449 correct |
| Label sources with rank and relevance | Directs the answer model's attention without hiding partial evidence |
| Luna query rewrite off | 5.5 s for no recall gain |
| Local cross-encoder kept | A free, local signal in the first pass that Jev's fusion builds on, and the ranking when there is no key |
| Jev is optional | Without `TYPESAFE_API_KEY`, or when Jev errors or takes more than 3 s, recall ranks locally and nothing leaves the machine |
| Extraction stays luna only | The gate, PARA and merge gains were small, and the gate's risk falls on coding sessions |

## Open ideas

- Use Jev routing (46/60) to send counts, timelines and preference questions to deep recall automatically.
- A "worth remembering" gate tuned on real coding sessions, which needs data LongMemEval doesn't have.

## Reproducing

The per-question rows, cached contexts and logs are kept outside the repo, in the local benchmark
archive (`kormem-benchmark-archive/2026-10-15/jev/`): `pool_rows.json`, `prevpool_rows.json`,
`prune_rows.json`, `gate_rows.json`, `para_rows.json`, `merge_rows.json`, `route_rows.json` and
`skip_rows.json`, plus each answer run and its official-judge output. Answer runs use
`benchmark.run_subset` with `KORMEM_RECALL_EFFORT`, `KORMEM_ANSWER_RANKS` and `KORMEM_CONTEXT_CACHE`; see
[benchmark/README.md](../../packages/engine/benchmark/README.md).
