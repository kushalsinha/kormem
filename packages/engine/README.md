# kormem

Local long-term memory for AI agents. kormem indexes everything you say to assistants and every file, folder or URL you add, files it into an Obsidian-compatible vault of linked notes, and returns the ranked notes and sources a question needs in a few seconds. `--deep` adds a small recall agent that returns dated findings.

The commands follow [QMD](https://github.com/tobi/qmd): `collection add`, `context add`, `search`, `vsearch`, `query`, `get`, `multi-get` and `ls` work the same way.

<p align="center">
  <a href="https://youtu.be/5kTirMTDepk">
    <img src="https://raw.githubusercontent.com/kushalsinha/kormem/main/docs/media/kormem-launch-poster.jpg" alt="Watch the kormem launch video (68 s)" width="720">
  </a>
  <br><sub>▶ <a href="https://youtu.be/5kTirMTDepk">Watch the launch video</a> (68 s)</sub>
</p>

## Add kormem to your agent

Paste this into Claude Code, Codex, Cursor or any agent that can run commands:

```text
Install kormem, long-term memory for this agent.

1. Run `npm install -g kormem` (or `pip install kormem`), then `kormem init`.
2. Add the kormem skill. Use one installation method:
   - In Claude Code, `kormem init` installs the kormem plugin. If it says it didn't, run
     `claude plugin marketplace add kushalsinha/kormem`, then
     `claude plugin install kormem@kormem`.
   - In another agent, run `npx skills add kushalsinha/kormem --skill kormem` and select
     your agent.
3. Ask me to add two keys to ~/.kormem/.env. Don't ask me to paste keys into the chat.
   - OPENAI_API_KEY (https://platform.openai.com/api-keys): files my chats and documents
     into memory and runs deep recall.
   - TYPESAFE_API_KEY (https://typesafe.ai), recommended: Jev ranks what recall returns
     in about 0.3 s for a tenth of a cent, with no LLM call.
   Then run `kormem doctor` to check both are set.
4. Read the skill:
   https://github.com/kushalsinha/kormem/blob/main/plugin/skills/kormem/SKILL.md
   (raw: https://raw.githubusercontent.com/kushalsinha/kormem/main/plugin/skills/kormem/SKILL.md)

Then use kormem to recall what I've said, decided and preferred before.
```

## Commands

```sh
pip install kormem            # or: uvx kormem …, npm install -g kormem, bunx kormem …
kormem init                   # key file, engine, background services, Claude Code plugin

kormem collection add ~/notes --name notes
kormem context add kormem://notes "Personal notes and ideas"
kormem search "project timeline"            # keyword search
kormem vsearch "how to deploy"              # semantic search
kormem query "what did we decide about X?"  # ranked notes and sources; --deep runs the recall agent
kormem get "#abc123"                        # open a source by docid
kormem remember "I prefer pnpm over npm"
kormem mcp                                  # MCP server for Claude Code, Claude Desktop, Cursor
```

One key, in `~/.kormem/.env` (`kormem init` creates it): `OPENAI_API_KEY` runs extraction and the recall agent with `gpt-6-luna` and embeddings with `text-embedding-3-small`. `GEMINI_API_KEY` is optional (`kormem model set embedding gemini-embedding-2`). `TYPESAFE_API_KEY` is optional too: [Jev](https://typesafe.ai), a fast decision model, judges which sources answer each question in about 0.3 s for about a tenth of a cent, with no LLM call; short previews of those sources are sent to TypeSafe. Without any key, embeddings run locally and kormem still indexes and searches. The answer comes from your assistant's own model. `kormem model` shows and changes the model behind each job; pick the embedding model once, since changing it re-embeds everything.

## Benchmark

[LongMemEval-S](https://arxiv.org/abs/2410.10813), all 500 questions, scored by the official GPT-4o judge (`evaluate_qa.py`) unless marked as Mem0's custom judge. Each question gets one answer from kormem's deep recall (linked vault and recall agent; about 8 full sessions, ~32k tokens of context). All Kormem rows answer from the same cached contexts.

| System | Answer model | Accuracy |
|---|---|---|
| **Kormem** | **Claude Sonnet 5.5 (Mem0's custom judge)** | **95.6%** |
| [Mastra Observational Memory](https://mastra.ai/research/observational-memory) | gpt-5-mini | 94.9% |
| **Kormem** | **Claude Sonnet 5.5** | **94.4%** |
| **Kormem** | **GPT-6.1 Sol** | **93.8%** |
| **Kormem** | **GPT-6 Astra** | **93.6%** |
| [Mem0 platform](https://github.com/mem0ai/memory-benchmarks) | GPT-5 (custom judge) | 93.4–94.4% |
| [gbrain](https://github.com/garrytan/gbrain) | Claude Sonnet 4.6 | 86.6% |
| [Supermemory](https://supermemory.ai/research/longmembench) | Gemini 3 Pro | 85.2% |
| [Zep](https://arxiv.org/abs/2501.13956) | GPT-4o | 71.2% |

Other systems' numbers are self-reported. Setups and caveats: [full report](https://github.com/kushalsinha/kormem/blob/main/docs/benchmarks/longmemeval.md). Mem0's figure uses its own, more lenient judge. Re-judging our misses with that judge (same prompt, GPT-5) raises Kormem with Sonnet to **95.6%**, Sol and Astra to 94.4%.

MemPalace publishes only retrieval recall (R@5: a gold session in the top 5), not answer accuracy, so it gets its own table. Same 500 questions, session level:

| System | LLM in ranking | R@5 | All gold sessions in top 5 |
|---|---|---|---|
| **Kormem** | **yes** | **99.8%** | **95.8%** |
| [MemPalace](https://github.com/mempalace/mempalace) hybrid v4 + LLM rerank | yes | 99.2% | 89.8% |
| [MemPalace](https://github.com/mempalace/mempalace) hybrid v4 (held-out 450) | no | 98.4% | 88.7% |
| [MemPalace](https://github.com/mempalace/mempalace) raw | no | 96.6% | 85.0% |

MemPalace figures are recomputed from its committed per-question results; method in the [full report](https://github.com/kushalsinha/kormem/blob/main/docs/benchmarks/longmemeval.md#retrieval-recall).

Full documentation, diagrams and the Claude Code plugin: https://github.com/kushalsinha/kormem
