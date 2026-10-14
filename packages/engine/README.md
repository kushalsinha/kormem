# kormem

Local long-term memory for AI agents. kormem indexes everything you say to assistants and every file, folder or URL you add, files it into an Obsidian-compatible vault of linked notes, and answers questions with a small recall agent that returns dated findings and their sources.

The commands follow [QMD](https://github.com/tobi/qmd): `collection add`, `context add`, `search`, `vsearch`, `query`, `get`, `multi-get` and `ls` work the same way.

```sh
pip install kormem            # or: uvx kormem …, npm install -g kormem, bunx kormem …
kormem init                   # key file, engine, background services, Claude Code plugin

kormem collection add ~/notes --name notes
kormem context add kormem://notes "Personal notes and ideas"
kormem search "project timeline"            # keyword search
kormem vsearch "how to deploy"              # semantic search
kormem query "what did we decide about X?"  # recall agent (best quality)
kormem get "#abc123"                        # open a source by docid
kormem remember "I prefer pnpm over npm"
kormem mcp                                  # MCP server for Claude Code, Claude Desktop, Cursor
```

kormem needs one key: put `GEMINI_API_KEY` in `~/.kormem/.env`, which `kormem init` creates. It runs embeddings, extraction and the recall agent. Without it, kormem still indexes and searches locally. `kormem doctor` checks the setup.

## Benchmark

LongMemEval-S, all 500 questions, scored by the official GPT-4o judge. Each question gets one answer from a curated context of about 19k tokens.

| System | Answer model | Accuracy |
|---|---|---|
| Mastra Observational Memory | gpt-5-mini | 94.9% |
| **Kormem** | **Claude Sonnet 5.5** | **94.4%** |
| **Kormem** | **GPT-6.1 Sol** | **94.0%** |
| Mem0 platform | GPT-5 (custom judge) | 93.4–94.4% |
| gbrain | Claude Sonnet 4.6 | 86.6% |
| Supermemory | Gemini 3 Pro | 85.2% |
| Zep | GPT-4o | 71.2% |

Other systems' numbers are self-reported. Setups and caveats: [full report](https://github.com/kushalsinha/kormem/blob/main/docs/benchmarks/longmemeval.md).

Full documentation, diagrams and the Claude Code plugin: https://github.com/kushalsinha/kormem
