# Claude Code / Claude Desktop

Kormem plugs into Claude Code in two places:

- **Hooks** send every conversation into memory. Each turn is indexed immediately. Once a session goes quiet (30 minutes idle) or ends, kormem extracts it into the vault: what you said, what the assistant suggested, decisions you made, and **lessons** (instructions and corrections such as "use bun, not npm"). At the start of every session, the hook loads your lessons into Claude's context.
- **MCP tools** let Claude recall that memory while it works.

## Plugin (recommended)

```bash
npm install -g kormem          # or: pip install kormem
kormem init                    # installs the plugin when the claude CLI is found
```

Or install the plugin yourself:

```bash
claude plugin marketplace add kushalsinha/kormem
claude plugin install kormem@kormem
```

API keys go in `~/.kormem/.env` (`OPENAI_API_KEY` runs extraction, recall and embeddings; `kormem model` explains the setup), so the hooks and the MCP server see them however Claude Code was started. The plugin adds the MCP server, the three hooks below and a `kormem` skill that tells Claude when to recall and how to open sources. To set things up by hand instead, use the two sections below.

## 1. Hooks: remember your sessions

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "kormem hook" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "kormem hook" }] }],
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "kormem hook" }] }]
  }
}
```

`kormem hook` reads the hook event from stdin:

- `SessionStart` prints your lessons (global ones plus the current project's) for Claude to read.
- `Stop` (after each turn) syncs the session transcript. Only text is kept; tool calls and tool output are skipped.
- `SessionEnd` also closes the session, so its last stretch is extracted right away instead of after the idle window.

The project is the session's git repository name, so a lesson like "use bun here" only loads in that repo. `kormem lessons --project NAME` shows what would load. The hook never fails a Claude Code turn: errors are dropped silently (set `KORMEM_DEBUG=1` to print them).

Use `--namespace work` in the command to keep a separate memory per context.

## 2. MCP: recall while working

```bash
claude mcp add kormem -- kormem mcp
```

Claude Desktop: add the same server to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{ "mcpServers": { "kormem": { "command": "kormem", "args": ["mcp"] } } }
```

### Tools

| Tool | What it does |
|---|---|
| `query` | The topic notes and the source chats or documents a question needs (as `#docid`), each source with its rank and relevance, in about 2 s. `deep: true` runs a recall agent that walks the vault's links and adds dated findings, for counting, timeline and preference questions. |
| `get` | Open a source by `#docid`, a note by title, or a file by `kormem://` address. `"#abc123:120:40"` reads lines 120–159. |
| `multi_get` | Several notes or sources by glob or comma-separated list. |
| `search` | Fast keyword or vector search over chunks, no agent. |
| `lessons` | What you have told assistants to remember about working with you (global plus a project's). |
| `remember` | Store a fact. Facts are batched and extracted a few minutes later. |
| `ingest` | Add a file, folder or URL. It is indexed now and extracted in the background. |
| `status` | Counts, collections, and extraction jobs still queued. |

## CLI alternative

```bash
kormem query "where did I redeem the creamer coupon?"   # recall: findings + notes + sources
kormem get "#abc123"                                     # open a source from those results
kormem search "coupon"                                   # fast keyword search over chunks
kormem extract --wait                                    # finish queued extraction now
kormem maintain                                          # tidy memory now (also runs daily)
```

## Cost

Indexing uses embeddings only. Extraction makes one model call per settled source, which files knowledge, lessons and memory lines into the vault. Recall shortlists sources with a local cross-encoder for free; with `TYPESAFE_API_KEY` set, Jev re-ranks the top 30 by their previews (about $0.001 a question). The recall agent runs only for `deep` queries.

## Keeping memory small

Memory grows with every session, so the engine keeps it compact: chunk text is not copied (chunks point into their chat or document), vectors are stored as 1-byte numbers, and the memory index is filed by month, with recall scoring the last 12 months (`KORMEM_MEMORY_SCAN_MONTHS`). Replaced facts are retired from the index; the topic note's timeline keeps them.

Once a day (`KORMEM_MAINTAIN_HOURS`), or on `kormem maintain`, a maintenance pass:

- archives topic notes with no activity for 180 days (`KORMEM_ARCHIVE_DAYS`) to `4-Archive/`. Recall still reaches them when no current note matches, and new knowledge moves them back.
- merges near-duplicate notes (`KORMEM_MERGE_SIMILARITY`, 0.92). The old title becomes an alias and links are rewritten.
- condenses notes over 6,000 characters (`KORMEM_NOTE_MAX_CHARS`). Older timeline and assistant bullets become an "Earlier history" summary, with one cheap model call per note. Every source link is kept.

Lessons loaded at session start are capped at about 2k tokens (`KORMEM_LESSONS_MAX_CHARS`): the project's lessons first, then global ones, newest first. To keep only keyword search for very old chats, set `KORMEM_DROP_VECTORS_AFTER_DAYS` (off by default).
