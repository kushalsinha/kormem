---
name: kormem
description: Recall the user's long-term memory with kormem: past sessions, decisions, preferences, people, projects and indexed notes or documents. Use when the user refers to something from before ("what did we decide", "like last time", "my usual setup"), asks about their own history, asks you to remember something, or when an answer depends on their preferences or earlier work.
license: MIT
compatibility: Requires the kormem CLI (npm install -g kormem, or pip install kormem; then kormem init).
allowed-tools: Bash(kormem:*), mcp__kormem__*
---

# kormem: the user's memory

kormem keeps everything the user has said to assistants, and every file, folder
or URL they added, in a local vault of linked notes. Each topic note has
**Current facts**, a dated **Timeline** and **How-to** steps, and every line
links back to the source chat or file it came from.

## When to use it

- Before answering anything that may depend on what the user said, decided,
  prefers or did before. Check memory first; don't ask them to repeat it.
- When they say "remember this", "from now on" or "always do X": call
  `remember`. Lessons about how to work with them are filed automatically and
  load at the start of each session.
- When they ask about their own notes or documents that live in a collection.

## How to recall

1. **`query`** (MCP) or `kormem query "<question>"` first. In about 2 seconds
   it returns the matching topic notes and sources, each with a rank (1 = most
   relevant), a relevance score and a source id such as `#a1b2c3`. Start from
   the top-ranked sources, but lower ranks can still hold one item of a count
   or an earlier value. Use `deep: true` (`--deep`) for counts, "how many
   times", timelines, "what changed" and "what do I prefer": a recall agent
   reads further (about 30 seconds) and adds dated findings.
2. Open a source with **`get`** when you need exact wording, numbers or quotes:
   `get("#a1b2c3")`, or a line range with `get("#a1b2c3:120:40")`. Notes open by
   title: `get("Lisbon trip")`.
3. Answer from what you retrieved, and name the docids or note titles you
   relied on. Findings (from `deep`) are leads with dates; when a detail
   matters, verify it in the source.

Use `search` (keyword) when you know an exact name, code symbol or rare phrase
and want speed rather than the agent.

## Commands (CLI)

```bash
kormem query "what did we decide about the deploy process?"   # ranked notes and sources, ~2 s
kormem query "how many times did I change the API key?" --deep  # recall agent, dated findings
kormem search "blue-green" -n 10                              # fast keyword search
kormem vsearch "how do we roll back a release"                 # semantic search
kormem get "#a1b2c3"                                           # a source by docid
kormem get "Deployment"                                        # a vault note by title
kormem multi-get "kormem://notes/meetings/2026-10*"            # several by glob
kormem remember "Use pnpm, not npm, in this repo"
kormem lessons --project <repo>                                # what loads at session start
kormem ls                                                      # collections and vault folders
```

Add `--json` for structured output, `--files` for a list of addresses.

## Addresses

- `#a1b2c3`: any source (chat, file, URL, remembered fact) by docid
- `kormem://<collection>/<path>`: a file in an indexed collection
- `kormem://vault/<folder>/<note>.md`, or just the note title: a vault note

## Things to know

- Indexing is immediate; filing into the vault happens in the background once
  a chat goes quiet or a file is read, so a fact from the last few minutes may
  show up in `search` before `query` finds it in a note.
- What the assistant suggested is kept apart from what the user said or did.
  Don't report a suggestion as something the user did.
- Replaced facts stay in the note's Timeline, marked as replaced. Use the
  current value unless the user asks about history.
