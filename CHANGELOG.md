# Changelog

All notable changes to kormem are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] - 2026-10-15

### Added

- Community and governance files: CONTRIBUTING, Code of Conduct, Security policy, issue and pull request templates, CODEOWNERS and Dependabot configuration.
- Benchmark: session-level retrieval recall (R@5) with MemPalace for comparison, `benchmark/retrieval_recall.py` to reproduce it, and source links for every compared system.
- `kormem model`: shows the model behind extraction, recall and embeddings with the recommended setup; `kormem model set` changes one. An embedding change warns, asks for confirmation and re-embeds the index.
- OpenAI embeddings: `kormem model set embedding text-embedding-3-small` (or `-3-large`), shortened to the index's 768 dimensions.
- `benchmark/recall_only.py`: retrieval-only LongMemEval check (R@k and gold sessions in the answer context), no answer model or judge.
- `benchmark/mem0_judge.py`: re-judges saved LongMemEval answers with Mem0's unified judge prompt (loaded from a clone of `mem0ai/memory-benchmarks`) to compare with Mem0's published score.

### Changed

- `scripts/release.sh` stages the npm release (`npm stage publish`); approve it on npmjs.com to publish.
- `gpt-6-luna` is the default for extraction and recall whenever `OPENAI_API_KEY` is set; Gemini runs them only when its key is the only one.
- One key: with `OPENAI_API_KEY` set, embeddings default to `text-embedding-3-small` (`embeddingProvider: auto`); Gemini embeddings are used when its key is the only one or when chosen with `kormem model set embedding`. Configs that already name a provider keep it, so existing indexes are not re-embedded. On 60 LongMemEval questions it matched `gemini-embedding-2` on retrieval (all@5 95.0% for both) at a tenth of the price.
- The engine re-embeds when the embedding model or dimension changes, not only the provider.
- Recall ranks sources by reciprocal rank fusion of the cross-encoder with memory-line, chunk-search and note-link signals, instead of the cross-encoder alone.
- The recall agent also returns sources holding a superseded value and every source behind one item of a count.
- Benchmark answer prompt: derive a detail from stated facts (durations, differences, rates, a nearby dated event) before answering "I don't know"; treat "not stated" lines in notes and memory as non-evidence.
- Benchmark results: LongMemEval-S on the current architecture, 94.4% (472/500) with Claude Sonnet 5.5, 93.8% with GPT-6.1 Sol, 93.6% with GPT-6 Astra (95.6% / 94.4% / 94.4% on Mem0's judge); R@5 99.8%, all gold sessions in top 5 95.8%.

## [0.1.1]

### Added

- `kormem init` sets everything up: the key file `~/.kormem/.env`, the engine, the background worker, daily maintenance and the Claude Code plugin (skip with `--skip-agents`).
- API keys are read from `~/.kormem/.env` (or `KORMEM_ENV_FILE`), so the MCP server and hooks see them however the client was started. `.env.example` lists the one key.
- Benchmark results in the npm and PyPI READMEs.

### Changed

- One key runs everything: `GEMINI_API_KEY` now covers embeddings, extraction, query rewriting and the recall agent (`gemini-3.8-flash`). OpenAI is optional: used when it is the only key, or when `KORMEM_EXTRACTION_MODEL` / `KORMEM_RECALL_MODEL` name a `gpt-*` model.
- The engine restarts on the next command when an API key is added or removed.
- Any command starts the engine and creates `~/.kormem` without `kormem init`.

### Fixed

- Changing the embedding provider (adding or removing `GEMINI_API_KEY`) no longer mixes vectors from two models: the old vectors are dropped and the index is re-embedded in the background.
- `kormem embed -f` re-embeds the whole index; it used to stop after 100 chunks.
- The npm README's relative links point at GitHub.

## [0.1.0]

First release.

### Added

- QMD-compatible commands: `collection`, `context`, `search`, `vsearch`, `query`, `get`, `multi-get`, `ls`, `update`, `embed`, `status`, `doctor`, `cleanup` and `mcp`.
- Memory commands: `remember`, `lessons`, `ingest`, `chat`, `extract`, `maintain` and `notes`.
- Obsidian-compatible vault of linked topic notes, with sources kept apart from what the assistant suggested.
- Recall agent behind `query` and the MCP `query` tool, with dated findings and source docids.
- Local background engine, background extraction worker and daily maintenance pass.
- MCP server and Claude Code plugin (session hooks and recall skill).
- Gemini or local (BAAI/bge-base-en-v1.5) embeddings; OpenAI-powered extraction and recall.
- Distribution as an npm launcher (`kormem`), a Python package (`kormem` on PyPI) and a JS SDK (`createMemoryStore`).

[Unreleased]: https://github.com/kushalsinha/kormem/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/kushalsinha/kormem/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/kushalsinha/kormem/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kushalsinha/kormem/releases/tag/v0.1.0
