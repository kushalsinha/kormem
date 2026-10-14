# Changelog

All notable changes to kormem are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Community and governance files: CONTRIBUTING, Code of Conduct, Security policy, issue and pull request templates, CODEOWNERS and Dependabot configuration.
- Benchmark: session-level retrieval recall (R@5) with MemPalace for comparison, `benchmark/retrieval_recall.py` to reproduce it, and source links for every compared system.

### Changed

- `scripts/release.sh` stages the npm release (`npm stage publish`); approve it on npmjs.com to publish.

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

[Unreleased]: https://github.com/kushalsinha/kormem/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/kushalsinha/kormem/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/kushalsinha/kormem/releases/tag/v0.1.0
