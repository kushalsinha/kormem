# Changelog

All notable changes to kormem are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Community and governance files: CONTRIBUTING, Code of Conduct, Security policy, issue and pull request templates, CODEOWNERS and Dependabot configuration.

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

[Unreleased]: https://github.com/kushalsinha/kormem/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kushalsinha/kormem/releases/tag/v0.1.0
