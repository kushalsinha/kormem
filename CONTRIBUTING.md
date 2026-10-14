# Contributing to kormem

Thanks for helping out. Bug reports, docs fixes, new features and benchmarks are all welcome.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Contributions are released under the [MIT license](LICENSE).

## Ways to help

- **Report a bug or ask a question:** open an [issue](../../issues/new/choose). Run `kormem doctor` and `kormem status` first and paste the output (remove any keys).
- **Suggest a feature:** open an issue describing the problem you want solved before writing code, so we can agree on direction.
- **Send a fix:** small, focused pull requests are the easiest to review.
- **Report a security problem:** do not open a public issue. See [SECURITY.md](SECURITY.md).

## Repository layout

- `packages/engine`: the Python engine, CLI and MCP server (published to PyPI as `kormem`).
- `packages/cli`: the npm launcher and JS SDK (published to npm as `kormem`).
- `plugin` and `.claude-plugin`: the Claude Code plugin (MCP server, hooks, recall skill).
- `docs`: layout notes, integrations and benchmarks.
- `scripts`: release and maintenance scripts.

## Set up a development environment

You need Python 3.11+ and Node 18+.

```bash
git clone https://github.com/kushalsinha/kormem.git
cd kormem

# Engine and CLI (Python)
cd packages/engine && pip install -e ".[dev]" && pytest tests -q

# npm launcher and JS SDK
cd ../cli && npm install && npx tsc

# Run a checkout; the engine runs from source
node packages/cli/dist/kormem.js status
```

Use a throwaway data directory while developing so you never touch your real memory: `export KORMEM_HOME=$(mktemp -d)`.

API keys are optional. Tests should pass without them; anything that needs a live model call must be skipped when no key is set.

## Pull request checklist

1. Branch from `main` and keep the change focused on one thing.
2. Add or update tests for behavior you change. `pytest tests -q` must pass.
3. Make sure `npx tsc` builds cleanly if you touched `packages/cli`.
4. Update the README or `docs/` when you change a command, flag or environment variable.
5. Add a line under **Unreleased** in [CHANGELOG.md](CHANGELOG.md).
6. Never commit keys, `.env` files, `*.sqlite` indexes or personal vault content.

## Style

- Python: type hints on public functions, small functions, no new dependency without a reason in the PR.
- TypeScript: strict mode, no `any` without a comment.
- Keep QMD-compatible commands and flags compatible; changes there need a note in the README migration table.
- Commit messages: a short imperative summary ("Fix chat close on empty session"), details in the body if needed.

## Releases

Maintainers cut releases with `scripts/release.sh` (use `--dry-run` to check the build). Versions follow [semantic versioning](https://semver.org/).
