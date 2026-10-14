# Security policy

kormem stores personal chat history, notes and API keys on your machine, so we take security reports seriously.

## Supported versions

Security fixes go into the latest released version. Please upgrade before reporting, if you can.

| Version | Supported |
| ------- | --------- |
| Latest release | Yes |
| Older releases | No |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately in one of two ways:

1. Use GitHub's [private vulnerability reporting](../../security/advisories/new) on this repository.
2. Email **sinha.kushal@gmail.com** with the subject line `kormem security`.

Include what you found, the version (`kormem doctor` output helps), steps to reproduce, and the impact you expect. Do not include real API keys or personal data.

## What to expect

- We aim to acknowledge your report within 3 working days.
- We aim to share an assessment and a fix plan within 14 days.
- We will credit you in the release notes unless you prefer to stay anonymous.

Please give us reasonable time to ship a fix before disclosing the issue publicly.

## Areas of particular interest

- Exposure of API keys, for example from `~/.kormem/.env` or the logs.
- Access to the local engine (the HTTP server the CLI, MCP server and hooks talk to) from other users or machines.
- Path traversal or unintended file reads in `ingest`, `collection add` or `get`.
- Prompt injection through indexed content that leads the recall agent to take unintended actions or leak data.
- Supply chain issues in the npm launcher, which installs the Python engine on first run.

## Your own data

Text you index and chats you send are shared with the model providers you configure (Gemini, OpenAI, Firecrawl) when their keys are set. Without keys, nothing leaves your machine. Treat `~/.kormem` as private: it holds your keys, index and vault.
