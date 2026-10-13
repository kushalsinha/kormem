# OpenClaw

Kormem follows the same integration pattern as [QMD](https://github.com/tobi/qmd): install globally, index local paths, point the agent at hybrid retrieval.

```bash
bun install -g @kormem/kormem
kormem init
kormem collection add ~/.openclaw/agents/main/memory --name agent-logs
kormem ingest ~/.openclaw/agents/main/memory --collection agent-logs
kormem embed
```

Conceptual config (when OpenClaw adds a kormem backend):

```yaml
memory:
  backend: kormem
  kormem:
    namespace: main
    home: ~/.kormem
    update:
      interval: 5m
      onBoot: true
    limits:
      maxResults: 8
      maxSnippetChars: 800
```

Until native backend support ships, use MCP:

```json
{
  "mcpServers": {
    "kormem": { "command": "kormem", "args": ["mcp"] }
  }
}
```
