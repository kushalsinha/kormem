# OpenClaw

Kormem follows the same integration pattern as [QMD](https://github.com/tobi/qmd): install globally, index local paths, point the agent at hybrid retrieval.

```bash
bun install -g kormem
kormem collection add ~/.openclaw/agents/main/memory --name agent-logs
kormem context add kormem://agent-logs "OpenClaw agent memory logs"
kormem update            # later: re-index only what changed
```

To remember the agent's conversations as chats (so they are extracted into the vault like Claude Code sessions), send turns as they happen:

```bash
kormem chat append --session <id> --role user --content "..."
kormem chat close --session <id>
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
