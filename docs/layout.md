# ~/.kormem layout

```
~/.kormem/
  config.yaml       # namespace, embedding provider, collections, optional API keys
  index.sqlite      # sources, documents, segments, FTS, embeddings
  guidance.sqlite   # persona/memory/playbooks per namespace
  queue/            # reserved for job spill files
  models/           # fastembed ONNX cache (offline fallback)
  logs/
```

Override directory: `export KORMEM_HOME=/path/to/data`

## config.yaml example

```yaml
defaultNamespace: main
embeddingProvider: auto        # auto | openai | gemini | local | hash
embeddingModel: text-embedding-3-small
embeddingDim: 768
localEmbedModel: BAAI/bge-base-en-v1.5
enginePort: 8787
firecrawlApiKey: ""            # optional, for web ingest
collections:
  notes:
    path: /Users/you/notes
    pattern: "**/*.md"
```

## Embeddings

| Mode | When | Model | Dimensions |
|------|------|-------|------------|
| **OpenAI (default)** | `OPENAI_API_KEY` set | `text-embedding-3-small` (or `-3-large`), shortened to 768 | 768 |
| **Gemini** | `GEMINI_API_KEY` is the only key, or `kormem model set embedding gemini-embedding-2` | `gemini-embedding-2` | 768 |
| **Local fallback** | No key for the chosen provider | `BAAI/bge-base-en-v1.5` (fastembed ONNX) | 768 |

Vectors are stored as int8 plus a scale: 772 bytes per 768-d vector, whichever model made it. Changing the model, provider or dimension re-embeds the whole index (`kormem model set embedding` asks first).

Environment variables:

- `OPENAI_API_KEY` — enables cloud embeddings (`text-embedding-3-small`)
- `GEMINI_API_KEY` — Gemini embeddings when it is the only key
- `GEMINI_EMBEDDING_MODEL` — embedding model name (default `gemini-embedding-2` for Gemini, `text-embedding-3-small` for OpenAI)
- `GEMINI_EMBEDDING_DIMENSIONS` — default `768`
- `KORMEM_LOCAL_EMBED_MODEL` — offline model path/name
- `KORMEM_EMBEDDING_PROVIDER` — force `gemini`, `openai`, `local`, or `hash` (prefer `kormem model set embedding`)

**Migration:** If you changed provider or dimension, run `kormem embed -f` or delete `index.sqlite` and re-ingest. `kormem doctor` warns on dimension mismatch.
