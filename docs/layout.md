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
embeddingProvider: gemini      # gemini | local | hash
embeddingModel: gemini-embedding-2
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
| **Gemini (default)** | `GEMINI_API_KEY` set | `gemini-embedding-2` | 768 |
| **Local fallback** | No API key | `BAAI/bge-base-en-v1.5` (fastembed ONNX) | 768 |

Environment variables:

- `GEMINI_API_KEY` — enables cloud embeddings
- `GEMINI_EMBEDDING_MODEL` — default `gemini-embedding-2`
- `GEMINI_EMBEDDING_DIMENSIONS` — default `768`
- `KORMEM_LOCAL_EMBED_MODEL` — offline model path/name
- `KORMEM_EMBEDDING_PROVIDER` — force `gemini`, `local`, or `hash`

**Migration:** If you changed provider or dimension, run `kormem embed -f` or delete `index.sqlite` and re-ingest. `kormem doctor` warns on dimension mismatch.
