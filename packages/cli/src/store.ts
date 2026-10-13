import { engineGet, enginePost } from "./engineClient.js";

export type SearchMode = "hybrid" | "keyword" | "vector";

export interface MemoryStoreOptions {
  home?: string;
  namespace?: string;
}

export interface QueryOptions {
  query: string;
  limit?: number;
  mode?: SearchMode;
  collectionId?: string;
  includeGuidance?: boolean;
  expand?: boolean;
  liveFetch?: boolean;
  window?: number;
  rerank?: boolean;
}

export interface KnowledgeItem {
  segmentKey: string;
  text: string;
  score: number;
  ceScore?: number;
  sourceUrl?: string;
  sourceTitle?: string;
  documentId?: string;
  fullDocumentText?: string;
  liveText?: string;
}

export interface NoteInfo {
  kormemId: string;
  title: string;
  para: string;
  relPath: string;
  updatedAt?: string;
}

export interface SearchResult {
  query: string;
  knowledge: KnowledgeItem[];
  guidance: { itemType: string; phrase: string; score: number; metadata?: Record<string, unknown> }[];
  citations: { index: number; segmentKey: string; textSnippet?: string }[];
  retrievalTimeMs: number;
}

export interface RecallResult {
  relevantMemory: string;
  lessons: string;
  notes: { title: string; folder: string; body: string }[];
  sources: { sid: string; title: string; kind: string; date: string; text: string }[];
  findings: string[];
  debug: Record<string, unknown>;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  idx?: number;
  at?: string;
}

export function createMemoryStore(options: MemoryStoreOptions = {}) {
  const namespaceId = options.namespace || "main";
  if (options.home) {
    process.env.KORMEM_HOME = options.home;
  }

  return {
    namespaceId,

    async init(): Promise<void> {
      await enginePost("/init", { namespaceId });
    },

    async remember(content: string, role?: "user" | "assistant"): Promise<void> {
      await enginePost("/ingest/text", {
        content,
        namespaceId,
        platform: "conversation",
        role,
      });
    },

    async ingestPath(path: string, collectionId?: string): Promise<unknown> {
      return enginePost("/ingest/path", {
        path,
        namespaceId,
        collectionId,
        recursive: true,
      });
    },

    async ingestUrl(url: string, collectionId?: string): Promise<unknown> {
      return enginePost("/ingest/url", { url, namespaceId, collectionId });
    },

    async extract(wait = false, maxJobs = 50): Promise<{ processed: number; pending: number }> {
      if (!wait) {
        const st = await engineGet<{ pendingExtractions?: number }>(
          `/status?namespaceId=${encodeURIComponent(namespaceId)}`,
        );
        return { processed: 0, pending: st.pendingExtractions ?? 0 };
      }
      return enginePost("/extract/run", { namespaceId, maxJobs });
    },

    /** Tidy memory: month shards, compact index, archive cold notes, merge duplicates, condense long notes. */
    async maintain(opts: { allNamespaces?: boolean; prefix?: string; condense?: boolean; vacuum?: boolean } = {}): Promise<unknown> {
      return enginePost("/maintain", {
        namespaceId: opts.allNamespaces || opts.prefix ? null : namespaceId,
        namespacePrefix: opts.prefix ?? "",
        condense: opts.condense ?? true,
        vacuum: opts.vacuum ?? false,
      });
    },

    async extractQueue(action: "park" | "resume", namespacePrefix = ""): Promise<{ moved: number }> {
      return enginePost("/extract/queue", { action, namespacePrefix });
    },

    async embed(): Promise<number> {
      const r = await enginePost<{ embedded: number }>("/embed", null);
      return r.embedded;
    },

    async query(opts: QueryOptions): Promise<SearchResult> {
      return enginePost<SearchResult>("/search", {
        query: opts.query,
        namespaceId,
        limit: opts.limit ?? 8,
        mode: opts.mode ?? "hybrid",
        collectionId: opts.collectionId,
        includeGuidance: opts.includeGuidance ?? true,
        expand: opts.expand ?? false,
        liveFetch: opts.liveFetch ?? false,
        window: opts.window ?? 2,
        rerank: opts.rerank,
      });
    },

    /** Notes, sources and findings for a question (recall agent + date resolution). */
    async recall(query: string, effort: "low" | "medium" = "low", project?: string): Promise<RecallResult> {
      return enginePost<RecallResult>("/recall", { query, namespaceId, effort, project });
    },

    async chatAppend(sessionId: string, turns: ChatTurn[], close = false, project?: string): Promise<unknown> {
      return enginePost("/chat/append", { namespaceId, sessionId, turns, close, project });
    },

    /** Global lessons plus the project's: what the user told the assistant to remember. */
    async lessons(project?: string): Promise<string> {
      const qs = `namespaceId=${encodeURIComponent(namespaceId)}${project ? `&project=${encodeURIComponent(project)}` : ""}`;
      const r = await engineGet<{ lessons: string }>(`/lessons?${qs}`);
      return r.lessons;
    },

    async chatClose(sessionId: string): Promise<unknown> {
      return enginePost("/chat/close", { namespaceId, sessionId });
    },

    async readNote(ref: string): Promise<Record<string, unknown>> {
      return enginePost("/notes/read", { ref, namespaceId });
    },

    async listLinks(ref: string): Promise<{ outbound: unknown[]; backlinks: unknown[] }> {
      return enginePost("/notes/links", { ref, namespaceId });
    },

    async getMoc(): Promise<string> {
      const r = await enginePost<{ content: string }>(
        `/notes/moc?namespaceId=${encodeURIComponent(namespaceId)}`,
        null,
      );
      return r.content;
    },

    async listNotes(para?: string): Promise<NoteInfo[]> {
      const qs = para
        ? `?namespaceId=${encodeURIComponent(namespaceId)}&para=${encodeURIComponent(para)}`
        : `?namespaceId=${encodeURIComponent(namespaceId)}`;
      return enginePost<NoteInfo[]>(`/notes/list${qs}`, null);
    },

    async readDocument(documentId: string): Promise<{ text: string; title?: string }> {
      return enginePost("/documents/read", { documentId, namespaceId });
    },

    async status(): Promise<Record<string, unknown>> {
      return engineGet(`/status?namespaceId=${encodeURIComponent(namespaceId)}`);
    },
  };
}
