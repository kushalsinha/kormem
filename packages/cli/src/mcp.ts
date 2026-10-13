import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { createMemoryStore } from "./store.js";

export async function startMcpServer(namespaceId = "main"): Promise<void> {
  const store = createMemoryStore({ namespace: namespaceId });
  await store.init();

  const server = new Server(
    { name: "kormem", version: "0.1.0" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: "remember",
        description: "Store text in long-term memory (user utterance or fact)",
        inputSchema: {
          type: "object",
          properties: {
            content: { type: "string" },
            role: { type: "string", enum: ["user", "assistant"] },
          },
          required: ["content"],
        },
      },
      {
        name: "lessons",
        description:
          "What this user has told assistants to remember about working with them (style, tools, workflow); global plus an optional project's",
        inputSchema: { type: "object", properties: { project: { type: "string" } } },
      },
      {
        name: "ingest",
        description: "Add a file, folder or URL to memory; it is indexed and extracted into the vault in the background",
        inputSchema: {
          type: "object",
          properties: {
            target: { type: "string", description: "Path or http(s) URL" },
            collection: { type: "string" },
          },
          required: ["target"],
        },
      },
      {
        name: "recall",
        description:
          "Answer-ready memory for a question: findings with dates, the topic notes and the source chats/documents they came from",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string" },
            deep: { type: "boolean", description: "Spend more effort (slower) for counting/timeline questions" },
          },
          required: ["query"],
        },
      },
      {
        name: "read_note",
        description: "Read a vault note by title, kormemId, or relPath",
        inputSchema: {
          type: "object",
          properties: { ref: { type: "string" } },
          required: ["ref"],
        },
      },
      {
        name: "list_links",
        description: "List outbound wikilinks and backlinks for a note",
        inputSchema: {
          type: "object",
          properties: { ref: { type: "string" } },
          required: ["ref"],
        },
      },
      {
        name: "get_moc",
        description: "Get the map-of-content index for the vault",
        inputSchema: { type: "object", properties: {} },
      },
      {
        name: "read_document",
        description: "Read full source document text by documentId",
        inputSchema: {
          type: "object",
          properties: { documentId: { type: "string" } },
          required: ["documentId"],
        },
      },
      {
        name: "status",
        description: "Memory index health and counts",
        inputSchema: { type: "object", properties: {} },
      },
    ],
  }));

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const name = req.params.name;
    const args = (req.params.arguments || {}) as Record<string, unknown>;

    if (name === "remember") {
      await store.remember(String(args.content), args.role as "user" | "assistant");
      return { content: [{ type: "text", text: "Stored in kormem memory; extraction runs in the background." }] };
    }
    if (name === "lessons") {
      const lessons = await store.lessons(args.project ? String(args.project) : undefined);
      return { content: [{ type: "text", text: lessons || "No lessons yet." }] };
    }
    if (name === "ingest") {
      const target = String(args.target);
      const collection = args.collection ? String(args.collection) : undefined;
      const r = /^https?:\/\//.test(target)
        ? await store.ingestUrl(target, collection)
        : await store.ingestPath(target, collection);
      return { content: [{ type: "text", text: JSON.stringify(r, null, 2) }] };
    }
    if (name === "recall") {
      const r = await store.recall(String(args.query), args.deep ? "medium" : "low");
      const parts: string[] = [];
      if (r.findings.length) parts.push(`Findings:\n${r.findings.map((f) => `- ${f}`).join("\n")}`);
      for (const n of r.notes) parts.push(`Note [[${n.title}]] (${n.folder}):\n${n.body.slice(0, 2000)}`);
      for (const src of r.sources) parts.push(`Source ${src.sid} · ${src.date} · ${src.title}:\n${src.text.slice(0, 3000)}`);
      if (r.relevantMemory) parts.push(`Memory:\n${r.relevantMemory.slice(0, 4000)}`);
      return {
        content: [{ type: "text", text: parts.length ? parts.join("\n\n---\n\n") : "No matching memory found." }],
      };
    }
    if (name === "read_note") {
      const note = await store.readNote(String(args.ref));
      return { content: [{ type: "text", text: JSON.stringify(note, null, 2) }] };
    }
    if (name === "list_links") {
      const links = await store.listLinks(String(args.ref));
      return { content: [{ type: "text", text: JSON.stringify(links, null, 2) }] };
    }
    if (name === "get_moc") {
      const moc = await store.getMoc();
      return { content: [{ type: "text", text: moc }] };
    }
    if (name === "read_document") {
      const doc = await store.readDocument(String(args.documentId));
      return { content: [{ type: "text", text: doc.text || "" }] };
    }
    if (name === "status") {
      const st = await store.status();
      return { content: [{ type: "text", text: JSON.stringify(st, null, 2) }] };
    }
    throw new Error(`Unknown tool: ${name}`);
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
}
