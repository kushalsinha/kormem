#!/usr/bin/env node
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { join } from "node:path";
import { getKormemHome } from "./paths.js";
import { engineGet, enginePost } from "./engineClient.js";
import { createMemoryStore } from "./store.js";
import { startMcpServer } from "./mcp.js";
import { runClaudeCodeHook } from "./hook.js";

const args = process.argv.slice(2);
const command = args[0];
const jsonMode = args.includes("--json");
const namespace = getArgValue("--namespace") || "main";

function getArgValue(flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 && args[i + 1] ? args[i + 1] : undefined;
}

function restArgs(): string[] {
  return args.slice(1).filter((a) => !a.startsWith("--") && a !== namespace);
}

const INIT_CONFIG = `defaultNamespace: main
embeddingProvider: gemini
embeddingModel: gemini-embedding-2
embeddingDim: 768
localEmbedModel: BAAI/bge-base-en-v1.5
enginePort: 8787
rerankEnabled: true
rerankModel: BAAI/bge-reranker-base
queryOptimize: true
`;

async function cmdInit(): Promise<void> {
  const home = getKormemHome();
  await mkdir(join(home, "queue"), { recursive: true });
  await mkdir(join(home, "models"), { recursive: true });
  await mkdir(join(home, "logs"), { recursive: true });
  const configPath = join(home, "config.yaml");
  try {
    await readFile(configPath, "utf-8");
  } catch {
    await writeFile(configPath, INIT_CONFIG, "utf-8");
  }
  await enginePost("/init", { namespaceId: namespace });
  console.log(`Initialized kormem at ${home}`);
}

async function cmdCollectionAdd(): Promise<void> {
  const name = args[1];
  const path = args[2];
  if (!name || !path) {
    console.error("Usage: kormem collection add <name> <path>");
    process.exit(1);
  }
  const home = getKormemHome();
  const configPath = join(home, "config.yaml");
  let yaml = "";
  try {
    yaml = await readFile(configPath, "utf-8");
  } catch {
    yaml = `${INIT_CONFIG}collections: {}\n`;
  }
  if (!yaml.includes("collections:")) {
    yaml += "\ncollections:\n";
  }
  yaml += `  ${name}:\n    path: ${path}\n    pattern: "**/*"\n`;
  await writeFile(configPath, yaml, "utf-8");
  console.log(`Collection '${name}' -> ${path}`);
}

async function cmdIngest(): Promise<void> {
  const target = args[1];
  if (!target) {
    console.error("Usage: kormem ingest <path-or-url> [--collection name]");
    process.exit(1);
  }
  const collectionId = getArgValue("--collection");
  const store = createMemoryStore({ namespace });
  if (target.startsWith("http://") || target.startsWith("https://")) {
    const r = await store.ingestUrl(target, collectionId);
    output(r);
  } else {
    const r = await store.ingestPath(target, collectionId);
    output(r);
  }
}

async function cmdRemember(): Promise<void> {
  const contentIdx = args.indexOf("--content");
  let content = contentIdx >= 0 ? args[contentIdx + 1] : restArgs().join(" ");
  const role = getArgValue("--role") as "user" | "assistant" | undefined;
  if (!content) {
    console.error("Usage: kormem remember <text> | kormem remember --content \"...\" [--role user]");
    process.exit(1);
  }
  const store = createMemoryStore({ namespace });
  await store.remember(content, role);
  console.log("Remembered. Extraction runs in the background (kormem extract --wait to finish now).");
}

async function cmdExtract(): Promise<void> {
  const store = createMemoryStore({ namespace });
  if (args.includes("--park") || args.includes("--resume")) {
    const action = args.includes("--park") ? "park" : "resume";
    const r = await store.extractQueue(action, getArgValue("--prefix") || "");
    output({ action, ...r });
    return;
  }
  const max = Number(getArgValue("--max")) || 50;
  const r = await store.extract(args.includes("--wait"), max);
  output(r);
}

async function cmdMaintain(): Promise<void> {
  const store = createMemoryStore({ namespace });
  output(await store.maintain({
    allNamespaces: args.includes("--all"),
    prefix: getArgValue("--prefix") || undefined,
    condense: !args.includes("--no-condense"),
    vacuum: args.includes("--vacuum"),
  }));
}

async function cmdRecall(): Promise<void> {
  const query = restArgs().join(" ");
  if (!query) {
    console.error("Usage: kormem query <question> [--deep] [--json]");
    process.exit(1);
  }
  const store = createMemoryStore({ namespace });
  const r = await store.recall(query, args.includes("--deep") ? "medium" : "low");
  if (jsonMode) {
    output(r);
    return;
  }
  if (r.findings.length) {
    console.log("Findings:");
    for (const f of r.findings) console.log(`  - ${f}`);
  }
  if (r.notes.length) console.log(`\nNotes: ${r.notes.map((n) => `[[${n.title}]]`).join(", ")}`);
  if (r.sources.length) {
    console.log("\nSources:");
    for (const s of r.sources) console.log(`  ${s.sid}  ${s.date}  ${s.kind}  ${s.title}`);
  }
}

async function cmdLessons(): Promise<void> {
  const store = createMemoryStore({ namespace });
  const lessons = await store.lessons(getArgValue("--project"));
  if (jsonMode) output({ lessons });
  else console.log(lessons || "No lessons yet.");
}

async function cmdChat(): Promise<void> {
  const sub = args[1];
  const sessionId = getArgValue("--session");
  if (!sessionId || (sub !== "append" && sub !== "close")) {
    console.error('Usage: kormem chat append --session ID --role user|assistant --content "..." | kormem chat close --session ID');
    process.exit(1);
  }
  const store = createMemoryStore({ namespace });
  if (sub === "close") {
    output(await store.chatClose(sessionId));
    return;
  }
  const role = getArgValue("--role") === "assistant" ? "assistant" : "user";
  const content = getArgValue("--content") || "";
  output(await store.chatAppend(sessionId, [{ role, content }]));
}

async function cmdSearch(): Promise<void> {
  const query = args[1];
  if (!query) {
    console.error("Usage: kormem search|vsearch <query> [--expand] [--live] [--no-rerank]");
    process.exit(1);
  }
  const mode =
    command === "search" ? "keyword" : "vector";
  const store = createMemoryStore({ namespace });
  const result = await store.query({
    query,
    mode,
    limit: Number(getArgValue("-n")) || 8,
    expand: args.includes("--expand"),
    liveFetch: args.includes("--live"),
    rerank: args.includes("--no-rerank") ? false : undefined,
  });
  output(result);
}

async function cmdNotes(): Promise<void> {
  const sub = args[1];
  const store = createMemoryStore({ namespace });
  if (sub === "ls" || sub === "list") {
    const para = getArgValue("--para");
    const notes = await store.listNotes(para);
    output(notes);
    return;
  }
  if (sub === "read") {
    const ref = args[2];
    if (!ref) {
      console.error("Usage: kormem notes read <ref>");
      process.exit(1);
    }
    output(await store.readNote(ref));
    return;
  }
  if (sub === "links") {
    const ref = args[2];
    if (!ref) {
      console.error("Usage: kormem notes links <ref>");
      process.exit(1);
    }
    output(await store.listLinks(ref));
    return;
  }
  if (sub === "moc") {
    console.log(await store.getMoc());
    return;
  }
  console.error("Usage: kormem notes ls|read|links|moc");
  process.exit(1);
}

function output(data: unknown): void {
  if (jsonMode) {
    console.log(JSON.stringify(data, null, 2));
  } else if (typeof data === "object" && data && "knowledge" in (data as object)) {
    const r = data as { knowledge: { text: string; score: number }[] };
    for (const k of r.knowledge) {
      console.log(`(${k.score.toFixed(3)}) ${k.text.slice(0, 300)}...\n`);
    }
  } else {
    console.log(data);
  }
}

async function cmdEmbed(): Promise<void> {
  const force = args.includes("-f") || args.includes("--force");
  const baseUrl = await import("./engineClient.js").then((m) => m.ensureEngineRunning());
  const url = `${baseUrl}/embed?force=${force}${namespace ? `&namespaceId=${encodeURIComponent(namespace)}` : ""}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`Embed failed: ${res.status}`);
  const data = (await res.json()) as { embedded: number };
  console.log(`Embedded ${data.embedded} segments${force ? " (force re-embed)" : ""}.`);
}

async function cmdStatus(): Promise<void> {
  const st = await engineGet<Record<string, unknown>>(
    `/status?namespaceId=${encodeURIComponent(namespace)}`,
  );
  output(st);
}

async function cmdDoctor(): Promise<void> {
  const checks: string[] = [];
  try {
    const { execSync } = await import("node:child_process");
    execSync("python3 --version", { stdio: "pipe" });
    checks.push("python3: ok");
  } catch {
    checks.push("python3: MISSING (required for engine)");
  }
  const geminiKey = process.env.GEMINI_API_KEY;
  checks.push(`GEMINI_API_KEY: ${geminiKey ? "set (gemini-embedding-2 @ 768)" : "not set (local fastembed fallback)"}`);
  try {
    await engineGet("/health");
    checks.push(`engine: ok (${getKormemHome()})`);
    const emb = await engineGet<{
      provider: string;
      model: string;
      dimension: number;
      storedDims: number[];
      warnings: string[];
      rerankEnabled?: boolean;
      rerankModel?: string;
    }>("/embeddings/status");
    checks.push(`embeddings: ${emb.provider} / ${emb.model} / ${emb.dimension}-dim`);
    if (emb.storedDims?.length) {
      checks.push(`index dims: ${emb.storedDims.join(", ")}`);
    }
    checks.push(`rerank: ${emb.rerankEnabled ? emb.rerankModel || "enabled" : "disabled"}`);
    for (const w of emb.warnings || []) {
      checks.push(`warning: ${w}`);
    }
  } catch {
    checks.push("engine: not running (will auto-start on next command)");
  }
  console.log(checks.join("\n"));
}

async function main(): Promise<void> {
  if (!command || command === "help" || command === "--help") {
    printHelp();
    return;
  }
  if (command === "mcp") {
    await startMcpServer(namespace);
    return;
  }
  if (command === "init") {
    await cmdInit();
    return;
  }
  if (command === "collection" && args[1] === "add") {
    await cmdCollectionAdd();
    return;
  }
  if (command === "ingest") {
    await cmdIngest();
    return;
  }
  if (command === "remember") {
    await cmdRemember();
    return;
  }
  if (command === "extract") {
    await cmdExtract();
    return;
  }
  if (command === "maintain") {
    await cmdMaintain();
    return;
  }
  if (command === "notes") {
    await cmdNotes();
    return;
  }
  if (command === "query") {
    await cmdRecall();
    return;
  }
  if (command === "lessons") {
    await cmdLessons();
    return;
  }
  if (command === "chat") {
    await cmdChat();
    return;
  }
  if (command === "hook") {
    await runClaudeCodeHook(namespace);
    return;
  }
  if (command === "search" || command === "vsearch") {
    await cmdSearch();
    return;
  }
  if (command === "embed") {
    await cmdEmbed();
    return;
  }
  if (command === "status") {
    await cmdStatus();
    return;
  }
  if (command === "doctor") {
    await cmdDoctor();
    return;
  }
  console.error(`Unknown command: ${command}`);
  printHelp();
  process.exit(1);
}

function printHelp(): void {
  console.log(`kormem — local memory for deep agents

Usage:
  kormem init
  kormem collection add <name> <path>
  kormem ingest <path-or-url> [--collection name]
  kormem remember <text> [--role user|assistant]
  kormem embed [-f]              # embed pending; -f force re-embed all
  kormem extract [--wait] [--max N]   # queue status; --wait drains it now
  kormem extract --park|--resume [--prefix lme_]   # hold or release a backlog
  kormem maintain [--all|--prefix P] [--no-condense] [--vacuum]   # shrink and tidy memory (runs daily)
  kormem query <question> [--deep]   # recall: agent walks notes + sources (--deep: more effort)
  kormem search <query>     # raw keyword search over chunks
  kormem vsearch <query>    # raw vector search over chunks
  kormem chat append --session ID --role user|assistant --content "..."
  kormem chat close --session ID
  kormem lessons [--project NAME]   # what you told the assistant to remember (global + project)
  kormem hook               # Claude Code Stop/SessionEnd hook (reads the event on stdin)
  kormem notes ls [--para areas]
  kormem notes read <ref>
  kormem notes links <ref>
  kormem notes moc
  kormem status [--json]
  kormem doctor
  kormem mcp

Environment:
  KORMEM_HOME                 Data directory (default ~/.kormem)
  KORMEM_ENGINE_URL           Engine base URL (default http://127.0.0.1:8787)
  GEMINI_API_KEY              Use gemini-embedding-2 @ 768 (default when set)
  GEMINI_EMBEDDING_MODEL      Override model (default gemini-embedding-2)
  GEMINI_EMBEDDING_DIMENSIONS Override dims (default 768)
  KORMEM_LOCAL_EMBED_MODEL    Offline fallback (default BAAI/bge-base-en-v1.5)
  KORMEM_RERANK_ENABLED       true|false (default true)
  KORMEM_RERANK_MODEL         Cross-encoder model (default BAAI/bge-reranker-base)
  KORMEM_QUERY_MODEL          Query optimizer model (default gpt-4o-mini)
`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
