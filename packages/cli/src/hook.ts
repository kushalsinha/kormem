import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { createMemoryStore, type ChatTurn } from "./store.js";

/**
 * Claude Code hook: `kormem hook` reads the hook event from stdin.
 *
 * - `SessionStart`: prints the user's lessons (global + this project's);
 *   Claude Code adds hook stdout to the session's context.
 * - `Stop` (after every turn) and `SessionEnd`: sync the session transcript
 *   into kormem; `SessionEnd` also closes the session so its last stretch is
 *   extracted right away.
 *
 * The project is the git repository (or directory) name of the session's cwd;
 * it scopes the lessons extraction finds.
 *
 * Only text is kept: tool calls and tool results are skipped. Turns carry
 * their position in the transcript, so re-sending the whole transcript is
 * idempotent. Errors are swallowed so a memory problem never blocks Claude.
 */
interface HookEvent {
  session_id?: string;
  transcript_path?: string;
  hook_event_name?: string;
  cwd?: string;
}

export function projectName(cwd?: string): string | undefined {
  if (!cwd) return undefined;
  try {
    const root = execFileSync("git", ["-C", cwd, "rev-parse", "--show-toplevel"], {
      stdio: ["ignore", "pipe", "ignore"],
    }).toString().trim();
    if (root) return basename(root);
  } catch {
    /* not a git repo */
  }
  return basename(cwd);
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((b) => b && typeof b === "object" && (b as { type?: string }).type === "text")
    .map((b) => String((b as { text?: string }).text || ""))
    .join("\n")
    .trim();
}

export function transcriptTurns(jsonl: string): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (const line of jsonl.split("\n")) {
    if (!line.trim()) continue;
    let entry: Record<string, unknown>;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    const type = entry.type;
    if ((type !== "user" && type !== "assistant") || entry.isMeta) continue;
    const message = (entry.message || {}) as { content?: unknown };
    const text = textOf(message.content);
    // Slash-command wrappers and local command output are not conversation.
    if (!text || text.startsWith("<command-") || text.startsWith("<local-command")) continue;
    turns.push({ role: type, content: text, idx: turns.length, at: entry.timestamp as string | undefined });
  }
  return turns;
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf-8");
}

export async function runClaudeCodeHook(namespace: string): Promise<void> {
  try {
    const event = JSON.parse((await readStdin()) || "{}") as HookEvent;
    const store = createMemoryStore({ namespace });
    const project = projectName(event.cwd);
    if (event.hook_event_name === "SessionStart") {
      const lessons = await store.lessons(project);
      if (lessons) {
        console.log(`Lessons from kormem: what this user has asked you to remember about working with them.\n\n${lessons}`);
      }
      return;
    }
    if (!event.session_id || !event.transcript_path) return;
    const turns = transcriptTurns(await readFile(event.transcript_path, "utf-8"));
    if (!turns.length) return;
    const close = event.hook_event_name === "SessionEnd";
    await store.chatAppend(`claude-code:${event.session_id}`, turns, close, project);
  } catch (err) {
    if (process.env.KORMEM_DEBUG) console.error(`kormem hook: ${String(err)}`);
  }
}
