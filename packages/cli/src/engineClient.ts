import { spawn } from "node:child_process";
import { enginePython } from "./engine.js";
import { getEngineBaseUrl } from "./paths.js";

let engineProcess: ReturnType<typeof spawn> | null = null;

async function waitForHealth(baseUrl: string, maxMs = 90000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < maxMs) {
    try {
      const res = await fetch(`${baseUrl}/health`);
      if (res.ok) return;
    } catch {
      /* retry */
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Kormem engine not reachable at ${baseUrl}`);
}

export async function ensureEngineRunning(): Promise<string> {
  const baseUrl = getEngineBaseUrl();
  try {
    const res = await fetch(`${baseUrl}/health`);
    if (res.ok) return baseUrl;
  } catch {
    /* start */
  }

  if (!engineProcess) {
    const { python, env } = enginePython();
    const port = new URL(baseUrl).port || "8787";
    engineProcess = spawn(python, ["-m", "kormem.cli", "serve", "--port", port], {
      stdio: "ignore",
      detached: true,
      env,
    });
    engineProcess.unref();
  }
  await waitForHealth(baseUrl);
  return baseUrl;
}

export async function enginePost<T>(
  path: string,
  body?: unknown,
): Promise<T> {
  const baseUrl = await ensureEngineRunning();
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Engine ${path} failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function engineGet<T>(path: string): Promise<T> {
  const baseUrl = await ensureEngineRunning();
  const res = await fetch(`${baseUrl}${path}`);
  if (!res.ok) {
    throw new Error(`Engine GET ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}
