import { homedir } from "node:os";
import { join } from "node:path";

export function getKormemHome(): string {
  return process.env.KORMEM_HOME?.trim() || join(homedir(), ".kormem");
}

export function getEnginePort(): number {
  const raw = process.env.KORMEM_ENGINE_PORT;
  return raw ? parseInt(raw, 10) : 8787;
}

export function getEngineBaseUrl(): string {
  return process.env.KORMEM_ENGINE_URL || `http://127.0.0.1:${getEnginePort()}`;
}
