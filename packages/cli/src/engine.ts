import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { getKormemHome } from "./paths.js";

/**
 * The kormem engine and command set are a Python package (`pip install kormem`).
 * The npm package installs the matching version once into ~/.kormem/engine and
 * runs it, so `npx kormem`, `bunx kormem` and `pip install kormem` give the same
 * `kormem` command.
 *
 * Order: $KORMEM_PYTHON (a Python that already has kormem installed), the engine
 * next to this file in a source checkout, then the managed install. The managed
 * install uses uv when it is on PATH (it fetches Python 3.12 if needed), else a
 * system Python 3.11+ with venv + pip. $KORMEM_ENGINE_SPEC overrides what is
 * installed (a version, a wheel path or a git URL).
 */

const here = dirname(fileURLToPath(import.meta.url));
const isWindows = process.platform === "win32";

export const VERSION: string = JSON.parse(readFileSync(join(here, "..", "package.json"), "utf-8")).version;

export interface EnginePython {
  python: string;
  env: NodeJS.ProcessEnv;
}

function sourceCheckout(): string | null {
  const dir = join(here, "..", "..", "engine");
  return existsSync(join(dir, "pyproject.toml")) && existsSync(join(dir, "kormem", "cli")) ? dir : null;
}

function runs(cmd: string, args: string[]): string | null {
  const r = spawnSync(cmd, args, { encoding: "utf-8", stdio: ["ignore", "pipe", "ignore"] });
  return r.status === 0 ? r.stdout.trim() : null;
}

/** A Python 3.11+ on PATH; with `withDeps`, one that can already import the engine's dependencies. */
function findPython(withDeps = false): string | null {
  const check = "import sys; assert sys.version_info >= (3, 11)" + (withDeps ? "; import httpx, fastapi, yaml" : "");
  for (const cmd of ["python3", "python", "python3.13", "python3.12", "python3.11"]) {
    if (runs(cmd, ["-c", check]) !== null) return cmd;
  }
  return null;
}

function step(cmd: string, args: string[]): void {
  const r = spawnSync(cmd, args, { stdio: ["ignore", "inherit", "inherit"] });
  if (r.status !== 0) {
    throw new Error(`kormem: '${cmd} ${args.join(" ")}' failed while installing the engine.`);
  }
}

/** Install into a fresh venv next to the current one and swap it in only when the install succeeded. */
function install(engineDir: string, venv: string, spec: string): void {
  process.stderr.write(`kormem: installing the engine (${spec}) into ${venv}. This happens once per version.\n`);
  const next = `${venv}.new`;
  const nextPython = isWindows ? join(next, "Scripts", "python.exe") : join(next, "bin", "python");
  rmSync(next, { recursive: true, force: true });
  mkdirSync(engineDir, { recursive: true });
  try {
    if (runs("uv", ["--version"]) !== null) {
      step("uv", ["venv", "--quiet", "--python", "3.12", next]);
      step("uv", ["pip", "install", "--quiet", "--python", nextPython, spec]);
    } else {
      const system = findPython();
      if (!system) {
        throw new Error(
          "kormem needs Python 3.11+ or uv to run its engine.\n" +
            "  Install uv (it fetches Python for you): curl -LsSf https://astral.sh/uv/install.sh | sh\n" +
            "  or install Python 3.11+ from https://www.python.org/downloads/",
        );
      }
      step(system, ["-m", "venv", next]);
      step(nextPython, ["-m", "pip", "install", "--quiet", "--upgrade", "pip"]);
      step(nextPython, ["-m", "pip", "install", "--quiet", spec]);
    }
  } catch (err) {
    rmSync(next, { recursive: true, force: true });
    throw err;
  }
  rmSync(venv, { recursive: true, force: true });
  renameSync(next, venv);
}

export function enginePython(): EnginePython {
  const env = { ...process.env };
  if (process.env.KORMEM_PYTHON) return { python: process.env.KORMEM_PYTHON, env };

  const checkout = sourceCheckout();
  if (checkout) {
    env.PYTHONPATH = [checkout, process.env.PYTHONPATH].filter(Boolean).join(isWindows ? ";" : ":");
    return { python: findPython(true) || findPython() || "python3", env };
  }

  const engineDir = join(getKormemHome(), "engine");
  const venv = join(engineDir, "venv");
  const python = isWindows ? join(venv, "Scripts", "python.exe") : join(venv, "bin", "python");
  const spec = process.env.KORMEM_ENGINE_SPEC || `kormem==${VERSION}`;
  const marker = join(engineDir, "installed");
  const installed = existsSync(marker) ? readFileSync(marker, "utf-8").trim() : "";
  if (!existsSync(python) || installed !== spec) {
    install(engineDir, venv, spec);
    writeFileSync(marker, spec + "\n");
  }
  return { python, env };
}
