#!/usr/bin/env node
import { spawn } from "node:child_process";
import { enginePython } from "./engine.js";

/** `kormem …` from npm / npx / bun / bunx: run the same command in the kormem Python package. */
let target;
try {
  target = enginePython();
} catch (err) {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
}

const child = spawn(target.python, ["-m", "kormem.cli", ...process.argv.slice(2)], {
  stdio: "inherit",
  env: target.env,
});
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
  process.on(signal, () => child.kill(signal));
}
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 1);
});
child.on("error", (err) => {
  console.error(`kormem: could not start the engine: ${err.message}`);
  process.exit(1);
});
