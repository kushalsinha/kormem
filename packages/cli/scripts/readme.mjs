// Copies the repo README into the npm package without the benchmark sections
// (between <!-- benchmark:start --> and <!-- benchmark:end -->); those stay on GitHub.
import { readFileSync, writeFileSync, copyFileSync } from "node:fs";

const readme = readFileSync("../../README.md", "utf8")
  .replace(/<!-- benchmark:start -->[\s\S]*?<!-- benchmark:end -->\n?/g, "")
  .replace(/\n{3,}/g, "\n\n");
if (/benchmark|longmemeval/i.test(readme)) throw new Error("README still mentions the benchmark outside the markers");
writeFileSync("README.md", readme);
copyFileSync("../../LICENSE", "LICENSE");
