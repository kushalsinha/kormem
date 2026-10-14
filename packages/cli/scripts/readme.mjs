// Copies the repo README into the npm package, pointing relative links at GitHub
// so they still work on npmjs.com.
import { readFileSync, writeFileSync, copyFileSync } from "node:fs";

const REPO = "https://github.com/kushalsinha/kormem/blob/main/";
const readme = readFileSync("../../README.md", "utf8")
  .replace(/\]\((?!https?:|#|mailto:)(?:\.\/|\/)?([^)]+)\)/g, (_, path) => `](${REPO}${path})`);
writeFileSync("README.md", readme);
copyFileSync("../../LICENSE", "LICENSE");
