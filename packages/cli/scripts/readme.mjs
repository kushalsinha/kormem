// Copies the repo README into the npm package, pointing relative links at GitHub
// so they still work on npmjs.com. npm can't play GitHub's inline video, so the
// launch video becomes a poster image linking to YouTube.
import { readFileSync, writeFileSync, copyFileSync } from "node:fs";

const REPO = "https://github.com/kushalsinha/kormem/blob/main/";
const VIDEO_POSTER = `<p align="center">
  <a href="https://youtu.be/5kTirMTDepk">
    <img src="https://raw.githubusercontent.com/kushalsinha/kormem/main/docs/media/kormem-launch-poster.jpg" alt="Watch the kormem launch video (68 s)" width="720">
  </a>
</p>`;
const readme = readFileSync("../../README.md", "utf8")
  .replace(/^https:\/\/github\.com\/user-attachments\/assets\/\S+$/m, VIDEO_POSTER)
  .replace(/\]\((?!https?:|#|mailto:)(?:\.\/|\/)?([^)]+)\)/g, (_, path) => `](${REPO}${path})`);
writeFileSync("README.md", readme);
copyFileSync("../../LICENSE", "LICENSE");
