import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
export const REPO = join(here, "..");

export const parity = JSON.parse(readFileSync(join(here, "fixtures", "parity.json"), "utf8"));
export const weights = JSON.parse(readFileSync(join(REPO, "site", "models", "weights.json"), "utf8"));
export const config = JSON.parse(readFileSync(join(REPO, "site", "core", "qorgan-config.json"), "utf8"));

/** base64 float32 -> plain number[] (the JS core works on plain arrays). */
export function decodeEmbedding(b64) {
  const bytes = Buffer.from(b64, "base64");
  const view = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return Array.from(view);
}

const cache = new Map();
/** A fixture-backed "embedder": async, returns the recorded vector for each exact text. */
export async function fixtureEmbed(texts) {
  return texts.map((text) => {
    if (!cache.has(text)) {
      const b64 = parity.embeddings[text];
      if (b64 === undefined) throw new Error(`no recorded embedding for text: ${JSON.stringify(text.slice(0, 60))}`);
      cache.set(text, decodeEmbedding(b64));
    }
    return cache.get(text);
  });
}

export const approx = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;
