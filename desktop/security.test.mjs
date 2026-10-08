import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { assertSender, externalUrl, isAppUrl, validateNotification } from "./security.mjs";
import { rendererHandler } from "./renderer-protocol.mjs";

test("IPC trusts only the owned main frame and exact workspace URL", () => {
  const frame = { url: "vectorlab://bundle/index.html" };
  const contents = { mainFrame: frame };
  const window = { webContents: contents, isDestroyed: () => false };
  assert.doesNotThrow(() => assertSender({ sender: contents, senderFrame: frame }, window));
  assert.throws(() => assertSender({ sender: contents, senderFrame: { ...frame } }, window));
  for (const url of ["vectorlab://bundle.evil/index.html", "vectorlab://bundle/other.html", "https://bundle/index.html", "vectorlab://bundle/index.html?evil=1"]) assert.equal(isAppUrl(url), false);
});

test("external links and notifications reject untrusted arguments", () => {
  assert.equal(externalUrl("https://docs.ollama.com/api/chat"), "https://docs.ollama.com/api/chat");
  for (const url of ["file:///C:/Windows", "javascript:alert(1)", "http://127.0.0.1:8788", "https://github.com.evil", "https://evil@github.com", "https://github.com:8080"]) assert.equal(externalUrl(url), null);
  assert.throws(() => validateNotification({ title: {}, body: "x" }));
  assert.throws(() => validateNotification({ title: "x", body: "x".repeat(501) }));
});

test("renderer serves assets with exact-port CSP, and rejects missing assets and traversal", async () => {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), "vectorlab-renderer-"));
  try {
    fs.writeFileSync(path.join(folder, "index.html"), "<html></html>");
    const handle = rendererHandler(folder, "http://127.0.0.1:12345");
    const response = handle(new Request("vectorlab://bundle/index.html"));
    assert.equal(response.status, 200);
    assert.equal(await response.text(), "<html></html>");
    assert.match(response.headers.get("Content-Security-Policy"), /connect-src http:\/\/127\.0\.0\.1:12345;/);
    for (const url of ["vectorlab://bundle/missing.js", "vectorlab://bundle/%2e%2e%2foutside"]) assert.equal(handle(new Request(url)).status, 404);
    assert.equal(handle(new Request("vectorlab://evil/index.html")).status, 400);
  } finally { fs.rmSync(folder, { recursive: true, force: true }); }
});
