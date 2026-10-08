import assert from "node:assert/strict";
import { test } from "node:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bridgeLaunch, resolveBridgePython, ServiceManager } from "./service-manager.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("real bridge startup uses an owned ephemeral port, authenticates, and exits on pipe closure", { timeout: 15000 }, async () => {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), "vectorlab-service-"));
  const manager = new ServiceManager({ launch: bridgeLaunch({ packaged: false, backendDir: path.join(root, "backend"), python: resolveBridgePython(root) }),
    cwd: root, env: { ...process.env, VECTOR_LAB_DATA_DIR: folder, VECTOR_LAB_ORIGINS: "vectorlab://bundle" }, timeoutMs: 5000 });
  try {
    const connection = await manager.start();
    const pid = manager.child.pid;
    assert.equal((await fetch(`${connection.url}/api/health`)).status, 401);
    const headers = { Authorization: `Bearer ${connection.token}`, Origin: "vectorlab://bundle" };
    assert.equal((await fetch(`${connection.url}/api/status`, { headers })).status, 200);
    assert.equal((await fetch(`${connection.url}/api/status`, { headers: { ...headers, Origin: "https://evil.example" } })).status, 403);
    const preflight = await fetch(`${connection.url}/api/chat`, { method: "OPTIONS", headers: { Origin: "vectorlab://bundle", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type,x-vector-lab" } });
    assert.equal(preflight.status, 200);
    assert.equal(preflight.headers.get("Access-Control-Allow-Origin"), "vectorlab://bundle");
    await manager.stop();
    assert.equal(manager.status.state, "stopped");
    assert.throws(() => process.kill(pid, 0));
  } finally { await manager.stop(); fs.rmSync(folder, { recursive: true, force: true }); }
});

test("early exit reports diagnostics immediately and startup timeout terminates the owned child", { timeout: 5000 }, async () => {
  const failed = new ServiceManager({ launch: { command: process.execPath, args: ["-e", "console.error('test startup failure'); process.exit(2)"] }, cwd: root, env: process.env, timeoutMs: 3000 });
  await assert.rejects(failed.start(), /test startup failure/);
  assert.equal(failed.child, null);
  const hanging = new ServiceManager({ launch: { command: process.execPath, args: ["-e", "setInterval(() => {}, 1000)"] }, cwd: root, env: process.env, timeoutMs: 100, stopTimeoutMs: 100 });
  await assert.rejects(hanging.start(), /timed out/);
  assert.equal(hanging.child, null);
});

const frozen = path.join(root, "build", "bridge", "vectorlab-bridge", process.platform === "win32" ? "vectorlab-bridge.exe" : "vectorlab-bridge");
test("frozen bridge starts outside the source tree and persists a native extraction", { timeout: 30000, skip: !fs.existsSync(frozen) }, async () => {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), "vectorlab-frozen-"));
  const manager = new ServiceManager({ launch: { command: frozen, args: [] }, cwd: folder,
    env: { ...process.env, VECTOR_LAB_DATA_DIR: folder, VECTOR_LAB_RUNTIME_DIR: path.join(root, "build", "native"), VECTOR_LAB_ORIGINS: "vectorlab://bundle" }, timeoutMs: 10000 });
  try {
    const connection = await manager.start();
    const headers = { Authorization: `Bearer ${connection.token}`, "X-Vector-Lab": "1", "Content-Type": "application/json" };
    const catalog = await fetch(`${connection.url}/api/models`, { headers });
    const installed = catalog.ok ? (await catalog.json()).models : [];
    if (!installed.some(model => model.name === "qwen2.5-coder:1.5b")) return; // Inference is checked on model-equipped hosts.
    const connected = await fetch(`${connection.url}/api/connect`, { method: "POST", headers, body: JSON.stringify({ model: "qwen2.5-coder:1.5b", engine: "native" }) });
    assert.equal(connected.status, 200, await connected.text());
    const extracted = await fetch(`${connection.url}/api/extract`, { method: "POST", headers, body: JSON.stringify({ name: "Frozen smoke", positive: "Everything is wonderful!", negative: "The form is on the desk.", mode: "final_token" }) });
    const vector = await extracted.json();
    assert.equal(extracted.status, 200, JSON.stringify(vector));
    assert.equal(vector.mode, "final_token");
    assert.ok(fs.existsSync(path.join(folder, `${vector.id}.npy`)));
    const chat = await fetch(`${connection.url}/api/chat`, { method: "POST", headers, body: JSON.stringify({ model: "qwen2.5-coder:1.5b", messages: [{ role: "user", content: "What is 2 + 2? Answer briefly." }], coefficients: [{ id: vector.id, value: 0.1 }] }) });
    assert.equal(chat.status, 200);
    const events = (await chat.text()).trim().split("\n").map(line => JSON.parse(line));
    assert.equal(events.at(-1).type, "done");
    assert.ok(events.some(event => event.type === "token"));
    const switched = await fetch(`${connection.url}/api/connect`, { method: "POST", headers, body: JSON.stringify({ model: "qwen2.5-coder:1.5b", engine: "ollama" }) });
    assert.equal(switched.status, 200);
    const ollama = await fetch(`${connection.url}/api/chat`, { method: "POST", headers, body: JSON.stringify({ model: "qwen2.5-coder:1.5b", messages: [{ role: "user", content: "What is 2 + 2? Answer briefly." }], coefficients: [] }) });
    const ollamaEvents = (await ollama.text()).trim().split("\n").map(line => JSON.parse(line));
    assert.equal(ollamaEvents.at(-1).type, "done");
    assert.equal(ollamaEvents.at(-1).injection_norm, null);
  } finally { await manager.stop(); fs.rmSync(folder, { recursive: true, force: true }); }
});
