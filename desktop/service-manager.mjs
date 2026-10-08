// Only this manager owns the bridge process. No port scanning or process-name kills.
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";

export function resolveBridgePython(appRoot) {
  const python = process.env.VECTOR_LAB_BRIDGE_PYTHON ?? (process.platform === "win32"
    ? path.join(appRoot, ".venv", "Scripts", "python.exe")
    : path.join(appRoot, ".venv", "bin", "python"));
  if (!fs.existsSync(python)) throw new Error(`Bridge interpreter missing: ${python}. Create the project venv and install backend/requirements.txt first.`);
  return python;
}

export function bridgeLaunch({ packaged, resourcesPath, backendDir, python, platform }) {
  if (!packaged) return { command: python, args: [path.join(backendDir, "main.py")] };
  const executable = path.join(resourcesPath, "bridge", platform === "win32" ? "vectorlab-bridge.exe" : "vectorlab-bridge");
  if (!fs.existsSync(executable)) throw new Error(`Packaged bridge missing: ${executable}. Reinstall Vector Lab.`);
  return { command: executable, args: [] };
}

export class ServiceManager extends EventEmitter {
  constructor({ launch, cwd, env, logFile, timeoutMs = 30000, stopTimeoutMs = 8000 }) {
    super();
    Object.assign(this, { launch, cwd, env, logFile, timeoutMs, stopTimeoutMs });
    this.child = null;
    this.status = { state: "stopped", detail: "" };
    this.token = randomBytes(32).toString("hex");
    // File import is a main-process operation, with a capability never sent to the page.
    this.controlToken = randomBytes(32).toString("hex");
    this.instance = randomBytes(16).toString("hex");
    this.tail = "";
    this.url = null;
    this.stopTask = null;
  }

  update(state, detail = "") {
    this.status = { state, detail };
    this.emit("status", this.status);
  }

  log(chunk) {
    const safe = String(chunk).replaceAll(this.token, "[redacted]").replaceAll(this.controlToken, "[redacted]");
    this.tail = (this.tail + safe).slice(-8000);
    if (!this.logFile) return;
    try {
      fs.mkdirSync(path.dirname(this.logFile), { recursive: true });
      if (fs.existsSync(this.logFile) && fs.statSync(this.logFile).size > 1024 * 1024) fs.writeFileSync(this.logFile, "", { mode: 0o600 });
      fs.appendFileSync(this.logFile, safe, { mode: 0o600 });
    } catch { /* Log failures must not strand the child process. */ }
  }

  async start() {
    if (this.child || this.status.state !== "stopped") throw new Error("The bridge has already been started.");
    this.update("starting");
    const child = spawn(this.launch.command, this.launch.args, {
      cwd: this.cwd, windowsHide: true, shell: false, stdio: ["pipe", "pipe", "pipe"],
      env: { ...this.env, VECTOR_LAB_PORT: "0", VECTOR_LAB_TOKEN: this.token,
        VECTOR_LAB_CONTROL_TOKEN: this.controlToken, VECTOR_LAB_INSTANCE_ID: this.instance,
        VECTOR_LAB_PARENT_PIPE: "1", PYTHONUNBUFFERED: "1" },
    });
    this.child = child;
    child.stdin.on("error", () => {}); // EPIPE is possible when startup fails.
    child.stderr.setEncoding("utf8").on("data", chunk => this.log(chunk));
    let timer;
    let buffer = "";
    const ready = new Promise((resolve, reject) => {
      this.failStartup = reject;
      timer = setTimeout(() => reject(new Error(`Bridge startup timed out after ${this.timeoutMs / 1000}s.`)), this.timeoutMs);
      child.stdout.setEncoding("utf8").on("data", chunk => {
        buffer = (buffer + chunk).slice(-65536);
        let newline;
        while ((newline = buffer.indexOf("\n")) !== -1) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          try {
            const message = JSON.parse(line);
            if (message.type === "vectorlab-ready" && message.instance === this.instance && Number.isInteger(message.port) && message.port > 0 && message.port < 65536) {
              resolve(`http://127.0.0.1:${message.port}`);
              continue;
            }
          } catch { /* Ordinary bridge stdout is a diagnostic. */ }
          this.log(line + "\n");
        }
      });
      child.once("error", reject);
      child.once("close", (code, signal) => {
        this.child = null;
        const detail = `Bridge exited (code ${code ?? "none"}, signal ${signal ?? "none"}).`;
        reject(new Error(detail));
        if (!this.stopTask) this.update("failed", `${detail} Restart Vector Lab.\n${this.tail}`);
      });
    });
    try {
      const url = await ready;
      clearTimeout(timer);
      const response = await fetch(`${url}/api/health`, { headers: { Authorization: `Bearer ${this.token}` }, signal: AbortSignal.timeout(2000) });
      const health = await response.json();
      if (!response.ok || health.instance !== this.instance || this.child !== child || this.stopTask) throw new Error("The owned bridge failed its readiness check.");
      this.url = url;
      this.update("running");
      return this.connection();
    } catch (error) {
      const detail = `${error.message}\n${this.tail}`.trim();
      await this.stop();
      this.update("failed", detail);
      throw new Error(detail);
    } finally {
      clearTimeout(timer);
      this.failStartup = null;
    }
  }

  connection() {
    if (this.status.state !== "running" || !this.child) throw new Error(this.status.detail || "The model bridge is unavailable.");
    return { url: this.url, token: this.token };
  }

  stop() {
    if (this.stopTask) return this.stopTask;
    this.update("stopping");
    this.failStartup?.(new Error("Bridge startup was cancelled."));
    const child = this.child;
    // Closing the parent pipe requests graceful shutdown on Windows too. If the
    // main process crashes, the OS closes this pipe and Python also shuts down.
    this.stopTask = new Promise(resolve => {
      if (!child || child.exitCode !== null || child.signalCode !== null) { resolve(); return; }
      const timer = setTimeout(() => {
        // Kill only our direct, one-folder PyInstaller child; never Ollama.
        child.kill("SIGKILL");
      }, this.stopTimeoutMs);
      child.once("close", () => { clearTimeout(timer); resolve(); });
      child.stdin.end();
    }).then(() => { this.url = null; this.update("stopped"); });
    return this.stopTask;
  }
}
