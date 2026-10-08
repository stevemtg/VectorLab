// Desktop launcher/builder for Vector Lab.
//   dev     — build the renderer SPA, then open Electron against the source-tree bridge
//   build   — build the renderer SPA and stage the frozen bridge (no installer)
//   package — build everything and run electron-builder; extra args are forwarded,
//             e.g. `node scripts/desktop.mjs package --win`
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { stageDesktop } from "./stage-desktop.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mode = process.argv[2];
if (!["dev", "build", "package"].includes(mode)) {
  console.error("Usage: node scripts/desktop.mjs <dev|build|package> [electron-builder args]");
  process.exit(2);
}

const run = (command, args, label, env = process.env) => {
  const result = spawnSync(command, args, { cwd: root, stdio: "inherit", windowsHide: true, env });
  if (result.error) { console.error(`${label}: ${result.error.message}`); process.exit(1); }
  if (result.status !== 0) { console.error(`${label} failed (exit ${result.status}).`); process.exit(result.status ?? 1); }
};

run(process.execPath, [path.join(root, "node_modules", "vite", "bin", "vite.js"), "build", "--config", path.join(root, "vite.renderer.config.ts")], "Renderer build");
if (mode === "dev") {
  const electronEnv = { ...process.env };
  delete electronEnv.ELECTRON_RUN_AS_NODE; // Some editor terminals inherit this.
  run(process.execPath, [path.join(root, "node_modules", "electron", "cli.js"), "."], "Electron", electronEnv);
  process.exit(0);
}

run(process.execPath, [path.join(root, "scripts", "build-bridge.mjs")], "Bridge freeze");
stageDesktop(root);
if (mode === "package") {
  const builderCli = path.join(root, "node_modules", "electron-builder", "cli.js");
  if (!fs.existsSync(builderCli)) { console.error("electron-builder is not installed. Run npm install first."); process.exit(1); }
  run(process.execPath, [builderCli, "--config", path.join(root, "electron-builder.yml"), "--publish", "never", ...process.argv.slice(3)], "electron-builder");
} else {
  console.log("Desktop artifacts staged. Run the package script for this host platform to build installers.");
}
