// Freezes the Python bridge into build/bridge/vectorlab-bridge/ for packaging.
// The llama.cpp runtime is deliberately NOT frozen here: electron-builder ships
// it verbatim as an app resource and the bridge finds it through
// VECTOR_LAB_RUNTIME_DIR (see backend/server.py). PyInstaller itself is not in
// backend/requirements.txt because the source-tree web workflow never needs it;
// install it into the same venv once: python -m pip install pyinstaller
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python = process.env.VECTOR_LAB_BRIDGE_PYTHON
  ?? (process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python"));
if (!fs.existsSync(python)) {
  console.error(`Bridge interpreter not found: ${python}\nRun Setup-VectorLab.ps1 (or set VECTOR_LAB_BRIDGE_PYTHON) first.`);
  process.exit(1);
}

const args = [
  "-m", "PyInstaller",
  "--clean", "--noconfirm",
  "--distpath", path.join(root, "build", "bridge"),
  "--workpath", path.join(root, "build", "bridge-work"),
  "--specpath", path.join(root, "build"),
  path.join(root, "backend", "main.py"),
  "--name", "vectorlab-bridge",
  "--paths", root,
  // uvicorn selects loop/protocol modules by name at runtime; PyInstaller's
  // static analysis cannot see any of them.
  "--hidden-import", "uvicorn.logging",
  "--hidden-import", "uvicorn.loops.asyncio",
  "--hidden-import", "uvicorn.protocols.http.h11_impl",
  "--hidden-import", "anyio._backends._asyncio",
];
const architecture = spawnSync(python, ["-c", "import platform; print(platform.machine().lower())"], { encoding: "utf8", windowsHide: true });
const pythonArch = { amd64: "x64", x86_64: "x64", aarch64: "arm64", arm64: "arm64" }[architecture.stdout?.trim()];
if (architecture.status !== 0 || pythonArch !== process.arch) {
  console.error("Python and Node/Electron must use the same target architecture. Build on that OS/architecture.");
  process.exit(1);
}
const result = spawnSync(python, args, { cwd: root, stdio: "inherit", windowsHide: true,
  env: { ...process.env, PYINSTALLER_CONFIG_DIR: path.join(root, "build", "pyinstaller-cache") },
});
if (result.status !== 0) {
  console.error(`PyInstaller failed (exit ${result.status ?? "signal"}). Is it installed in the bridge venv? Try: ${python} -m pip install pyinstaller`);
  process.exit(result.status ?? 1);
}
console.log("Bridge executable staged at build/bridge/vectorlab-bridge/.");
fs.writeFileSync(path.join(root, "build", "bridge-target.json"), JSON.stringify({ platform: process.platform, arch: process.arch }) + "\n");
