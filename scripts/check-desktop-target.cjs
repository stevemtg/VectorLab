// electron-builder hook: PyInstaller and native libraries are host builds.
/* eslint-disable @typescript-eslint/no-require-imports */
const fs = require("node:fs");
const path = require("node:path");
module.exports = async context => {
  const root = context.packager.projectDir;
  const bridge = JSON.parse(fs.readFileSync(path.join(root, "build", "bridge-target.json"), "utf8"));
  const native = JSON.parse(fs.readFileSync(path.join(root, "build", "native", "runtime.json"), "utf8"));
  const arch = ["ia32", "x64", "armv7l", "arm64", "universal"][context.arch];
  for (const resource of ["build/renderer/index.html", "build/bridge/vectorlab-bridge/" + (context.electronPlatformName === "win32" ? "vectorlab-bridge.exe" : "vectorlab-bridge")]) {
    if (!fs.existsSync(path.join(root, resource))) throw new Error(`Missing desktop resource: ${resource}. Run npm run build:electron.`);
  }
  if ([bridge, native].some(target => target.platform !== context.electronPlatformName || target.arch !== arch)) throw new Error("Build Electron, Python and b11146 native resources on the target OS/architecture. Cross-target and universal packaging are unsupported.");
};
