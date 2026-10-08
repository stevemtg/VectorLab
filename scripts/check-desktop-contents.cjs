// Fail the build if builder's dependency collection adds unrelated Node packages.
/* eslint-disable @typescript-eslint/no-require-imports */
const path = require("node:path");
const asar = require("@electron/asar");
const fs = require("node:fs");
module.exports = async context => {
  const resources = context.electronPlatformName === "darwin"
    ? path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`, "Contents", "Resources")
    : path.join(context.appOutDir, "resources");
  const entries = asar.listPackage(path.join(resources, "app.asar"));
  const allowed = new Set(["/desktop", "/desktop/main.mjs", "/desktop/preload.cjs", "/desktop/security.mjs",
    "/desktop/renderer-protocol.mjs", "/desktop/service-manager.mjs", "/package.json"]);
  if (entries.some(entry => !allowed.has(entry.replaceAll("\\", "/")))) throw new Error("Unexpected file in Electron ASAR. Ship only the main/preload modules and app manifest.");
  for (const resource of ["renderer/index.html", "native/runtime.json", "bridge/" + (context.electronPlatformName === "win32" ? "vectorlab-bridge.exe" : "vectorlab-bridge")]) {
    if (!fs.existsSync(path.join(resources, resource))) throw new Error(`Missing packaged resource: ${resource}`);
  }
};
