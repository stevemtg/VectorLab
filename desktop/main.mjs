import { app, BrowserWindow, dialog, ipcMain, Notification, protocol, session, shell } from "electron";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bridgeLaunch, resolveBridgePython, ServiceManager } from "./service-manager.mjs";
import { APP_ORIGIN, assertSender, externalUrl, isAppUrl, validateNotification } from "./security.mjs";
import { rendererHandler } from "./renderer-protocol.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP_ROOT = path.resolve(HERE, "..");
app.setName("Vector Lab");
// Development uses its own per-user data, without changing the source web store.
const userDataOverride = app.commandLine.getSwitchValue("user-data-dir");
if (userDataOverride) app.setPath("userData", path.resolve(userDataOverride));
else if (!app.isPackaged) app.setPath("userData", process.env.VECTOR_LAB_DEV_USER_DATA
  ? path.resolve(process.env.VECTOR_LAB_DEV_USER_DATA) : path.join(app.getPath("appData"), "Vector Lab Development"));
app.setAppUserModelId("dev.vectorlab.app");
protocol.registerSchemesAsPrivileged([{ scheme: "vectorlab", privileges: {
  standard: true, secure: true, supportFetchApi: true, corsEnabled: true, stream: true,
} }]);

let window = null;
let service = null;
let quitting = false;
let shutdownDone = false;
let exitCode = 0;
let importing = false;
let lastNotification = 0;

function wireIpc() {
  const handle = (channel, fn) => ipcMain.handle(channel, (event, ...args) => {
    assertSender(event, window);
    return fn(...args);
  });
  handle("vectorlab:get-service-connection", () => service.connection());
  handle("vectorlab:get-service-status", () => service.status);
  handle("vectorlab:show-notification", payload => {
    const options = validateNotification(payload);
    if (!Notification.isSupported() || window?.isFocused() || Date.now() - lastNotification < 3000) return false;
    lastNotification = Date.now();
    new Notification(options).show();
    return true;
  });
  handle("vectorlab:import-vectors", async () => {
    if (importing) throw new Error("A vector import is already in progress.");
    importing = true;
    try {
      const result = await dialog.showOpenDialog(window, { title: "Import an existing data/vectors folder", properties: ["openDirectory"] });
      if (result.canceled || !result.filePaths[0]) return 0;
      const { url, token } = service.connection();
      const response = await fetch(`${url}/api/vectors/import`, { method: "POST",
        headers: { "Content-Type": "application/json", "X-Vector-Lab": "1",
          Authorization: `Bearer ${token}`, "X-Vector-Lab-Control": service.controlToken },
        body: JSON.stringify({ source: result.filePaths[0] }), signal: AbortSignal.timeout(120000),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Vector import failed.");
      return body.imported;
    } finally { importing = false; }
  });
}

async function createWindow() {
  if (quitting || window) return;
  window = new BrowserWindow({ width: 1440, height: 900, minWidth: 980, minHeight: 640,
    backgroundColor: "#101115", autoHideMenuBar: true, title: "Vector Lab", show: false,
    webPreferences: { preload: path.join(HERE, "preload.cjs"), contextIsolation: true,
      nodeIntegration: false, sandbox: true, webSecurity: true, allowRunningInsecureContent: false,
      webviewTag: false, spellcheck: false, devTools: !app.isPackaged },
  });
  const created = window;
  const openExternal = raw => { const url = externalUrl(raw); if (url) void shell.openExternal(url).catch(() => {}); };
  created.webContents.setWindowOpenHandler(({ url }) => { openExternal(url); return { action: "deny" }; });
  created.webContents.on("will-navigate", (event, url) => { if (!isAppUrl(url)) { event.preventDefault(); openExternal(url); } });
  created.webContents.on("will-redirect", (event, url) => { if (!isAppUrl(url)) event.preventDefault(); });
  created.webContents.on("will-attach-webview", event => event.preventDefault());
  created.on("closed", () => { if (window === created) window = null; });
  await created.loadURL(`${APP_ORIGIN}/index.html`);
  created.show();
}

async function start() {
  const userData = app.getPath("userData");
  fs.mkdirSync(userData, { recursive: true });
  const launch = bridgeLaunch({ packaged: app.isPackaged, resourcesPath: process.resourcesPath,
    backendDir: path.join(APP_ROOT, "backend"), python: app.isPackaged ? undefined : resolveBridgePython(APP_ROOT), platform: process.platform });
  service = new ServiceManager({ launch, cwd: app.isPackaged ? userData : APP_ROOT,
    logFile: path.join(userData, "logs", "bridge.log"), env: { ...process.env,
      VECTOR_LAB_DATA_DIR: path.join(userData, "data", "vectors"), VECTOR_LAB_ORIGINS: APP_ORIGIN,
      VECTOR_LAB_RUNTIME_DIR: app.isPackaged ? path.join(process.resourcesPath, "native") :
        process.platform === "win32" && process.arch === "x64" ? path.join(APP_ROOT, ".runtime", "llama") : path.join(APP_ROOT, "resources", "native", `${process.platform}-${process.arch}`),
    },
  });
  service.on("status", status => {
    if (window && !window.isDestroyed()) window.webContents.send("vectorlab:service-status", status);
    if (status.state === "failed" && !quitting && window) dialog.showErrorBox("Vector Lab bridge stopped", `${status.detail}\nLogs: ${path.join(userData, "logs", "bridge.log")}`);
  });
  await service.start();
  if (quitting) return;
  const rendererDir = app.isPackaged ? path.join(process.resourcesPath, "renderer") : path.join(APP_ROOT, "build", "renderer");
  protocol.handle("vectorlab", rendererHandler(rendererDir, service.url));
  // Clipboard writes are limited to the workspace main frame; all other web permissions are denied.
  session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    callback(contents === window?.webContents && permission === "clipboard-sanitized-write" && details.isMainFrame === true && isAppUrl(contents.getURL()));
  });
  session.defaultSession.setPermissionCheckHandler((contents, permission) => contents === window?.webContents
    && permission === "clipboard-sanitized-write" && isAppUrl(contents.getURL()));
  session.defaultSession.on("will-download", event => event.preventDefault());
  wireIpc();
  await createWindow();
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (window) { if (window.isMinimized()) window.restore(); window.show(); window.focus(); }
    else if (service?.status.state === "running") void createWindow().catch(fail);
  });
  app.on("activate", () => { if (!window && service?.status.state === "running") void createWindow().catch(fail); });
  app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
  app.on("before-quit", event => {
    if (shutdownDone) return;
    event.preventDefault();
    if (quitting) return;
    quitting = true;
    void (service?.stop() ?? Promise.resolve()).finally(() => { shutdownDone = true; app.exit(exitCode); });
  });
  for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => app.quit());
  void app.whenReady().then(start).catch(fail);
}

function fail(error) {
  if (quitting) return;
  exitCode = 1;
  dialog.showErrorBox("Vector Lab could not start", `${error.message}\nLogs: ${path.join(app.getPath("userData"), "logs", "bridge.log")}`);
  app.quit();
}
