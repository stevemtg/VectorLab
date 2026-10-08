# Vector Lab desktop setup and release guide

This change implements an Electron shell and a separate Vite SPA renderer. It
keeps inference and extraction in the Python bridge. Windows x64 is the tested
native target. The macOS/Linux loader and installer configuration are supplied,
but their native libraries, signing, and platform validation remain release work.

## 1. Use the separate desktop renderer

The web build still uses vinext/Workers and produces `dist/client/` and
`dist/server/`. Its client assets do not provide a standalone `index.html`.
The desktop build uses [vite.renderer.config.ts](../vite.renderer.config.ts),
[renderer/index.html](../renderer/index.html), and
[renderer/main.tsx](../renderer/main.tsx) to bundle the existing client workspace
page and shared components into `build/renderer/index.html` plus local assets.
`next/link` resolves to a small anchor adapter. Server routes, hosted auth, and
Workers are outside this renderer. New server-only page dependencies will need a
desktop implementation before they can be used here.

```text
vector-lab/
  app/                         # Shared client workspace
  components/                  # Shared UI, preferences, notification opt-in
  lib/
    desktop.ts                 # Typed, narrow preload API and service discovery
    model-api.ts               # Fetch/NDJSON client with desktop authentication
  renderer/
    index.html
    main.tsx
    next-link.tsx
  desktop/
    main.mjs                   # Windows, IPC, lifecycle
    preload.cjs                 # Sandboxed CommonJS preload
    service-manager.mjs         # Child ownership, readiness, status, shutdown
    security.mjs                # Sender, URL, argument and CSP policy
    renderer-protocol.mjs       # Read-only static asset protocol
    *.test.mjs
  backend/
    main.py                    # Bound socket, readiness announcement, parent pipe
    server.py                  # FastAPI, authentication, persistence, inference
    native.py                  # b11146 ctypes adapter and platform loader
    requirements.txt
    requirements-build.txt     # Adds pinned PyInstaller on build hosts
  resources/native/
    LICENSE-llama.cpp
    README.md
    <platform>-<arch>/          # Supplied native artifacts, ignored by Git
  scripts/
    desktop.mjs
    build-bridge.mjs
    stage-desktop.mjs
    check-desktop-target.cjs
    check-desktop-contents.cjs
  desktop-tests/
  playwright.electron.config.ts
  vite.renderer.config.ts
  electron-builder.yml
  build/                       # Generated; separate from the web build
    renderer/
    bridge/vectorlab-bridge/    # Executable + _internal Python dependencies
    native/
    electron/                  # Dependency-free Electron app manifest/modules
  release/desktop/             # Generated installers/unpacked application
  dist/client/                 # Existing web output
  dist/server/
```

Electron serves the static renderer from `vectorlab://bundle/index.html`. Both
desktop development and production use this same origin and security policy.
Development builds once, then launches the source Python bridge; edit and relaunch
for renderer changes. There is no HMR server or production frontend server to
ship. Build scripts use separate directories so a web build cannot delete the
desktop renderer or installers.

## 2. Install build dependencies

Build hosts need Node >=22.13 and Python 3.13. End users receive the frozen Python
runtime and need neither Python nor Node. Ollama and model weights remain separate
prerequisites. Download weights while online; do not put weights in the installer.

Windows PowerShell, from the repository root:

```powershell
Set-Location -LiteralPath '.\vector-lab'
npm.cmd ci
# These exact versions are already declared and locked by this change.
# Run this only when applying the dependency change to another checkout:
npm.cmd install --save-dev --save-exact electron@44.5.1 electron-builder@26.15.3

# Existing Windows x64 setup: installs requirements and verifies b11146 archive.
.\Setup-VectorLab.ps1
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-build.txt

# If npm skips Electron's reviewed install script, explicitly download its binary:
node .\node_modules\electron\install.js
npm.cmd run dev:electron
```

Electron 44.5.1 replaces the unsupported Electron 37 dependency. Recheck and
update to a supported patched release before shipping; the lockfile makes the
chosen build repeatable. [Electron release schedule](https://releases.electronjs.org/schedule).

macOS/Linux, Bash or zsh, from `vector-lab/`:

```bash
npm ci
python3.13 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-build.txt
node node_modules/electron/install.js
# Supply resources/native/<platform>-<arch> using step 5 before packaging.
npm run dev:electron
```

The relevant `package.json` fields are:

```json
{
  "type": "module",
  "main": "desktop/main.mjs",
  "engines": { "node": ">=22.13.0" },
  "devDependencies": { "electron": "44.5.1", "electron-builder": "26.15.3" },
  "scripts": {
    "dev:electron": "node scripts/desktop.mjs dev",
    "build:renderer": "vite build --config vite.renderer.config.ts",
    "build:electron": "node scripts/desktop.mjs build",
    "package:win": "node scripts/desktop.mjs package --win",
    "package:mac": "node scripts/desktop.mjs package --mac",
    "package:linux": "node scripts/desktop.mjs package --linux",
    "test:desktop": "node --test desktop/*.test.mjs",
    "test:electron": "playwright test --config playwright.electron.config.ts"
  }
}
```

These fields supplement the existing manifest. `dev`, `build`, `start`, `local`,
`typecheck`, `lint`, and browser `test` scripts keep their existing web roles.

## 3. Run the main process and owned service

Complete executable implementations, rather than pseudocode, are in
[desktop/main.mjs](../desktop/main.mjs) and
[desktop/service-manager.mjs](../desktop/service-manager.mjs).

The main process requests a single-instance lock before starting Python. A second
launch focuses/restores the existing window. Windows/Linux quit when the last
window closes. macOS keeps the service alive and recreates the window on Dock
activation; Quit shuts it down. The window configuration explicitly includes:

```js
webPreferences: {
  preload: path.join(HERE, "preload.cjs"),
  contextIsolation: true,
  nodeIntegration: false,
  sandbox: true,
  webSecurity: true,
  allowRunningInsecureContent: false,
  webviewTag: false,
  spellcheck: false,
  devTools: !app.isPackaged,
}
```

The child starts with `shell: false`, piped stdin/stdout/stderr, and
`windowsHide: true`. Production launches
`process.resourcesPath/bridge/vectorlab-bridge[.exe]`; development launches
`backend/main.py` with the project venv interpreter. All model work remains in
that child, and launching the window does not load a GGUF.

Python binds `127.0.0.1` with port zero, retains the socket, starts Uvicorn, and
prints one readiness message containing its actual port and per-launch instance
ID. The manager accepts that message only from its child and performs an
authenticated `/api/health` check. Startup has a 30-second timeout; spawn errors,
early exits, missing resources, and readiness failures produce an error dialog
and bounded diagnostics in `userData/logs/bridge.log`. A later bridge crash is
reported through IPC and the workspace stops using the stale connection.

No port is probed and then released for another process to claim. Existing
listeners on 8788 or 11434 are never reused or killed. Quit waits for Python:
stdin EOF requests cancellation and Uvicorn shutdown, with an 8-second deadline
before terminating only the owned child. EOF also occurs if Electron crashes.
The PyInstaller one-folder layout avoids a separate unpacking child process.

## 4. Use the narrow preload and authenticated fetch client

[desktop/preload.cjs](../desktop/preload.cjs) is CommonJS because a sandboxed
preload cannot use Node's ESM loader, despite the application's ESM manifest.
[lib/desktop.ts](../lib/desktop.ts) contains the matching declarations. This is
the complete public API shape:

```ts
type ServiceStatus = {
  state: "stopped" | "starting" | "running" | "stopping" | "failed";
  detail: string;
};
type ServiceConnection = { url: string; token?: string };
type DesktopBridge = {
  readonly desktop: true;
  readonly platform: string;
  getServiceConnection(): Promise<ServiceConnection>;
  getServiceStatus(): Promise<ServiceStatus>;
  onServiceStatus(callback: (status: ServiceStatus) => void): () => void;
  showNotification(title: string, body: string): Promise<boolean>;
  importVectors(): Promise<number>;
};
```

There is no general filesystem, process, shell, or arbitrary IPC API. Import
opens a main-owned directory dialog and performs the backend request itself; the
page never supplies a filesystem path. Status listeners strip the Electron IPC
event and return an unsubscribe function. Notification strings are validated
and bounded, notifications are rate-limited, and users opt in through workspace
settings. Notification bodies contain no prompt or response content.

Every handler verifies the sending `webContents`, its owned main frame, and the
parsed exact workspace URL. URL prefix checks alone are insufficient. The
actual client in [lib/model-api.ts](../lib/model-api.ts) shares one request
function for ordinary API calls and NDJSON streaming:

```ts
async function bridgeFetch(path: string, options: RequestInit): Promise<Response> {
  if (!/^\/api\/[a-z0-9/-]+$/.test(path)) throw new Error("Invalid bridge API path.");
  const connection = await serviceConnection(API);
  const headers = new Headers({ "Content-Type": "application/json", "X-Vector-Lab": "1" });
  if (connection.token) headers.set("Authorization", `Bearer ${connection.token}`);
  return fetch(connection.url + path, {
    ...options, headers, redirect: "error", credentials: "omit",
  });
}
```

Desktop discovery validates `http://127.0.0.1:<port>` and a random 256-bit token.
Rejected discovery promises are retryable. The web renderer still uses
`http://127.0.0.1:8788` without a desktop token. Existing streamed event decoding,
telemetry, `AbortSignal`, `/api/stop`, and partial-response retention remain in
place. Python uses pinned HTTPX 0.28.1 for cancellable asynchronous Ollama streams,
including stalled headers and tokens. Stream disconnects release the inference
lock. Local requests disable environment proxies and redirects so prompts and
metadata stay on loopback. [HTTPX asynchronous streaming](https://www.python-httpx.org/async/).

## 5. Freeze Python and supply the exact native ABI

[scripts/build-bridge.mjs](../scripts/build-bridge.mjs) invokes the pinned
PyInstaller version using the project interpreter. It bundles FastAPI, Uvicorn,
NumPy, HTTPX and Python into a one-folder distribution. It explicitly includes runtime
selected Uvicorn/AnyIO modules and selects asyncio/h11 without WebSockets.
Python and Node must use the same architecture. PyInstaller builds on the target
OS; it is not a cross compiler. [PyInstaller operating modes](https://pyinstaller.org/en/stable/operating-mode.html).

The backend configuration is:

| Environment variable | Desktop value/purpose |
| --- | --- |
| `VECTOR_LAB_DATA_DIR` | `userData/data/vectors` |
| `VECTOR_LAB_RUNTIME_DIR` | `process.resourcesPath/native` |
| `VECTOR_LAB_ORIGINS` | `vectorlab://bundle` |
| `VECTOR_LAB_TOKEN` | Random per-launch bearer capability |
| `VECTOR_LAB_CONTROL_TOKEN` | Separate main-only folder-import capability |
| `VECTOR_LAB_INSTANCE_ID` | Random readiness identity |
| `VECTOR_LAB_PARENT_PIPE` | `1`, watch stdin EOF |
| `VECTOR_LAB_PORT` | `0`, atomic OS port selection |
| `VECTOR_LAB_MIGRATE_FROM` | Optional explicitly chosen legacy vector directory |
| `OLLAMA_MODELS` | Optional model directory shared with Ollama |

Frozen builds reject missing writable-data and authentication configuration.
Source web builds retain their existing `data/vectors/` and `.runtime/llama/`
defaults. Standard Ollama chat uses its API and does not require the desktop user
to read service-owned weights. Native steering still needs a single supported
GGUF blob and stores vectors against that blob's exact SHA-256 checkpoint digest.

`OLLAMA_MODELS` takes precedence; the bridge otherwise uses `~/.ollama/models`.
Linux service installations commonly store weights at
`/usr/share/ollama/.ollama/models`; explicitly configure a readable model directory
for native mode. GUI launches must inherit that setting too. Do not change weight
ownership automatically. Missing Ollama errors explain how to start the external
service; no installer launches or terminates Ollama. For offline use, select local
models and disable Ollama cloud features in Ollama's own configuration.
[Ollama configuration and model paths](https://docs.ollama.com/faq).

Only the Windows x64 archive is supplied by the existing setup. Its DLLs are
verified against the pinned archive before staging. Other targets need
`resources/native/<platform>-<arch>/`, using Node platform names (`darwin`,
`linux`, `win32`) and architecture names (`x64`, `arm64`). Include core libraries,
CPU plugins, all loader dependencies, SONAME aliases/targets, and their licenses.
The adapter now loads `.dll`, `.dylib`, and `.so`; this is a loader implementation,
not evidence of tested cross-platform inference.

macOS/Linux Bash, proposed CPU runtime build, run separately on each target host
with Git, CMake and a native C/C++ toolchain installed:

```bash
git clone --depth 1 --branch b11146 https://github.com/ggml-org/llama.cpp.git build/llama-source
# Linux:
cmake -S build/llama-source -B build/llama-native \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN'
# macOS: use the same command, replacing the last RPATH value with '@loader_path'.
cmake --build build/llama-native --config Release --parallel

# Example Linux x64 staging; choose the correct target directory for your host.
mkdir -p resources/native/linux-x64
cp -L build/llama-native/bin/lib*.so* resources/native/linux-x64/
cp build/llama-source/LICENSE resources/native/linux-x64/LICENSE-llama.cpp
printf '%s\n' '{"ref":"b11146","platform":"linux","arch":"x64"}' \
  > resources/native/linux-x64/runtime.json
ldd resources/native/linux-x64/libllama.so
# macOS equivalent: cp -L .../lib*.dylib .../darwin-arm64/ and inspect with otool -L.
```

These flags and output locations are based on the pinned
[b11146 CMake configuration](https://github.com/ggml-org/llama.cpp/blob/b11146/CMakeLists.txt).
Keep relative RPATH/install names for every dependency. Use a baseline deployment
target and the oldest supported Linux glibc build image; test on a clean host.
Do not rename a newer library to satisfy this loader. Compare C `sizeof` and
`offsetof` for the pinned public structures against the ctypes layouts, and run
the real extraction/reversible-logit tests for each OS/architecture. A provenance
manifest records the maintainer's assertion; it cannot detect a mismatched ABI.

## 6. Package resources, data and installers

The complete [electron-builder.yml](../electron-builder.yml) is:

```yaml
appId: dev.vectorlab.app
productName: Vector Lab
directories:
  app: build/electron
  output: release/desktop
files:
  - desktop/*.mjs
  - desktop/preload.cjs
  - package.json
  - '!**/node_modules{,/**}'
asar: true
npmRebuild: false
beforePack: scripts/check-desktop-target.cjs
afterPack: scripts/check-desktop-contents.cjs
extraResources:
  - from: build/renderer
    to: renderer
  - from: build/bridge/vectorlab-bridge
    to: bridge
  - from: build/native
    to: native
win:
  target: [nsis]
nsis:
  oneClick: false
  perMachine: false
  allowToChangeInstallationDirectory: true
mac:
  target: [dmg]
  category: public.app-category.developer-tools
  darkModeSupport: true
  hardenedRuntime: true
  notarize: true
  binaries:
    - Contents/Resources/bridge/vectorlab-bridge
linux:
  target: [AppImage, deb]
  category: Development
```

Native libraries and the Python executable stay outside ASAR. The generated
`build/electron/package.json` has no production dependencies. Explicit exclusions
and an ASAR content assertion also prevent electron-builder's parent-directory
dependency fallback from copying web packages. Staging includes only necessary
native libraries and notices. Source, tests, venvs, tool caches, model weights,
user vectors and preferences are not application resources. Electron's own
runtime notices must remain; supply consolidated frontend/Python dependency
notices before release. [Builder application contents](https://www.electron.build/v26/docs/contents/).

Windows PowerShell:

```powershell
npm.cmd run build:electron
npm.cmd run package:win -- --x64
# Development package without the installer:
node .\node_modules\electron-builder\cli.js --config electron-builder.yml --win --x64 --dir --publish never
```

macOS/Linux Bash, on matching native hosts:

```bash
npm run package:mac -- --arm64   # Apple Silicon; use --x64 on an Intel host
npm run package:linux -- --x64   # Linux x64; use --arm64 on a validated arm64 host
```

The target hook refuses mismatched Python/native/Electron targets, including
universal macOS packages. Never generate a DMG around a Windows bridge. Builds
explicitly disable automatic publishing.

Vectors live under Electron `userData/data/vectors/`, and workspace preferences
use the stable scheme's Chromium localStorage under `userData`. Development uses
`Vector Lab Development` as a separate profile. `--user-data-dir=<path>` supports
isolated profiles for CI and smoke tests.

For migration, click **Import existing vector folder**, choose the source
checkout's `data/vectors`, then connect its original checkpoint. Import validates
UUIDs, metadata, array shapes and finite numeric data; disables NumPy pickle;
skips duplicate or malformed pairs and links escaping the selected directory;
and copies arrays before atomically publishing JSON. It never rewrites digests
or deletes the source. Existing browser preferences can be set again in desktop
settings; the application does not copy browser profile databases.

Optional one-time migration for Windows development:

```powershell
$env:VECTOR_LAB_MIGRATE_FROM = (Resolve-Path -LiteralPath '.\data\vectors').Path
npm.cmd run dev:electron
Remove-Item Env:VECTOR_LAB_MIGRATE_FROM
```

## 7. Keep the service and renderer boundaries intact

The scheme handler provides CSP as a response header. Its `connect-src` allows
only the owned bridge's exact loopback origin. Scripts are local without eval;
inline styles remain allowed because the existing UI positions components and
colors dynamically. Frames, plugins, forms, and base-URL changes are denied.
Assets are checked through real paths inside the renderer directory; missing
assets return 404 instead of HTML. Navigation, redirects, downloads and webviews
are blocked; external links use an explicit HTTPS hostname allowlist. Web
permissions are denied except sanitized clipboard writes from the workspace.
Web security and sandboxing remain enabled. [Electron security guidance](https://www.electronjs.org/docs/latest/tutorial/security).

The Python Host allowlist and Origin rejection still apply. Desktop token mode
uses only `vectorlab://bundle`, replacing hosted/web origins for that child.
Mutations still require `X-Vector-Lab: 1`. Every data request also needs the
per-launch bearer token. CORS allows Authorization preflights from the exact
renderer; OPTIONS carries no data and is exempt from bearer authentication.
Folder import additionally requires a main-only capability never exposed by
preload and not included in browser CORS headers. No `file://` or `null` origin
is allowlisted, and security flags are not disabled to work around CORS.

The token is an in-memory capability available to the trusted renderer so it can
stream directly; it is absent from URLs, preferences, and logs. Origin checks
and the mutation header alone would not authenticate another local client.
This boundary does not defend against a compromised same-user OS process that
can read application memory. Inference is serialized in Python, and model reuse
avoids loading the same checkpoint again. Ollama owns its own model lifecycle.

Optional tray controls are proposed, not enabled by this change. Add a main-owned
`Tray` with Show, Stop generation (authenticated `/api/stop`), and Quit actions.
Require an explicit user preference before making window close hide to tray;
otherwise preserve the lifecycle described above. Keep the tray object alive,
provide OS-appropriate icons, destroy it on Quit, and run the same service shutdown
path. Tray behavior and native notifications need platform-specific manual tests.

## 8. Verify development and packaged behavior

Windows PowerShell:

```powershell
npm.cmd run typecheck
npm.cmd run lint
.\.venv\Scripts\python.exe -m unittest backend.test_server backend.test_native
npm.cmd run build:renderer
npm.cmd run test:desktop
npm.cmd run test:electron
npm.cmd run build                 # Existing web build

# After build:electron and an unpacked Windows package:
$env:VECTOR_LAB_TEST_PACKAGED_EXE = (Resolve-Path -LiteralPath '.\release\desktop\win-unpacked\Vector Lab.exe').Path
npm.cmd run test:electron
Remove-Item Env:VECTOR_LAB_TEST_PACKAGED_EXE
```

macOS/Linux: run the same npm scripts with `npm`, and Python tests with
`.venv/bin/python`. Set `VECTOR_LAB_RUNTIME_DIR` to the supplied native directory
and `OLLAMA_MODELS` to the real weights location before native integration tests.
The existing browser `npm test` suite still expects its web server on 5173 and a
source-mode bridge; use `VECTOR_LAB_TEST_BRIDGE` for an isolated test bridge.

The desktop tests exercise real Python readiness/auth/CORS/shutdown, failures and
timeouts, exact IPC frames, asset containment/CSP, actual Electron preload use,
preferences, and the frozen bridge outside the source tree. Frozen native tests
use the installed `qwen2.5-coder:1.5b` when available; do not count a skipped native
test as cross-platform validation.

Review verification on Windows x64 (2026-10-06): TypeScript and ESLint passed;
30 Python tests passed, including real b11146 extraction and reversible steering;
all 14 existing browser tests passed against an isolated source-mode bridge;
all 6 desktop tests passed, including the frozen bridge's native and Ollama
streams; the actual Electron smoke test passed in development and in the rebuilt
unpacked application. Both web and desktop builds passed. The NSIS installer was
built locally and its Authenticode status is `NotSigned`; installation/upgrade,
signing, and clean-machine checks remain release verification. No macOS/Linux
native inference or installer validation was performed.

Before release, manually verify on each clean target:

1. Install without Python/Node, launch without a terminal, and launch a second
   instance. Confirm one owned bridge and correct window focus/macOS reactivation.
2. With Ollama absent, confirm a useful error and a responsive window. Start
   Ollama, refresh, and confirm local model discovery. Missing native dependencies,
   missing bridge, denied data writes and startup timeout must surface diagnostics.
3. Disconnect Internet after dependencies/weights are installed. Confirm startup,
   both pooling modes, saved vectors, independent coefficients/injection toggle,
   real NDJSON output and native telemetry.
4. Cancel before the first token and during output in both engines. Confirm partial
   text remains and the next request succeeds; a stalled Ollama response also
   stops. Quit during generation and confirm the child and listening port disappear
   while the external Ollama service continues.
5. Import legacy vectors, restart, and confirm exact checkpoint association;
   changing the model must not apply incompatible vectors. Confirm preferences
   persist, corrupt imports are skipped, and the source folder is unchanged.
6. Try a wrong Host/Origin, missing bearer token, missing mutation header, forbidden
   import capability, external navigation and unsupported permission requests.
   Confirm rejection in the packaged application, with web security enabled.
7. Test opt-in completion notifications while unfocused, macOS signing/quarantine,
   Windows install/uninstall/upgrade, and Linux desktop integration on supported
   distributions. Inspect package contents and third-party notices.

## 9. Build and sign releases on platform CI

Use separate Windows x64, macOS x64/arm64, and Linux x64/arm64 jobs. Every job uses
the matching Node/Python/toolchain, creates a fresh build venv, installs locked
dependencies, supplies independently validated b11146 artifacts, runs checks,
builds its installer, and tests the installed/unpacked application. Preserve
checksums, native provenance, test evidence and signing results as CI artifacts.
GUI Electron tests on Linux need a display (for example an Xvfb session); the
packaged sandbox must also work under the target distribution's normal policies.

Windows production builds need a trusted signing certificate or configured
hardware/cloud signing provider; inject `CSC_LINK` and `CSC_KEY_PASSWORD` through
CI secrets for certificate-based signing and set `forceCodeSigning: true` in the
release job. Sign the bridge as well as the app/installer. Do not equate an
unsigned local package with a signed release.

macOS production builds need a Developer ID Application identity, hardened
runtime, and notarization credentials. Inject `CSC_LINK`/`CSC_KEY_PASSWORD` as
needed, plus either `APPLE_API_KEY`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER`, or
`APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID`. The config enables
notarization and lists the bridge executable for signing. Verify all nested
Python/native Mach-O files are signed by the expected identity, inspect any
required entitlements, and verify notarization/stapling/Gatekeeper on a clean
machine. Do not disable library validation to conceal unsigned dependencies.
[Builder macOS signing configuration](https://www.electron.build/v26/docs/mac/).

Before publishing, supply the untested platform runtimes and evidence, final
application ID/owner/maintainer metadata, `.ico`/`.icns`/Linux icons, signing
credentials, notices for all redistributed frontend/Python/native dependencies,
the supported OS/deployment baseline, fully locked Python transitive dependencies,
and installer upgrade/migration tests. Update the Electron patch release before
freezing the release. Publishing and an updater are separate deployment work;
these commands only create local artifacts.
