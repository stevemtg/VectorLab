// Build a dependency-free Electron app package and a native resource directory.
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { resolveBridgePython } from "../desktop/service-manager.mjs";

export function stageDesktop(root) {
  const platform = process.platform, arch = process.arch;
  const source = platform === "win32" && arch === "x64" ? path.join(root, ".runtime", "llama")
    : path.join(root, "resources", "native", `${platform}-${arch}`);
  const pattern = platform === "win32" ? /^(llama|ggml|ggml-base|ggml-cpu[^/]*|libomp)\.dll$/
    : /^lib(llama|ggml|ggml-base|ggml-cpu[^/]*|omp|gomp)(?:\.[0-9]+)*\.(?:so|dylib)(?:\.[0-9]+)*$/;
  if (!fs.existsSync(source)) throw new Error(`Missing b11146 native resources for ${platform}-${arch}: ${source}`);
  const libraries = fs.readdirSync(source).filter(name => pattern.test(name));
  const required = platform === "win32" ? ["llama.dll", "ggml.dll", "ggml-base.dll"]
    : platform === "darwin" ? ["libllama.dylib", "libggml.dylib", "libggml-base.dylib"] : ["libllama.so", "libggml.so", "libggml-base.so"];
  if (required.some(name => !libraries.includes(name)) || !libraries.some(name => name.includes("ggml-cpu"))) throw new Error("Native runtime is missing its core libraries or CPU backend.");
  if (platform === "win32" && arch === "x64") {
    // Verify each staged DLL against the checksum-pinned archive used by Setup.
    const verify = `import hashlib, pathlib, sys, zipfile, json
root = pathlib.Path(sys.argv[1])
archive = root / '.runtime/llama-cpu.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '14cf1303ca9ac3abd94816850532f9f9a69ac66fbaca3776fc6f9061c2fac1d1', 'Pinned archive checksum mismatch'
with zipfile.ZipFile(archive) as bundle:
    for name in json.loads(sys.argv[2]):
        entries = [n for n in bundle.namelist() if pathlib.PurePosixPath(n).name == name]
        assert len(entries) == 1 and bundle.read(entries[0]) == (root / '.runtime/llama' / name).read_bytes(), 'Runtime differs from pinned archive: ' + name
`;
    const checked = spawnSync(resolveBridgePython(root), ["-c", verify, root, JSON.stringify(libraries)], { stdio: "inherit", windowsHide: true });
    if (checked.error || checked.status !== 0) throw new Error("Native runtime verification failed. Run Setup-VectorLab.ps1 with the pinned archive.");
  } else {
    const metadata = JSON.parse(fs.readFileSync(path.join(source, "runtime.json"), "utf8"));
    if (metadata.ref !== "b11146" || metadata.platform !== platform || metadata.arch !== arch) throw new Error("Native runtime manifest must match b11146 and this host architecture.");
  }
  const native = path.resolve(root, "build", "native");
  // Restrict the only recursive cleanup to this known generated directory.
  if (native !== path.join(fs.realpathSync(root), "build", "native")) throw new Error("Invalid native staging directory.");
  fs.rmSync(native, { recursive: true, force: true });
  fs.mkdirSync(native, { recursive: true });
  const names = [...libraries, ...fs.readdirSync(source).filter(name => /^(LICENSE|COPYING|NOTICE)/i.test(name) && fs.statSync(path.join(source, name)).isFile())];
  const hashes = {};
  for (const name of names) {
    const content = fs.readFileSync(path.join(source, name)); // Dereference SONAME links into actual files.
    fs.writeFileSync(path.join(native, name), content);
    hashes[name] = createHash("sha256").update(content).digest("hex");
  }
  fs.copyFileSync(path.join(root, "resources", "native", "LICENSE-llama.cpp"), path.join(native, "LICENSE-llama.cpp"));
  fs.writeFileSync(path.join(native, "runtime.json"), JSON.stringify({ ref: "b11146", platform, arch, files: hashes }, null, 2) + "\n");

  const appDir = path.join(root, "build", "electron");
  fs.mkdirSync(path.join(appDir, "desktop"), { recursive: true });
  for (const name of ["main.mjs", "preload.cjs", "service-manager.mjs", "security.mjs", "renderer-protocol.mjs"]) fs.copyFileSync(path.join(root, "desktop", name), path.join(appDir, "desktop", name));
  const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
  // Builder always collects production dependencies. A separate app manifest
  // prevents Next, Wrangler, React sources, and unrelated packages entering ASAR.
  fs.writeFileSync(path.join(appDir, "package.json"), JSON.stringify({ name: "vector-lab-desktop", version: pkg.version,
    description: "Local activation steering workspace", author: "Vector Lab contributors",
    type: "module", main: "desktop/main.mjs", dependencies: {} }, null, 2) + "\n");
}
