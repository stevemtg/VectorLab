# Native resources

Windows x64 development uses `.runtime/llama/`, populated by `Setup-VectorLab.ps1`
from the checksum-pinned b11146 archive. The desktop staging script verifies each
shipped DLL against that archive, copies only llama/ggml/CPU/OpenMP libraries and
notices into `build/native/`, and writes a provenance manifest. CLI tools, the RPC
server, and unrelated inference libraries are excluded.

Other platforms require libraries supplied by the release maintainer:

```text
resources/native/
  LICENSE-llama.cpp
  darwin-arm64/       # or darwin-x64, linux-x64, linux-arm64, win32-arm64
    runtime.json
    libllama.dylib
    libggml.dylib
    libggml-base.dylib
    libggml-cpu.dylib
    LICENSE-...
```

Linux must include `libllama.so`, `libggml.so`, `libggml-base.so`, the CPU plugins,
and every SONAME target/dependency needed by those libraries. macOS must include
unversioned `.dylib` aliases and their versioned targets/dependencies. Staging
dereferences these aliases, preserving each required filename as a real file.
Use relative `$ORIGIN` RPATH on Linux and `@loader_path` on macOS; verify dependency
resolution on a clean machine. Do not use environment search paths into a build
checkout as a release workaround.

An example provenance manifest for Apple Silicon is:

```json
{"ref":"b11146","platform":"darwin","arch":"arm64"}
```

This is a maintainer assertion, not an ABI detector. The loader checks the tag and
process OS/architecture. Native extraction, ctypes struct sizes/offsets, callbacks,
steering, cancellation, and reversible logits must still be validated per target.
A file rename cannot make a Windows runtime run on another OS, and an arbitrary
llama.cpp build may crash before Python can report an ABI error.

No macOS/Linux native binaries or successful platform validation are supplied by
this change. The setup scripts still target Windows x64. Follow the build and
release steps in [docs/desktop.md](../../docs/desktop.md) before publishing.
