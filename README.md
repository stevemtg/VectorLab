# Vector Lab — real local model steering

A React, TypeScript, Tailwind CSS, and Lucide dashboard connected to your actual local models. The Python bridge runs on your computer. There are **no scripted completions, invented activation readings, or simulated extraction scores**.

## Start

From this directory in Windows PowerShell:

```powershell
.\Start-VectorLab.ps1
```

Open **http://127.0.0.1:5173/**. Keep the launcher terminal open. It starts the frontend and the loopback model bridge at `127.0.0.1:8788`, reusing healthy existing services. Ctrl+C stops only processes it started. Ollama must be running at `127.0.0.1:11434`.

On a fresh checkout, run `./Setup-VectorLab.ps1` first. Setup installs project dependencies, an isolated Python environment, and the official portable llama.cpp **b11146 Windows x64 CPU build**, with a pinned SHA-256 check. It does not download model weights. Requires Node.js 22.13+, Python 3.13, and your installed Ollama models.

## Two real runtimes

| Runtime | Inference | Extraction and injection | Metrics |
| --- | --- | --- | --- |
| Native steering · CPU | Loads the existing GGUF directly with llama.cpp | Actual per-layer residual capture and additive control vectors | Residual L2 norms, concept cosine alignment, injection norm, token count and throughput |
| Ollama chat | Streams `/api/chat` from the selected installed model | Disabled: Ollama's chat API does not expose residual injection | Actual token count and throughput; hidden activations are unavailable |

Select a model and runtime, then click **Connect**. Native mode was verified with your installed `qwen2.5-coder:1.5b` (28 layers, 1,536 hidden dimensions). Four real directions—Pleasure, Happiness / Joy, Sycophancy, and Euphoria—were extracted and saved for that exact checkpoint. All sliders start at zero. Your other installed models appear in the selector. Compatibility with every architecture is not guaranteed; unsupported native graphs/templates return an explicit error, and no simulation is substituted.

The native bridge runs on CPU, leaving your existing GPU-backed Ollama session alone. There is no fixed GGUF file-size limit; loading larger models depends on available system memory and llama.cpp support for the model architecture. Native steering uses a rolling 2,048-token context, so generation continues beyond one context window until the model finishes or you stop it. Ollama generation is likewise unbounded. Responses stream token-by-token into the Live playground. For unsupported templates, use Ollama chat. Native sampling is greedy for reproducible comparisons.

## Actual extraction and steering

The backend evaluates each prompt line independently (up to eight per class) and captures `l_out-N` residual tensors through llama.cpp's graph evaluation callback. Choose mean-token or final-token pooling. Extraction subtracts the baseline mean from the positive mean and projects out the baseline principal components covering 50% of its variance when multiple baseline samples are available. The resulting contrast retains its raw magnitude.

The extraction layer is selected by training AUC. Its vector stays fixed while candidate injection layers are compared using the vector norm divided by the mean final-token residual norm of three independent neutral probes. The candidate closest to 0.6 is selected. Each coefficient scales that same extracted vector at the selected injection layer. Multiple vectors add at their respective layers. The native adapter supports layers 1 through N−1 (zero-based decoder indices); layer 0 is not steered.

This follows the raw-vector and ratio-selection mechanics of [Pain-axis](https://github.com/valen-research/Pain-axis/tree/4d75cd90e206ea962f7a9101e65c85efea56723b). It is not a full reproduction: the reference selects extraction layers with grouped five-fold held-out AUC and uses its research datasets. This workspace's small user-supplied samples provide **training AUC only**, explicitly labeled in the UI.

New extraction results report the **raw contrast L2 at the extraction layer**. Training AUC, even with multiple sample lines, does not establish concept specificity or behavioral reliability. Real models do not guarantee the original simulator's prescribed tone at a particular coefficient. Larger values may change semantics, coherence, or correctness instead of simply increasing enthusiasm.

Each completion snapshots its coefficients. Moving sliders during generation stages values for the next completion. Reset stages zero; the injection switch bypasses all directions. The backend clears the KV cache for each generation and uses the supplied history, preventing stale cached activations from contaminating comparisons.

Vectors and prompt pairs persist under ignored `data/vectors/`. They are bound to the SHA-256 identity of the model blob; a direction from a different checkpoint is rejected. The frontend restores saved vectors at neutral strength. Older unit-normalized vectors remain usable with their original injection across layers and are labeled Legacy; unavailable AUCs are not invented. Re-extract a concept to use the new raw-vector method. Deleting a vector removes its local metadata and NumPy file. Chat history and slider values are session-local.

## Measured telemetry

Native generation samples actual layer norms and alignment after the first generated token, then every eight tokens and at completion. The chart shows a sequence of measurements, not a fabricated time series. Injection L2 is the norm of the actual combined control tensor. The pause button freezes displayed measurements. Ollama-only mode explicitly shows unavailable hidden-state measurements.

## Local access

The bridge binds to loopback only, validates Host and Origin, and requires a custom header on mutations. It accepts model names from Ollama's installed catalog; the browser cannot supply arbitrary filesystem paths. Model operations are serialized, coefficients are bounded, stored arrays use `allow_pickle=False`, and cancellation signals the native CPU evaluation callback.

The Site frontend can also connect to this local bridge on the same computer; the browser may request local-network access. The Python runtime and model weights are not hosted in the Site or uploaded. If browser policy blocks a hosted page from accessing loopback, use the local URL above.

## Verification

```powershell
node node_modules/typescript/bin/tsc --noEmit
.venv\Scripts\python.exe -m unittest backend.test_native backend.test_server -v
node node_modules/@playwright/test/cli.js test
node scripts/run-framework.mjs build
```

Keep the frontend and bridge running for browser tests. Tests use Microsoft Edge on Windows. They exercise real extraction, slider bounds, actual inference, measured tensors, deletion, cancellation, coefficient snapshots, injection bypass, invalid vector rejection, Origin checks, and responsive widths from 320 to 1,440 pixels.

The native test checks the model's **actual logits**: λ=3 changes them, while resetting λ=0 restores the baseline. An independent Ollama API check also generated a real arithmetic response. Tests require the existing 1.5B checkpoint and never download a substitute.

## Files

- `app/page.tsx`, `app/globals.css`: dashboard and responsive styling.
- `lib/model-api.ts`: typed bridge client and genuine NDJSON streaming.
- `backend/native.py`: pinned C ABI bindings, tensor capture, extraction, native injection, and inference.
- `backend/server.py`: local API, model discovery, persistence, and Ollama integration.
- `Setup-VectorLab.ps1`, `Start-VectorLab.ps1`: reproducible setup and launcher.

Optional WebMCP tools expose read-only workspace inspection and coefficient staging when the browser supports `document.modelContext`. A native WebMCP-capable browser context was unavailable during testing, so that optional compatibility remains unverified.

Runtime references: [llama.cpp control vectors](https://github.com/ggml-org/llama.cpp/blob/b11146/include/llama.h), [residual extraction example](https://github.com/ggml-org/llama.cpp/tree/b11146/tools/cvector-generator), and [Ollama chat API](https://docs.ollama.com/api/chat).
