# Vector Lab

**A local workspace for extracting, managing, and testing concept vectors in language models.**

Vector Lab connects a React dashboard to models installed through Ollama. A Python bridge runs on your computer, using llama.cpp for activation steering or Ollama for standard chat. Extract directions from contrasting prompts, adjust their strength, and inspect model responses alongside measured internal activity.

- **Extract concept vectors** with mean-token or final-token activation pooling.
- **Combine steering directions** with independent coefficients and a shared injection switch.
- **Compare responses** in a streaming playground that records each completion's coefficients.
- **Inspect model activity** through residual norms, concept alignment, injection magnitude, and generation speed.
- **Save vectors locally**, associated with the exact model checkpoint used to extract them.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Using Vector Lab](#using-vector-lab)
- [How steering works](#how-steering-works)
- [Local data and access](#local-data-and-access)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Requirements

The included setup and launcher target **Windows x64**. The native runtime uses the CPU.

- **Git** to clone the repository.
- **Node.js 22.13 or later**, including npm.
- **Python 3.13** (the tested version), available as `python.exe` on your `PATH`.
- **Ollama**, running at `http://127.0.0.1:11434`, with at least one locally installed model.
- Enough system memory to load the selected model. Vector Lab does not impose a fixed model file-size limit.

The native integration tests target `qwen2.5-coder:1.5b`, with 28 layers and 1,536 hidden dimensions. Other installed models appear in the selector, but native compatibility depends on the model architecture and chat template. The bridge currently expects a single GGUF model blob in Ollama's local model store.

## Quick start

Run the following commands in Windows PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/stevemtg/CustomConceptVectorManager.git
Set-Location -LiteralPath '.\CustomConceptVectorManager'
```

If you already have a checkout, open PowerShell in the directory containing `Setup-VectorLab.ps1`.

### 2. Install dependencies

```powershell
.\Setup-VectorLab.ps1
```

Setup installs the JavaScript dependencies, creates an isolated Python environment in `.venv/`, and downloads the official llama.cpp **b11146 Windows x64 CPU build** into `.runtime/llama/`. It verifies the runtime archive against a pinned SHA-256 checksum.

Model weights are installed separately through Ollama. To see your installed models:

```powershell
ollama list
```

### 3. Start the workspace

With Ollama running, launch Vector Lab:

```powershell
.\Start-VectorLab.ps1
```

Open **[http://127.0.0.1:5173/](http://127.0.0.1:5173/)** in your browser.

| Service | Address |
| --- | --- |
| Dashboard | `http://127.0.0.1:5173` |
| Python model bridge | `http://127.0.0.1:8788` |
| Ollama | `http://127.0.0.1:11434` |

Keep the launcher terminal open. It starts the dashboard and bridge, reuses healthy existing services, and stops only the processes it started when you press **Ctrl+C**.

## Using Vector Lab

### Choose a runtime

Select an installed model and a runtime, then click **Connect**.

| Capability | Native steering · CPU | Ollama chat |
| --- | --- | --- |
| Inference | Loads the local GGUF with llama.cpp | Streams responses through Ollama |
| Vector extraction and injection | Supported | Unavailable |
| Internal measurements | Residual L2 norms, concept cosine alignment, and injection L2 | Unavailable |
| Generation measurements | Token count and throughput | Token count and throughput |

Use **Native steering · CPU** to extract and apply concept vectors. Use **Ollama chat** for standard chat or when a model's chat template is unsupported by the native adapter.

### Extract and apply a vector

1. Connect a model with **Native steering · CPU**.
2. Enter a **Vector name**, a **Positive prompt**, and a **Neutral / negative prompt**. Each prompt field accepts up to eight nonempty lines; each line is evaluated independently.
3. Choose **Mean over tokens** or **Final token** under **Activation pooling**, then click **Extract vector**.
4. Adjust the vector's coefficient under **Active steering** and send a message in the **Live playground**.
5. Compare the response and measurements with a neutral run. **Reset** stages all coefficients at zero, and the injection switch bypasses all directions.

Coefficients range from **−5 to +5** and start at zero. Positive values add the extracted direction; negative values subtract it. A fresh clone starts with an empty vector library. You can save up to 16 vectors per model.

Each completion captures its coefficients when generation starts. Changes made during generation apply to the **next completion**. For controlled comparisons, clear the chat history and repeat the same prompt with different coefficients.

### Read the measurements

During native generation, Vector Lab samples layer norms and concept alignment after the first generated token, every eight tokens, and at completion. **Injection L2** is the norm of the combined control tensor. The telemetry pause button freezes displayed measurements while generation continues.

The measurements describe the model's activity; they do not establish that a vector reliably represents a concept. Large coefficient values can change meaning, coherence, or correctness. The **training AUC** shown for an extracted vector measures separation on the prompts used to create it, rather than performance on held-out examples.

## How steering works

The native backend captures `l_out-N` residual tensors through llama.cpp's graph evaluation callback. Extraction and injection proceed in four stages:

1. **Capture and pool activations.** Evaluate each positive and baseline prompt independently, using the selected pooling mode.
2. **Calculate a contrast.** Subtract the baseline mean from the positive mean. When multiple baseline samples have nonzero variance, project out the leading baseline principal components that together account for at least 50% of that variance. Preserve the contrast's raw magnitude.
3. **Select the extraction layer.** Choose the usable layer with the highest training area under the ROC curve (AUC).
4. **Select the injection layer.** Keep the extracted vector fixed and compare candidate layers using its norm divided by the mean final-token residual norm from three independent neutral probes. Select the candidate whose ratio is closest to **0.6**.

Each coefficient scales the extracted vector at its selected injection layer. Multiple vectors add at their respective layers. Supported injection layers are **1 through N−1**, using zero-based decoder indices; layer 0 is excluded. **Contrast L2** reports the raw contrast norm at the extraction layer.

This method draws on the raw-vector and layer-ratio approach in [Pain-axis](https://github.com/valen-research/Pain-axis/tree/4d75cd90e206ea962f7a9101e65c85efea56723b). Vector Lab uses small, user-supplied prompt sets and training AUC; it does not reproduce the reference's research datasets or grouped five-fold held-out evaluation.

### Generation behavior

Native inference uses greedy sampling and clears the KV cache before each completion. The initial conversation, including its chat-template formatting, must fit within **2,048 tokens**. During generation, the context rolls forward, retaining the most recent 1,024 tokens when the window fills.

Neither runtime imposes an application-level generated-token limit. A response continues until the model finishes, an error occurs, or you stop generation. The rolling context permits longer output, but older context is discarded as the window advances.

## Local data and access

Vectors and their prompt pairs are stored in **`data/vectors/`**, which is excluded from Git. Each vector is bound to the SHA-256 identity of its model blob, so a vector from a different checkpoint is rejected. Saved vectors reload at neutral strength; chat history and slider values remain local to the browser session.

Deleting a vector removes its metadata and NumPy file. Older unit-normalized vectors are labeled **Legacy** and retain their original injection across layers. Re-extract a concept to use the current raw-vector method.

The launcher binds the bridge to loopback. The API validates the request's Host and browser Origin, requires `X-Vector-Lab: 1` on mutations, and accepts model names from Ollama's installed catalog. Model operations are serialized, coefficient values are bounded, and stored arrays are loaded with `allow_pickle=False`.

The bridge reads models from `%USERPROFILE%\.ollama\models` by default and respects `OLLAMA_MODELS` when it is set in the launcher's environment.

A permitted hosted frontend can also connect to the bridge on the same computer. The browser may request local-network access; if it blocks the connection, use the local dashboard URL. Model execution and vector storage remain on your computer. Hosted frontend origins must be listed in `ORIGINS` in [`backend/server.py`](backend/server.py).

## Development

The frontend uses React, TypeScript, Tailwind CSS, and Lucide, with a vinext/Vite development and build workflow. The backend uses FastAPI, NumPy, and a Python `ctypes` adapter for the pinned llama.cpp runtime.

Run commands from the repository root after completing setup.

### Common commands

| Command | Purpose |
| --- | --- |
| `.\Start-VectorLab.ps1` | Start the local dashboard and Python bridge |
| `npm.cmd run dev` | Start the frontend only |
| `npm.cmd run typecheck` | Check TypeScript types |
| `npm.cmd run lint` | Run ESLint |
| `npm.cmd run build` | Build the frontend |
| `npm.cmd test` | Run the Playwright browser suite |

These examples use `npm.cmd` for Windows PowerShell. The Python bridge and Ollama must remain available when developing features that use local models.

### Backend tests

```powershell
.\.venv\Scripts\python.exe -m unittest backend.test_native backend.test_server -v
```

The suite includes checks for extraction math, pooling, layer selection, vector storage, streaming, and API validation. Its native integration tests load `qwen2.5-coder:1.5b` and verify that steering changes the model's logits and that resetting the coefficient restores the baseline.

The native integration tests expect the model in the default `%USERPROFILE%\.ollama\models` location and skip when its manifest is absent. They do not use `OLLAMA_MODELS` or download model weights. The tests that load the model also require the runtime installed by the setup script.

### Browser tests

Keep the launcher and Ollama running, then run this command in a second terminal:

```powershell
npm.cmd test
```

On Windows, Playwright uses **Microsoft Edge**, which must be installed. The suite includes:

- **Model integration tests** in [`tests/workspace.spec.ts`](tests/workspace.spec.ts): extraction, coefficient controls, streamed inference, measured tensors, deletion, cancellation, injection bypass, request validation, and responsive layouts from 320 to 1,440 pixels. These require the locally installed `qwen2.5-coder:1.5b` model.
- **UI compatibility tests** in [`tests/vector-compatibility.spec.ts`](tests/vector-compatibility.spec.ts): legacy and current vector formats, saved-vector rendering, and coefficient staging, using mocked bridge responses.

### Project layout

| Path | Responsibility |
| --- | --- |
| [`app/page.tsx`](app/page.tsx), [`app/globals.css`](app/globals.css) | Dashboard behavior, layout, and styling |
| [`components/`](components/) | Reusable UI components |
| [`lib/model-api.ts`](lib/model-api.ts) | Typed bridge client and newline-delimited JSON streaming |
| [`backend/native.py`](backend/native.py) | llama.cpp bindings, activation capture, extraction, steering, and inference |
| [`backend/server.py`](backend/server.py) | Local API, model discovery, vector persistence, and Ollama integration |
| [`backend/test_native.py`](backend/test_native.py), [`backend/test_server.py`](backend/test_server.py) | Backend and native integration tests |
| [`tests/`](tests/) | Playwright browser tests |
| [`Setup-VectorLab.ps1`](Setup-VectorLab.ps1) | Dependency and native runtime installation |
| [`scripts/start-local.mjs`](scripts/start-local.mjs) | Local service startup and shutdown |

When contributing, include a clear description of the behavior change and the checks you ran. For model-specific issues, include the model tag, runtime, operating system, and steps to reproduce the issue.

Optional WebMCP tools expose workspace inspection and coefficient staging when the browser supports `document.modelContext`. Compatibility with a native WebMCP-capable browser remains unverified.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| The launcher asks you to run setup | Run `.\Setup-VectorLab.ps1` from the repository root, then launch again. |
| No models appear, or Ollama is unavailable | Confirm Ollama is running at `127.0.0.1:11434`, check `ollama list`, and use **Refresh local models**. |
| A model manifest or blob cannot be found | Confirm the model is installed locally and that the bridge's `OLLAMA_MODELS` matches Ollama's model directory. |
| Extraction or injection is unavailable | Connect with **Native steering · CPU**. These features are unavailable in **Ollama chat**. |
| The native graph or chat template is unsupported | Try a compatible instruct/chat model. For an unsupported native chat template, try **Ollama chat**. |
| The conversation exceeds the native context | Clear the chat history or shorten the prompt so the formatted conversation fits within 2,048 tokens. |
| Extraction reports no measurable contrast | Use different positive and baseline examples that produce a detectable activation difference. |
| The model is busy | Wait for the current operation to finish, or stop generation before starting another operation. |
| A hosted frontend cannot reach the bridge | Use [the local dashboard](http://127.0.0.1:5173/) and confirm the launcher is running. |

## References

- [Pain-axis reference implementation](https://github.com/valen-research/Pain-axis/tree/4d75cd90e206ea962f7a9101e65c85efea56723b)
- [llama.cpp b11146 API and control vectors](https://github.com/ggml-org/llama.cpp/blob/b11146/include/llama.h)
- [llama.cpp residual extraction example](https://github.com/ggml-org/llama.cpp/tree/b11146/tools/cvector-generator)
- [Ollama chat API](https://docs.ollama.com/api/chat)
