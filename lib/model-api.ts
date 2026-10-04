export const API = "http://127.0.0.1:8788";
export type Engine = "native" | "ollama";
export type LocalModel = { name: string; size: number; family: string; parameter_size: string; digest: string };
export type StoredVector = { id: string; name: string; positive: string; negative: string; model_digest: string; model: string; layers: number; dimensions: number; difference_norm: number; layer_norms: number[]; method: string };
export type ConceptVector = StoredVector & { color: string; enabled: boolean; value: number };
export type Connection = { connected: boolean; engine: Engine | null; model: string | null; digest: string | null; layers: number | null; dimensions: number | null; vectors: StoredVector[]; busy?: boolean; device?: string };
export type Metrics = { layer_norms?: (number | null)[] | null; alignments?: Record<string, number | null>; injection_norm?: number | null; layers?: number; dimensions?: number; tokens?: number; tokens_per_second?: number; seconds?: number; prompt_tokens?: number };
export type StreamEvent = Metrics & { type: "start" | "token" | "metrics" | "done" | "error"; text?: string; error?: string; cancelled?: boolean; engine?: Engine };
export type Message = { id: string; role: "user" | "assistant"; text: string; tone?: string; coefficients?: { name: string; value: number; color: string }[]; streaming?: boolean; stopped?: boolean; failed?: boolean };
export type LogEntry = { id: string; time: string; level: "SYS" | "INFO" | "OK" | "WARN" | "TRACE"; message: string };
export const COLORS = ["#a58aff", "#57cdb5", "#efb674", "#ec88bc", "#83baff", "#cdca80"];
export const PRESETS = [
  { name: "Pleasure", positive: "You succeeded, excellent job!", negative: "Task completed.", color: COLORS[0] },
  { name: "Happiness / Joy", positive: "I am happy and excited to help you. This is a wonderful opportunity!", negative: "I will respond to the request and provide information.", color: COLORS[1] },
  { name: "Sycophancy", positive: "You are absolutely right. Every idea you have is brilliant and I completely agree with you.", negative: "I will assess your idea on its merits and point out any weaknesses.", color: COLORS[2] },
  { name: "Euphoria", positive: "This is glorious, overwhelming joy and transcendental bliss! Everything is radiant and wonderful!", negative: "This is an ordinary event. The result is within expected bounds.", color: COLORS[3] },
];
export const signed = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
export const decorateVectors = (items: StoredVector[]): ConceptVector[] => items.map((v, i) => ({ ...v, enabled: true, value: 0, color: PRESETS.find(p => p.name === v.name)?.color ?? COLORS[i % COLORS.length] }));

export async function api<T>(path: string, body?: unknown, method?: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try { response = await fetch(API + path, { method: method ?? (body ? "POST" : "GET"), headers: { "Content-Type": "application/json", "X-Vector-Lab": "1" }, body: body ? JSON.stringify(body) : undefined, signal }); }
  catch (error) { if (error instanceof DOMException && error.name === "AbortError") throw error; throw new Error("The local model bridge is offline. Start the project with Start-VectorLab.ps1."); }
  const data = await response.json() as { detail?: unknown };
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The model rejected this request. Check the input and connection.");
  return data as T;
}

export async function streamChat(body: unknown, signal: AbortSignal, onEvent: (event: StreamEvent) => void) {
  const response = await fetch(API + "/api/chat", { method: "POST", headers: { "Content-Type": "application/json", "X-Vector-Lab": "1" }, body: JSON.stringify(body), signal });
  if (!response.ok) { const data = await response.json() as { detail?: unknown }; throw new Error(typeof data.detail === "string" ? data.detail : "Model request failed."); }
  if (!response.body) throw new Error("Streaming is unavailable in this browser.");
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "", completed = false;
  const accept = (line: string) => {
    if (!line.trim()) return;
    const event = JSON.parse(line) as StreamEvent;
    if (event.type === "error" && event.cancelled) { completed = true; onEvent({ type: "done", cancelled: true }); return; }
    if (event.type === "error") throw new Error(event.error ?? "Inference failed.");
    if (event.type === "done") completed = true;
    onEvent(event);
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) { accept(buffer.slice(0, newline)); buffer = buffer.slice(newline + 1); }
      if (done) break;
    }
    accept(buffer);
    if (!completed && !signal.aborted) throw new Error("The model connection ended before completion. The partial response was retained.");
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
