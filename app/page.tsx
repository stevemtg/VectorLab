"use client";

import { useCallback, useEffect, useRef, useState, type CSSProperties, type FormEvent } from "react";
import Link from "next/link";
import { Activity, AudioLines, Box, Check, CheckCheck, ChevronRight, CircleHelp, Copy, Cpu, FlaskConical, Layers3, LoaderCircle, MessageSquare, Pause, Play, Plus, RotateCcw, Send, SlidersHorizontal, Sparkles, Square, Terminal, Trash2, TriangleAlert, Waves, X, PlugZap } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DEFAULT_WORKSPACE_PREFERENCES, WORKSPACE_PREFERENCES_KEY, WorkspaceSettings, readWorkspacePreferences, type WorkspacePreferences } from "@/components/workspace-settings";
import { api, streamChat, importVectorFolder, decorateVectors, injectionDescription, vectorSummary, signed, COLORS, PRESETS, type ConceptVector, type Connection, type Engine, type LocalModel, type LogEntry, type Message, type Metrics, type PoolingMode, type StoredVector } from "@/lib/model-api";

const ICONS = [Sparkles, Waves, AudioLines, Activity];
const WELCOME: Message = { id: "welcome", role: "assistant", text: "Connect a local model, extract a concept from a contrastive pair, then send a message. Responses and activation measurements come from your actual model.", tone: "Local model workspace" };
const EMPTY: Connection = { connected: false, engine: null, model: null, digest: null, layers: null, dimensions: null, vectors: [] };
const uid = () => crypto.randomUUID();
type Sample = { alignments: Record<string, number | null> };

function SectionTitle({ icon: Icon, children, extra }: { icon: typeof Activity; children: React.ReactNode; extra?: React.ReactNode }) {
  return <div className="section-title"><div className="flex items-center gap-2.5"><Icon size={16}/><h2>{children}</h2></div>{extra}</div>;
}

function ActivationChart({ vectors, samples }: { vectors: ConceptVector[]; samples: Sample[] }) {
  return <div className="chart-wrap"><div className="chart-labels"><span>1.0</span><span>0.5</span><span>0.0</span><span>−0.5</span><span>−1.0</span></div><svg viewBox="0 0 280 160" className="activation-chart" role="img" aria-label="Measured cosine similarity between residual activations and active concept directions">
    {[8, 44, 80, 116, 152].map(y => <line key={y} x1="0" x2="280" y1={y} y2={y} stroke="#292b35" strokeDasharray="3 5"/>)}
    {[0, 70, 140, 210, 279].map(x => <line key={x} x1={x} x2={x} y1="8" y2="152" stroke="#22242d"/>)}
    {vectors.map(v => { const points = samples.map((s, i) => { const n = s.alignments[v.id]; return n == null ? null : `${i / Math.max(1, samples.length - 1) * 280},${80 - Math.max(-1, Math.min(1, n)) * 72}`; }).filter(Boolean); return <polyline key={v.id} points={points.join(" ")} fill="none" stroke={v.color} strokeWidth="2" strokeLinejoin="round"/>; })}
  </svg><div className="chart-times"><span>Earlier samples</span><span>Latest</span></div>{!samples.length && <span className="chart-empty">Awaiting model measurements</span>}</div>;
}

export default function Home() {
  const [ready, setReady] = useState(false);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [modelName, setModelName] = useState("qwen2.5-coder:1.5b");
  const [engine, setEngine] = useState<Engine>("native");
  const [connection, setConnection] = useState<Connection>(EMPTY);
  const [connecting, setConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [vectors, setVectors] = useState<ConceptVector[]>([]);
  const [vectorToDelete, setVectorToDelete] = useState<ConceptVector | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [importing, setImporting] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [injection, setInjection] = useState(true);
  const [vectorName, setVectorName] = useState("Pleasure");
  const [positive, setPositive] = useState(PRESETS[0].positive);
  const [negative, setNegative] = useState(PRESETS[0].negative);
  const [extracting, setExtracting] = useState(false);
  const [extraction, setExtraction] = useState<StoredVector | null>(null);
  const [formError, setFormError] = useState("");
  const [mode, setMode] = useState<PoolingMode>("mean");
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([{ id: "boot", time: "—", level: "SYS", message: "Workspace initialized. Waiting for the local model bridge." }]);
  const [paused, setPaused] = useState(false);
  const [metrics, setMetrics] = useState<Metrics>({});
  const [samples, setSamples] = useState<Sample[]>([]);
  const [notice, setNotice] = useState("");
  const [copied, setCopied] = useState<string | null>(null);
  const [selection, setSelection] = useState("");
  const [workspacePreferences, setWorkspacePreferences] = useState(DEFAULT_WORKSPACE_PREFERENCES);
  const operationLock = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const pausedRef = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const nameRef = useRef<HTMLInputElement>(null);
  const libraryRef = useRef<HTMLDivElement>(null);
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null);
  const vectorsRef = useRef(vectors);
  const connectionRef = useRef(connection);
  const active = vectors.filter(v => v.enabled);
  const nativeReady = connection.connected && connection.engine === "native";
  const applied = injection && nativeReady ? active.filter(v => v.value !== 0) : [];
  const selected = vectors.find(v => v.id === selection);
  const extreme = applied.some(v => Math.abs(v.value) >= 3);
  const busy = connecting || streaming || extracting || deleting || importing;
  const measured = Object.values(metrics.alignments ?? {}).filter((v): v is number => v !== null);
  const cosine = measured.length ? measured.reduce((a, b) => a + b, 0) / measured.length : null;
  const norms = metrics.layer_norms ?? [];
  const maxNorm = Math.max(1, ...norms.filter((v): v is number => v !== null));
  const modeLabel = !connection.connected ? "Not connected" : connection.engine === "ollama" ? "Real chat · no injection" : applied.length ? "Native additive steering" : "Neutral baseline";

  const addLog = useCallback((level: LogEntry["level"], message: string) => setLogs(prev => [...prev, { id: uid(), time: new Date().toLocaleTimeString("en-GB", { hour12: false }), level, message }].slice(-70)), []);
  const restoreConnection = useCallback((state: Connection) => {
    setConnection(state); setVectors(decorateVectors(state.vectors)); setSelection(state.vectors[0]?.id ?? "");
    if (state.model) setModelName(state.model);
    if (state.engine) setEngine(state.engine);
  }, []);

  const refresh = useCallback(async () => {
    setConnectionError("");
    try {
      const [catalog, state] = await Promise.all([api<{ models: LocalModel[]; default_model: string }>("/api/models"), api<Connection>("/api/status")]);
      setModels(catalog.models); restoreConnection(state);
      if (!state.connected && state.native_available === false) setEngine("ollama");
      if (!state.connected) setModelName(catalog.models.some(m => m.name === catalog.default_model) ? catalog.default_model : catalog.models[0]?.name ?? "");
      addLog("OK", `Local bridge online · ${catalog.models.length} installed models found.${state.model ? ` Connected to ${state.model}.` : ""}`);
    } catch (error) { setConnectionError((error as Error).message); addLog("WARN", (error as Error).message); }
  }, [addLog, restoreConnection]);

  useEffect(() => { setWorkspacePreferences(readWorkspacePreferences()); setReady(true); void refresh(); return () => abortRef.current?.abort(); }, [refresh]);
  useEffect(() => {
    const desktop = window.vectorLab;
    if (!desktop) return;
    return desktop.onServiceStatus(status => {
      if (status.state === "failed") {
        abortRef.current?.abort();
        setConnection(EMPTY);
        setConnectionError("The model bridge stopped. Restart Vector Lab.");
        addLog("WARN", "The model bridge stopped. Restart Vector Lab.");
      }
    });
  }, [addLog]);
  useEffect(() => { pausedRef.current = paused; }, [paused]);
  useEffect(() => { vectorsRef.current = vectors; connectionRef.current = connection; }, [vectors, connection]);
  useEffect(() => { if (workspacePreferences.autoScroll && scrollRef.current && stickToBottom.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [messages, workspacePreferences.autoScroll]);
  useEffect(() => { if (!notice) return; const timer = setTimeout(() => setNotice(""), 5000); return () => clearTimeout(timer); }, [notice]);

  useEffect(() => {
    type Registry = { registerTool: (tool: { name: string; description: string; inputSchema: object; annotations: object; execute: (input: unknown) => unknown }, options: { signal: AbortSignal }) => void | Promise<void> };
    const registry = (document as Document & { modelContext?: Registry }).modelContext;
    if (!registry?.registerTool) return;
    const lifecycle = new AbortController();
    const register = (tool: Parameters<Registry["registerTool"]>[0]) => { try { Promise.resolve(registry.registerTool(tool, { signal: lifecycle.signal })).catch(() => {}); } catch { /* Optional browser API. */ } };
    register({ name: "read_steering_workspace", description: "Read the real local model connection and extracted vectors.", inputSchema: { type: "object", properties: {}, additionalProperties: false }, annotations: { readOnlyHint: true, untrustedContentHint: true }, execute: () => ({ connection: connectionRef.current, vectors: vectorsRef.current }) });
    register({ name: "set_steering_coefficients", description: "Stage coefficients for the next native-model completion. Does not send a message or alter an in-progress generation.", inputSchema: { type: "object", properties: { changes: { type: "array", minItems: 1, items: { type: "object", properties: { id: { type: "string" }, value: { type: "number", minimum: -5, maximum: 5 } }, required: ["id", "value"], additionalProperties: false } } }, required: ["changes"], additionalProperties: false }, annotations: { readOnlyHint: false, untrustedContentHint: false }, execute: async input => {
      const changes = (input as { changes?: unknown } | null)?.changes;
      if (!Array.isArray(changes) || !changes.length || changes.some(c => !c || typeof c.id !== "string" || !Number.isFinite(c.value) || Math.abs(c.value) > 5 || !vectorsRef.current.some(v => v.id === c.id)) || new Set(changes.map(c => c.id)).size !== changes.length) throw new Error("Use unique existing vector IDs and finite coefficients between -5 and 5.");
      const validated = changes.map(c => ({ id: c.id, value: Math.round(c.value * 10) / 10 }));
      setVectors(prev => prev.map(v => { const change = validated.find(c => c.id === v.id); return change ? { ...v, value: change.value } : v; }));
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      return { staged: validated, applies: "next native completion" };
    } });
    return () => lifecycle.abort();
  }, []);

  async function connectModel() {
    if (operationLock.current || !modelName) return;
    operationLock.current = true; setConnecting(true); setConnectionError("");
    addLog("INFO", `Connecting ${modelName} through ${engine === "native" ? "native llama.cpp (CPU)" : "Ollama"}…`);
    try {
      const state = await api<Connection>("/api/connect", { model: modelName, engine });
      restoreConnection(state); setMessages([WELCOME]); setMetrics({}); setSamples([]); setExtraction(null);
      addLog("OK", `Connected · ${state.layers ?? "unknown"} layers · ${state.dimensions ?? "unknown"} hidden dimensions · ${state.engine}.`);
      setNotice(`Connected to ${state.model}.`);
    } catch (error) { setConnectionError((error as Error).message); addLog("WARN", (error as Error).message); const state = await api<Connection>("/api/status").catch(() => EMPTY); restoreConnection(state); }
    finally { operationLock.current = false; setConnecting(false); }
  }

  function applyPreset(preset: typeof PRESETS[number]) {
    setVectorName(preset.name); setPositive(preset.positive); setNegative(preset.negative); setFormError(""); setExtraction(null); nameRef.current?.focus();
  }

  async function extractVector(event: FormEvent) {
    event.preventDefault();
    if (operationLock.current) return;
    if (!nativeReady) { setFormError("Connect a model with Native steering to extract real vectors."); return; }
    if (!vectorName.trim() || !positive.trim() || !negative.trim()) { setFormError("Add a name and both contrastive prompts."); return; }
    if (positive.trim().toLowerCase() === negative.trim().toLowerCase()) { setFormError("Use different positive and neutral prompts."); return; }
    operationLock.current = true; setExtracting(true); setFormError(""); setExtraction(null);
    addLog("INFO", `Evaluating contrastive prompts for '${vectorName.trim()}' on ${connection.model}.`);
    try {
      const result = await api<StoredVector>("/api/extract", { name: vectorName.trim(), positive: positive.trim(), negative: negative.trim(), mode });
      setVectors(prev => [...prev, { ...result, color: PRESETS.find(p => p.name === result.name)?.color ?? COLORS[prev.length % COLORS.length], enabled: true, value: 0 }]);
      setExtraction(result); setSelection(result.id); addLog("OK", `Extracted '${result.name}' · contrast L2 ${result.difference_norm.toFixed(2)} · ${vectorSummary(result)}.`);
      setNotice(`${result.name} extracted and saved for this model.`);
    } catch (error) { setFormError((error as Error).message); addLog("WARN", (error as Error).message); }
    finally { operationLock.current = false; setExtracting(false); }
  }

  async function deleteVector(vector: ConceptVector) {
    if (operationLock.current) return;
    operationLock.current = true; setDeleting(true); setDeleteError("");
    try {
      await api(`/api/vectors/${encodeURIComponent(vector.id)}`, undefined, "DELETE");
      setVectors(prev => prev.filter(v => v.id !== vector.id));
      setConnection(prev => ({ ...prev, vectors: prev.vectors.filter(v => v.id !== vector.id) }));
      setSelection(current => current === vector.id ? vectors.find(v => v.id !== vector.id)?.id ?? "" : current);
      setExtraction(current => current?.id === vector.id ? null : current);
      setVectorToDelete(null);
      addLog("INFO", `Deleted '${vector.name}' from this model's vector library.`);
      setNotice(`${vector.name} deleted from the vector library.`);
    } catch (error) { setDeleteError((error as Error).message); addLog("WARN", (error as Error).message); }
    finally { operationLock.current = false; setDeleting(false); }
  }

  async function stopStream() {
    const stopping = api("/api/stop", {}).catch(() => {});
    abortRef.current?.abort();
    await stopping;
    addLog("INFO", "Stop requested. Partial response retained.");
  }

  async function sendMessage(text = input) {
    const prompt = text.trim();
    if (!prompt || operationLock.current || !connection.connected) return;
    operationLock.current = true; setStreaming(true); setInput(""); stickToBottom.current = true;
    const controller = new AbortController(); abortRef.current = controller;
    const snapshot = applied.map(v => ({ ...v }));
    const coefficients = snapshot.map(v => ({ name: v.name, value: v.value, color: v.color }));
    const id = uid();
    const history = messages.filter(m => m.id !== "welcome" && m.text && !m.failed).map(m => ({ role: m.role, content: m.text }));
    setMessages(prev => [...prev, { id: uid(), role: "user", text: prompt }, { id, role: "assistant", text: "", tone: connection.model ?? "Local model", coefficients, streaming: true }]);
    setMetrics({}); setSamples([]);
    addLog("INFO", `Sending to ${connection.model} · ${snapshot.length} nonzero control vectors.`);
    let finished = false;
    let completedNormally = false;
    try {
      await streamChat({ model: connection.model, messages: [...history, { role: "user", content: prompt }], coefficients: snapshot.map(v => ({ id: v.id, value: v.value })) }, controller.signal, event => {
        if (event.type === "token") setMessages(prev => prev.map(m => m.id === id ? { ...m, text: m.text + (event.text ?? "") } : m));
        if ((event.type === "metrics" || event.type === "done") && !pausedRef.current) { setMetrics(event); if (Object.keys(event.alignments ?? {}).length) setSamples(prev => [...prev, { alignments: event.alignments ?? {} }].slice(-45)); }
        if (event.type === "done") { finished = true; completedNormally = !event.cancelled; if (event.cancelled) setMessages(prev => prev.map(m => m.id === id ? { ...m, stopped: true } : m)); addLog("OK", `${event.cancelled ? "Stopped" : "Completed"} · ${event.tokens ?? 0} generated tokens · ${event.tokens_per_second?.toFixed(1) ?? "—"} tokens/s.`); }
      });
      if (completedNormally && !controller.signal.aborted && workspacePreferences.completionNotifications) {
        void window.vectorLab?.showNotification("Vector Lab", "Your model response is ready.").catch(() => {});
      }
    } catch (error) {
      const stopped = controller.signal.aborted;
      setMessages(prev => prev.map(m => m.id === id ? { ...m, stopped, failed: !stopped, text: m.text || (stopped ? "Generation stopped before the first token." : (error as Error).message) } : m));
      if (!stopped) addLog("WARN", (error as Error).message);
    } finally {
      setMessages(prev => prev.map(m => m.id === id ? { ...m, streaming: false, stopped: m.stopped || (controller.signal.aborted && !finished) } : m));
      operationLock.current = false; abortRef.current = null; setStreaming(false);
    }
  }

  async function copyMessage(message: Message) {
    try { await navigator.clipboard.writeText(message.text); setCopied(message.id); setNotice("Response copied."); } catch { setNotice("Select the response text to copy it."); }
  }

  async function importVectors() {
    if (operationLock.current) return;
    operationLock.current = true; setImporting(true);
    try {
      const imported = await importVectorFolder();
      if (imported) {
        const state = await api<Connection>("/api/status");
        restoreConnection(state);
      }
      setNotice(imported ? `Imported ${imported} vectors. Connect their original model to view them.` : "No new vectors imported.");
    } catch (error) { setNotice((error as Error).message); }
    finally { operationLock.current = false; setImporting(false); }
  }

  function updateWorkspacePreferences(preferences: WorkspacePreferences) {
    setWorkspacePreferences(preferences);
    try { localStorage.setItem(WORKSPACE_PREFERENCES_KEY, JSON.stringify(preferences)); }
    catch { setNotice("Workspace settings applied for this session. Browser storage is unavailable."); }
  }

  return <div className="app-shell" data-ready={ready}>
    <header className="topbar">
    <Link className="brand" href="/" aria-label="Vector Lab home">
      <div className="brand-icon"><Box size={23} strokeWidth={1.7}/></div>
      <span>vector<span className="brand-light">lab</span><span className="version">LOCAL</span></span>
    </Link>
    <div className="top-breadcrumb"><span>Research workspace</span><ChevronRight size={13}/>
    <span>Activation steering</span></div>
    <div className="top-right">
      <span className="simulation-badge">
        <Cpu size={13}/>{connection.connected ? "Real model connected" : "Local model workspace"}
      </span>
    </div>
    </header>
    <div className={`workspace-grid ${workspacePreferences.showTelemetry ? "" : "telemetry-hidden"}`}>
      <aside className="left-sidebar">
        <section className="library-section"><SectionTitle icon={Layers3} extra={<span className="count-badge">{vectors.length}</span>}>Vector library</SectionTitle><p className="section-description">Extracted for the connected model.</p><div className="vector-library" ref={libraryRef}>
          {vectors.map((v, index) => { const Icon = ICONS[index % ICONS.length]; return <div key={v.id} className={`library-vector ${selection === v.id ? "selected" : ""}`} style={{ "--vector-color": v.color } as CSSProperties}><button className="vector-select" onClick={() => setSelection(v.id)} aria-pressed={selection === v.id}><span className="vector-icon"><Icon size={16}/></span><span className="vector-info"><strong>{v.name}</strong><small>{vectorSummary(v)}</small></span></button><Switch aria-label={`Enable ${v.name}`} checked={v.enabled} disabled={!nativeReady} onCheckedChange={enabled => { setVectors(prev => prev.map(item => item.id === v.id ? { ...item, enabled } : item)); addLog("INFO", `${enabled ? "Enabled" : "Disabled"} '${v.name}' for the next completion.`); }} className="vector-switch"/><button type="button" className="icon-button vector-delete" disabled={busy} aria-label={`Delete ${v.name}`} title={`Delete ${v.name}`} onClick={event => { deleteTriggerRef.current = event.currentTarget; setDeleteError(""); setVectorToDelete(v); }}><Trash2 size={15}/></button></div>; })}
          {!vectors.length && <p className="library-empty">No extracted vectors yet. Start with a concept pair below.</p>}
          {ready && window.vectorLab && <button className="add-vector-button" disabled={busy} onClick={() => void importVectors()}>{importing ? "Importing vectors..." : "Import existing vector folder"}</button>}
        </div><div className="preset-grid">{PRESETS.map(p => <button key={p.name} disabled={busy} onClick={() => applyPreset(p)} style={{ color: p.color }} title={`Use the ${p.name} contrastive pair`}><Plus size={12}/>{p.name}</button>)}</div><button className="add-vector-button" disabled={busy} onClick={() => { setVectorName(""); setExtraction(null); setFormError(""); nameRef.current?.focus(); }}><Plus size={15}/> Create custom vector</button>
          {selected && <div className="selected-detail"><span>Contrast L2 {selected.difference_norm.toFixed(2)} · {injectionDescription(selected)}</span></div>}
        </section>
        <section className="extraction-section"><SectionTitle icon={FlaskConical} extra={<span className="tiny-tag">RAW DIFF</span>}>Extract a vector</SectionTitle><p className="section-description">One sample per line. The raw activation contrast is measured on the model.</p><form onSubmit={extractVector} className="extraction-form">
          <label htmlFor="vector-name">Vector name</label><input ref={nameRef} id="vector-name" value={vectorName} onChange={e => { setVectorName(e.target.value); setFormError(""); }} maxLength={32} disabled={!ready || busy} placeholder="e.g. Curiosity" autoComplete="off"/>
          <label htmlFor="positive-prompt" className="prompt-label"><span className="prompt-dot positive-dot"/>Positive prompt <span>+</span></label><textarea id="positive-prompt" value={positive} onChange={e => setPositive(e.target.value)} maxLength={4000} disabled={!ready || busy} rows={3}/>
          <label htmlFor="negative-prompt" className="prompt-label"><span className="prompt-dot neutral-dot"/>Neutral / negative prompt <span>−</span></label><textarea id="negative-prompt" value={negative} onChange={e => setNegative(e.target.value)} maxLength={4000} disabled={!ready || busy} rows={2}/>
          <label htmlFor="pool-mode" className="prompt-label">Activation pooling <span>·</span> injection layer auto-selected</label><Select value={mode} onValueChange={v => setMode(v as PoolingMode)} disabled={!ready || busy}><SelectTrigger id="pool-mode" aria-label="Activation pooling mode"><SelectValue/></SelectTrigger><SelectContent position="popper"><SelectItem value="mean">Mean over tokens</SelectItem><SelectItem value="final_token">Final token</SelectItem></SelectContent></Select>
          <div className="extraction-method"><span>Residual layers</span><code>{connection.layers ? `1 – ${connection.layers - 1}` : "Connect model"}<Layers3 size={11}/></code></div><button className="primary-button extract-button" disabled={!ready || busy || !nativeReady} type="submit">{extracting ? <LoaderCircle size={15} className="spin"/> : <Plus size={16}/>} {extracting ? "Measuring activations…" : "Extract vector"}</button>
          {extracting && <div className="native-progress" role="status"><span/>Evaluating both prompts on the model…</div>}{formError && <p className="form-error" role="alert"><TriangleAlert size={14}/>{formError}</p>}{extraction && <div className="extraction-success" role="status"><CheckCheck size={15}/><p><strong>{extraction.name} extracted</strong><span>{injectionDescription(extraction)} · contrast L2 {extraction.difference_norm.toFixed(2)}</span></p></div>}
        </form><div className="extraction-note"><CircleHelp size={13}/><p>Raw positive − baseline contrast, denoised against the baseline spread. The extracted direction is injected at one layer, selected using independent neutral probes. AUC is measured on the training prompts; it does not validate concept quality. Legacy vectors retain their original unit directions across layers.</p></div></section><div className="sidebar-bottom"><span className="small-logo"><Box size={15}/> Representation engineering</span><span>Vectors stay on your computer.</span></div>
      </aside>

      <main className="main-panel"><div className="workspace-heading"><div><div className="eyebrow">THE STEERING WORKSPACE</div><hr></hr></div><WorkspaceSettings preferences={workspacePreferences} onChange={updateWorkspacePreferences}/></div>
        <section className="connection-card" aria-label="Model connection"><div className="connection-title"><span><PlugZap size={16}/> Model connection</span><button className="icon-button" disabled={busy} onClick={refresh} aria-label="Refresh local models"><RotateCcw size={14}/></button></div><div className="connection-fields"><Select value={modelName} onValueChange={setModelName} disabled={busy || !models.length}><SelectTrigger aria-label="Local model" className="model-picker"><SelectValue placeholder="No models found"/></SelectTrigger><SelectContent position="popper">{models.map(m => <SelectItem key={m.name} value={m.name}>{m.name} · {(m.size / 1e9).toFixed(1)} GB</SelectItem>)}</SelectContent></Select><Select value={engine} onValueChange={v => setEngine(v as Engine)} disabled={busy}><SelectTrigger aria-label="Model runtime"><SelectValue/></SelectTrigger><SelectContent><SelectItem value="native">Native steering · CPU</SelectItem><SelectItem value="ollama">Ollama chat</SelectItem></SelectContent></Select><button className="primary-button connect-button" disabled={busy || !models.length} onClick={connectModel}>{connecting ? <LoaderCircle size={15} className="spin"/> : <PlugZap size={15}/>} {connecting ? "Loading…" : "Connect"}</button></div><p>{connection.connected ? `${connection.model} · ${connection.engine === "native" ? "llama.cpp · live tensor access" : "Ollama · activation injection unavailable"}` : "Start the local bridge, then connect an installed model."}</p>{connectionError && <div className="form-error" role="alert"><TriangleAlert size={14}/>{connectionError}</div>}</section>
        <section className="control-card"><div className="card-heading"><div className="flex items-center gap-2.5"><SlidersHorizontal size={17} className="purple-text"/><h2>Active steering</h2><span className="count-badge">{active.length}</span></div><button className="text-button" onClick={() => { setVectors(prev => prev.map(v => ({ ...v, value: 0 }))); addLog("INFO", "All coefficients staged at zero for the next completion."); }}><RotateCcw size={13}/> Reset</button></div><div className="injection-bar"><span><span className={`status-dot ${injection && nativeReady ? "purple-dot" : "muted-dot"}`}/>{!nativeReady ? "Native model required for injection" : injection ? "Injection enabled · next completion" : "Injection bypassed"}</span><Switch checked={injection && nativeReady} disabled={!nativeReady} onCheckedChange={setInjection} aria-label="Enable vector injection"/></div>
          <div className={`sliders ${!injection || !nativeReady ? "bypassed" : ""}`}>{!active.length && <div className="empty-steering"><Layers3 size={23}/><p>Extract your first direction</p><span>Choose a concept pair, then measure it on your model.</span></div>}{active.map((v, index) => { const Icon = ICONS[index % ICONS.length]; return <div className="steering-row" key={v.id} style={{ "--vector-color": v.color } as CSSProperties}><div className="steering-label"><div><Icon size={15}/><label id={`label-${v.id}`}>{v.name}</label><span className="steering-tone">{v.value === 0 ? "Neutral" : v.value < 0 ? "Inverted" : v.value >= 3 ? "High gain" : "Positive"}</span></div><output className="coefficient" aria-label={`${v.name} coefficient`}>λ <strong>{signed(v.value)}</strong></output></div><div className="slider-container"><Slider value={[v.value]} min={-5} max={5} step={0.1} disabled={!injection || !nativeReady} aria-labelledby={`label-${v.id}`} onValueChange={values => setVectors(prev => prev.map(item => item.id === v.id ? { ...item, value: values[0] } : item))} onValueCommit={values => addLog(Math.abs(values[0]) >= 3 ? "WARN" : "INFO", `Staged '${v.name}' at λ = ${signed(values[0])} · ${injectionDescription(v)}.`)} className="concept-slider"/><span className="slider-center"/></div><div className="slider-scale"><span>−5.0</span><span>0.0</span><span>+5.0</span></div></div>; })}</div>
          {extreme && <div className="steering-warning" role="status"><TriangleAlert size={15}/><span><strong>High injection strength.</strong> Large coefficients may degrade coherence. Effects depend on the model and direction.</span></div>}<div className="control-footer"><span><Layers3 size={12}/> Additive residual injection</span><code>h′ = h + Σ λᵢvᵢ</code></div>
        </section>
        <section className="chat-card"><div className="card-heading"><div className="flex items-center gap-2.5"><MessageSquare size={17}/><h2>Live playground</h2></div><button className="icon-button" aria-label="Clear chat" disabled={streaming || messages.length === 1} onClick={() => { setMessages([WELCOME]); setMetrics({}); setSamples([]); addLog("INFO", "Chat history cleared."); }}><Trash2 size={15}/></button></div><div className="model-strip"><span className="model-strip-name"><Box size={13}/>{connection.model ?? "No model connected"}</span><span className="response-mode"><span className="status-dot purple-dot"/>{modeLabel}</span></div>
          <div className="chat-history" ref={scrollRef} role="log" aria-label="Chat history" aria-live="off" onScroll={() => { const el = scrollRef.current; if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 64; }}>{messages.map(message => <article key={message.id} className={`chat-message ${message.role}`}><div className={`message-avatar ${message.role === "assistant" ? "assistant-avatar" : "user-avatar"}`}>{message.role === "assistant" ? <Box size={17}/> : "Y"}</div><div className="message-body"><div className="message-title"><strong>{message.role === "assistant" ? "Model response" : "You"}</strong><span>{message.role === "assistant" ? message.tone : "Prompt"}</span>{message.role === "assistant" && message.id !== "welcome" && !message.streaming && <button className="copy-button" onClick={() => copyMessage(message)} aria-label="Copy response">{copied === message.id ? <CheckCheck size={13}/> : <Copy size={13}/>}</button>}</div><p className={`message-text ${message.failed ? "failed-message" : ""}`}>{message.text}{message.streaming && <span className="stream-caret"/>}</p>{!!message.coefficients?.length && <div className="message-coefficients">{message.coefficients.map((c, i) => <span key={i} style={{ color: c.color }}>{c.name} {signed(c.value)}</span>)}</div>}{message.stopped && <span className="stopped-label">Generation stopped</span>}</div></article>)}{workspacePreferences.showSuggestions && messages.length === 1 && <div className="suggested-prompts"><span>TRY A PROMPT</span>{["Explain how neural networks learn", "Help me plan a creative project", "I finally finished my first app!"].map(prompt => <button key={prompt} disabled={busy || !connection.connected} onClick={() => sendMessage(prompt)}>{prompt}<Plus size={12}/></button>)}</div>}</div>
          <div className="composer-wrap"><form className="chat-composer" onSubmit={e => { e.preventDefault(); void sendMessage(); }}><textarea aria-label="Message the model" value={input} disabled={!connection.connected} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void sendMessage(); } }} maxLength={4000} rows={1} placeholder={connection.connected ? "Send a message. See the shift." : "Connect a model to begin."}/><div className="composer-actions"><span>{streaming ? "Generating…" : "↵ to send"}</span>{streaming ? <button type="button" className="send-button stop-button" aria-label="Stop generation" onClick={stopStream}><Square size={14} fill="currentColor"/></button> : <button className="send-button" type="submit" aria-label="Send message" disabled={!input.trim() || busy || !connection.connected}><Send size={16}/></button>}</div></form><div className="composer-note"><span><Cpu size={11}/> Real inference · coefficients apply when a completion starts</span></div></div><span role="status" className="sr-only">{streaming ? "Generating a real model response" : "Ready"}</span>
        </section><footer className="workspace-footer"><span><span className={`status-dot ${connection.connected ? "green-dot" : "muted-dot"}`}/>{connection.connected ? "Local model connected" : "Awaiting connection"}</span><span>VECTOR LAB <span className="muted">/</span> v1.0.0</span></footer>
      </main>

      {workspacePreferences.showTelemetry && <aside className="right-sidebar"><div className="telemetry-heading"><SectionTitle icon={Activity} extra={<button className={`live-button ${paused ? "is-paused" : ""}`} onClick={() => setPaused(v => !v)} aria-label={paused ? "Resume live telemetry" : "Pause live telemetry"}>{paused ? <Play size={10}/> : <Pause size={10}/>} {paused ? "PAUSED" : "LIVE"}</button>}>Internal state</SectionTitle><p className="section-description">Measured residuals from the latest run.</p></div>
        <section className="metric-section"><div className="metric-header"><h3>Cosine alignment</h3><span title="Mean cosine similarity between measured residual activations and each applied vector across layers."><CircleHelp size={13}/></span></div><div className="similarity-stat"><strong>{cosine === null ? "—" : cosine.toFixed(3)}</strong><span>mean alignment</span><span className="sim-pill">REAL</span></div><ActivationChart vectors={vectors} samples={samples}/><div className="chart-legend">{active.slice(0, 5).map(v => <span key={v.id}><i style={{ background: v.color }}/>{v.name}</span>)}</div>{connection.engine === "ollama" && <p className="measurement-note">Ollama does not expose hidden activations.</p>}</section>
        <section className="metric-section layers-section"><div className="metric-header"><h3>Residual L2 norms</h3><span className="mono muted">{connection.layers ?? "—"} layers</span></div><div className="layer-map" aria-label="Measured residual magnitude by layer" role="img">{norms.length ? norms.map((value, i) => <div className="layer-bar" key={i} style={{ height: `${value === null ? 0 : Math.max(2, value / maxNorm * 100)}%`, background: "#9c7ed7" }} title={`Layer ${i}: ${value?.toFixed(3) ?? "unavailable"} L2`}/>) : <span className="measurement-note">No activation samples yet</span>}</div><div className="layer-axis"><span>L0</span><span>L{connection.layers ? Math.floor(connection.layers / 2) : "—"}</span><span>L{connection.layers ? connection.layers - 1 : "—"}</span></div><div className="layer-key"><span>Relative magnitude</span><i/><span>High</span></div></section>
        <section className="tensor-stats"><div><span>Hidden dimensions</span><code>{connection.dimensions?.toLocaleString() ?? "—"}</code></div><div><span>Injection L2 · last run</span><code>{metrics.injection_norm?.toFixed(3) ?? "—"}</code></div><div><span>Generated tokens</span><code>{metrics.tokens ?? "—"}</code></div><div><span>Generation speed</span><code>{metrics.tokens_per_second?.toFixed(1) ?? "—"}<span className="unit"> t/s</span></code></div></section>
        <section className="terminal-section"><div className="terminal-heading"><span><Terminal size={14}/> Event stream</span><span className="terminal-count">{logs.length} events</span></div><div className="terminal-content" role="log" aria-label="Model event log" aria-live="off">{[...logs].reverse().map(log => <div className="log-line" key={log.id}><div><time>{log.time}</time><span className={`log-level log-${log.level.toLowerCase()}`}>[{log.level}]</span></div><p>{log.message}</p></div>)}</div><div className="terminal-footer"><span className={`status-dot ${connection.connected ? "green-dot" : "muted-dot"}`}/>{streaming ? "Receiving model output" : "Waiting for the next operation"}<span className="terminal-cursor">_</span></div></section><div className="telemetry-note"><Cpu size={13}/><span>{connection.engine === "native" ? "llama.cpp b11146 · CPU tensor capture" : "Local runtime · no fabricated metrics"}</span></div>
      </aside>}
    </div>{notice && <div className="toast" role="status"><Check size={16}/><span>{notice}</span><button className="icon-button" aria-label="Dismiss notification" onClick={() => setNotice("")}><X size={14}/></button></div>}
    <AlertDialog open={vectorToDelete !== null} onOpenChange={open => { if (!open && !deleting) setVectorToDelete(null); }}>
      <AlertDialogContent className="delete-vector-dialog" onEscapeKeyDown={event => { if (deleting) event.preventDefault(); }} onCloseAutoFocus={event => {
        event.preventDefault();
        const target = deleteTriggerRef.current?.isConnected ? deleteTriggerRef.current : libraryRef.current?.querySelector<HTMLButtonElement>(".vector-select") ?? nameRef.current;
        target?.focus();
      }}>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete vector?</AlertDialogTitle>
          <AlertDialogDescription>Delete “{vectorToDelete?.name}” from the vector library and active steering? This permanently removes the saved vector and cannot be undone.</AlertDialogDescription>
        </AlertDialogHeader>
        {deleteError && <p className="form-error" role="alert"><TriangleAlert size={14}/>{deleteError}</p>}
        <AlertDialogFooter>
          <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" disabled={busy} onClick={event => { event.preventDefault(); if (vectorToDelete) void deleteVector(vectorToDelete); }}>
            {deleting ? <><LoaderCircle size={15} className="spin"/>Deleting…</> : "Delete vector"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  </div>;
}
