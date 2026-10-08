// Typed bridge between the desktop (Electron) renderer and the main process.
// In the web build window.vectorLab is undefined and callers fall back to the
// fixed development bridge address; in the desktop app the main process owns
// the bridge child process and reports its real loopback port through preload.
export type ServiceStatus = { state: "stopped" | "starting" | "running" | "stopping" | "failed"; detail: string };
export type ServiceConnection = { url: string; token?: string };
export type DesktopBridge = {
  readonly desktop: true;
  readonly platform: string;
  getServiceConnection(): Promise<ServiceConnection>;
  getServiceStatus(): Promise<ServiceStatus>;
  onServiceStatus(callback: (status: ServiceStatus) => void): () => void;
  showNotification(title: string, body: string): Promise<boolean>;
  importVectors(): Promise<number>;
};

declare global {
  interface Window {
    vectorLab?: DesktopBridge;
  }
}

export const isDesktop = typeof window !== "undefined" && window.vectorLab?.desktop === true;

let resolved: Promise<ServiceConnection> | undefined;

// Cache successful desktop discovery only; a rejected IPC request must be retryable.
// Web/SSR calls never populate the desktop cache.
export function serviceConnection(defaultUrl: string): Promise<ServiceConnection> {
  const bridge = typeof window !== "undefined" ? window.vectorLab : undefined;
  if (!bridge) return Promise.resolve({ url: defaultUrl });
  resolved ??= bridge.getServiceConnection().then(connection => {
    const url = new URL(connection.url);
    if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || !url.port || url.username || url.password
      || url.pathname !== "/" || url.search || url.hash || !/^[0-9a-f]{64}$/.test(connection.token ?? "")) throw new Error("The desktop bridge reported an invalid connection.");
    return { url: url.origin, token: connection.token };
  }).catch(error => { resolved = undefined; throw error; });
  return resolved;
}
