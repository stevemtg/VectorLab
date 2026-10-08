export const APP_ORIGIN = "vectorlab://bundle";

// Node's URL.origin is "null" for custom schemes: compare parsed components.
export function isAppUrl(raw) {
  try {
    const url = new URL(raw);
    return url.protocol === "vectorlab:" && url.host === "bundle" && !url.username && !url.password
      && ["/", "/index.html"].includes(url.pathname) && !url.search;
  } catch { return false; }
}

export function assertSender(event, window) {
  if (!window || window.isDestroyed() || event.sender !== window.webContents
    || event.senderFrame !== window.webContents.mainFrame || !isAppUrl(event.senderFrame.url)) {
    throw new Error("Untrusted IPC sender.");
  }
}

export function validateNotification(payload) {
  if (!payload || typeof payload.title !== "string" || typeof payload.body !== "string"
    || !payload.title.trim() || payload.title.length > 120 || payload.body.length > 500) throw new Error("Invalid notification.");
  return { title: payload.title, body: payload.body, silent: true };
}

// The workspace has no need to open arbitrary Internet or loopback URLs.
export function externalUrl(raw) {
  try {
    const url = new URL(raw);
    if (url.protocol === "https:" && !url.username && !url.password && !url.port
      && ["github.com", "docs.ollama.com", "ollama.com"].includes(url.hostname)) return url.href;
  } catch { /* Deny malformed URLs and other protocol handlers. */ }
  return null;
}

export function rendererCsp(bridgeUrl) {
  return `default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src ${bridgeUrl}; object-src 'none'; base-uri 'none'; form-action 'none'; frame-src 'none'; frame-ancestors 'none'`;
}
