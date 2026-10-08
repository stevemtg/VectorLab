import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { rendererCsp } from "./security.mjs";

const MIME = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml",
  ".png": "image/png", ".ico": "image/x-icon", ".woff2": "font/woff2" };

export function rendererHandler(rendererDir, bridgeUrl) {
  const root = fs.realpathSync(rendererDir);
  return request => {
    try {
      const url = new URL(request.url);
      if (request.method !== "GET" || url.protocol !== "vectorlab:" || url.host !== "bundle") return new Response("Bad request", { status: 400 });
      const pathname = decodeURIComponent(url.pathname);
      const file = fs.realpathSync(path.resolve(root, `.${pathname === "/" ? "/index.html" : pathname}`));
      const relative = path.relative(root, file);
      if (relative.startsWith("..") || path.isAbsolute(relative) || !fs.statSync(file).isFile()) return new Response("Not found", { status: 404 });
      // No SPA fallback: a missing script must never be returned as HTML.
      return new Response(Readable.toWeb(fs.createReadStream(file)), { headers: {
        "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream",
        "Content-Security-Policy": rendererCsp(bridgeUrl), "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
      } });
    } catch { return new Response("Not found", { status: 404 }); }
  };
}
