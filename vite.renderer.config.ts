import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/postcss";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)));

// Standalone static build of the existing workspace page for the Electron
// renderer. The vinext/Workers pipeline in vite.config.ts is untouched and
// still powers the web workflow; this config only bundles app/page.tsx and the
// shared components/lib for the desktop window.
export default defineConfig({
  root: path.join(root, "renderer"),
  publicDir: path.join(root, "public"),
  base: "./",
  resolve: {
    alias: {
      "@": root,
      "next/link": path.join(root, "renderer", "next-link.tsx"),
    },
  },
  css: { postcss: { plugins: [tailwindcss()] } },
  plugins: [react()],
  build: { outDir: path.join(root, "build", "renderer"), emptyOutDir: true },
});
