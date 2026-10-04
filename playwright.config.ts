import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:5173", browserName: "chromium", channel: process.platform === "win32" ? "msedge" : undefined, headless: true, viewport: { width: 1440, height: 1100 } },
});
