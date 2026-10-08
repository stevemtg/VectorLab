import { _electron, test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

test("desktop renderer uses the real sandboxed preload, authenticated CORS, and persistent preferences", async () => {
  const userData = path.resolve("outputs", `electron-test-${randomUUID()}`);
  fs.mkdirSync(userData, { recursive: true });
  const executablePath = process.env.VECTOR_LAB_TEST_PACKAGED_EXE;
  const electronEnv: Record<string, string> = Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined));
  electronEnv.VECTOR_LAB_DEV_USER_DATA = userData;
  delete electronEnv.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ ...(executablePath ? { executablePath, args: [`--user-data-dir=${userData}`] } : { args: [".", `--user-data-dir=${userData}`] }),
    cwd: process.cwd(), env: electronEnv,
  });
  try {
    const page = await app.firstWindow();
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await expect(page.locator(".app-shell")).toHaveAttribute("data-ready", "true");
    expect(await page.evaluate(() => typeof (window as Window & { require?: unknown }).require)).toBe("undefined");
    expect(await page.evaluate(() => typeof (window as Window & { process?: unknown }).process)).toBe("undefined");
    const status = await page.evaluate(async () => {
      const { url, token } = await window.vectorLab!.getServiceConnection();
      const result = await fetch(`${url}/api/status`, { headers: { Authorization: `Bearer ${token}`, "X-Vector-Lab": "1" } });
      const body = await result.json() as { connected: boolean };
      return { code: result.status, connected: body.connected };
    });
    expect(status).toEqual({ code: 200, connected: false });
    await expect(page.getByRole("button", { name: "Import existing vector folder" })).toBeVisible();
    await page.getByRole("button", { name: "Workspace settings" }).click();
    await page.getByRole("switch", { name: "Completion notifications" }).click();
    await page.getByRole("button", { name: "Done", exact: true }).click();
    await page.reload();
    await page.getByRole("button", { name: "Workspace settings" }).click();
    await expect(page.getByRole("switch", { name: "Completion notifications" })).toBeChecked();
    expect(errors).toEqual([]);
  } finally {
    await app.close();
    // Verified generated target under outputs; no user profile data is deleted.
    if (path.dirname(userData) !== path.resolve("outputs")) throw new Error("Invalid smoke-test data directory.");
    fs.rmSync(userData, { recursive: true, force: true });
  }
});
