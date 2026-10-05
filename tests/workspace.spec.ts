import { test, expect } from "@playwright/test";

const defaultBridge = "http://127.0.0.1:8788";
const bridge = process.env.VECTOR_LAB_TEST_BRIDGE ?? defaultBridge;
const headers = { "X-Vector-Lab": "1" };

test.beforeEach(async ({ page }) => {
  if (bridge !== defaultBridge) {
    await page.route(`${defaultBridge}/**`, route => route.continue({
      url: route.request().url().replace(defaultBridge, bridge),
    }));
  }
});

test.beforeAll(async ({ request }) => {
  const result = await request.post(`${bridge}/api/connect`, { headers, data: { model: "qwen2.5-coder:1.5b", engine: "native" } });
  expect(result.ok()).toBeTruthy();
});

test("real extraction, sliders, streamed inference, measured tensors, deletion", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator(".app-shell")).toHaveAttribute("data-ready", "true");
  await expect(page.getByText("Real model connected", { exact: true })).toBeVisible();
  const name = `Browser_test_${Date.now()}`;
  await page.getByLabel("Vector name", { exact: true }).fill(name);
  await page.getByLabel("Neutral / negative prompt").fill(await page.getByLabel("Positive prompt").inputValue());
  await page.getByRole("button", { name: "Extract vector", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("different");
  await page.getByLabel("Neutral / negative prompt").fill("Task completed.");
  await page.getByRole("button", { name: "Extract vector", exact: true }).click();
  await expect(page.getByText(`${name} extracted`, { exact: true })).toBeVisible();
  await expect(page.getByRole("switch", { name: `Enable ${name}`, exact: true })).toBeChecked();
  const slider = page.getByRole("slider", { name, exact: true });
  await slider.focus(); await slider.press("End");
  await expect(page.getByLabel(`${name} coefficient`, { exact: true })).toContainText("+5.0");
  await expect(page.getByText("High injection strength.", { exact: true })).toBeVisible();
  await slider.press("Home");
  await expect(page.getByLabel(`${name} coefficient`, { exact: true })).toContainText("-5.0");
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await slider.focus(); await slider.press("ArrowRight");
  await page.getByLabel("Message the model", { exact: true }).fill("What is 2 + 2? Answer briefly.");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("log", { name: "Chat history" })).toContainText("4");
  await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeVisible();
  await expect(page.getByRole("log", { name: "Chat history" })).toContainText(`${name} +0.1`);
  expect(await page.locator(".layer-bar").count()).toBe(28);
  await expect(page.locator(".tensor-stats")).toContainText("1,536");
  await page.getByRole("button", { name: `Delete ${name}`, exact: true }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete vector", exact: true }).click();
  await expect(page.getByRole("switch", { name: `Enable ${name}`, exact: true })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("backend rejects invalid vectors and unauthorized browser origins", async ({ request }) => {
  const bad = await request.post(`${bridge}/api/chat`, { headers, data: { model: "qwen2.5-coder:1.5b", messages: [{ role: "user", content: "Hi" }], coefficients: [{ id: "not-a-real-vector", value: 1 }] } });
  expect(bad.status()).toBe(422);
  const origin = await request.post(`${bridge}/api/stop`, { headers: { ...headers, Origin: "https://unrelated.example" }, data: {} });
  expect(origin.status()).toBe(403);
});

test("real streaming cancellation, coefficient snapshot, and injection bypass", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByText("Real model connected", { exact: true })).toBeVisible();
  const slider = page.getByRole("slider", { name: "Pleasure", exact: true });
  await slider.focus(); await slider.press("End");
  await page.getByLabel("Message the model", { exact: true }).fill("Write a long, detailed explanation of how neural networks learn, with many examples.");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop generation", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(page.getByRole("log", { name: "Chat history" })).toContainText("Pleasure +5.0");
  await page.getByRole("button", { name: "Stop generation", exact: true }).click();
  await expect(page.getByText("Generation stopped", { exact: true })).toBeVisible();
  await expect.poll(async () => (await (await request.get(`${bridge}/api/status`)).json()).busy).toBe(false);
  await page.getByRole("button", { name: "Clear chat", exact: true }).click();
  await page.getByRole("switch", { name: "Enable vector injection", exact: true }).click();
  await expect(slider).toBeDisabled();
  await page.getByLabel("Message the model", { exact: true }).fill("What is 2 + 2? Answer briefly.");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("log", { name: "Chat history" })).toContainText("4");
  await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeVisible();
  await expect(page.locator(".tensor-stats")).toContainText("0.000");
});

test("responsive real-model workspace and screenshots", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Real model connected", { exact: true })).toBeVisible();
  for (const width of [1440, 1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.screenshot({ path: "outputs/vector-lab-real-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "outputs/vector-lab-real-mobile.png", fullPage: true });
});
