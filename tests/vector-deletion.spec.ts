import { test, expect, type Page } from "@playwright/test";
import type { StoredVector } from "../lib/model-api";

const bridge = "http://127.0.0.1:8788";
const custom: StoredVector = {
  id: "custom", name: "Custom curiosity", positive: "Explore new ideas", negative: "Record the facts",
  model_digest: "test-digest", model: "test-model", layers: 4, dimensions: 2,
  difference_norm: 3, layer_norms: [3, 4, 5, 6], method: "raw difference", best_layer: 3,
};
const other: StoredVector = { ...custom, id: "other", name: "Saved direction", difference_norm: 7 };

async function mockBridge(page: Page, initialVectors = [custom, other]) {
  const state = { vectors: [...initialVectors], deleted: [] as string[] };
  await page.route(`${bridge}/api/models`, route => route.fulfill({ json: {
    models: [{ name: "test-model", size: 1000, family: "test", parameter_size: "2", digest: "test-digest" }],
    default_model: "test-model",
  } }));
  await page.route(`${bridge}/api/status`, route => route.fulfill({ json: {
    connected: true, engine: "native", model: "test-model", digest: "test-digest",
    layers: 5, dimensions: 2, vectors: state.vectors,
  } }));
  await page.route(`${bridge}/api/vectors/*`, route => {
    expect(route.request().method()).toBe("DELETE");
    expect(route.request().headers()["x-vector-lab"]).toBe("1");
    const id = route.request().url().split("/").pop()!;
    state.deleted.push(id);
    state.vectors = state.vectors.filter(vector => vector.id !== id);
    return route.fulfill({ json: { deleted: id } });
  });
  return state;
}

test("delete buttons are available per vector and cancellation keeps saved vectors", async ({ page }) => {
  const state = await mockBridge(page);
  await page.goto("/");
  const deleteButton = page.getByRole("button", { name: `Delete ${other.name}`, exact: true });
  await expect(deleteButton).toBeVisible();
  await deleteButton.click();
  const dialog = page.getByRole("alertdialog");
  await expect(dialog).toContainText(other.name);
  await expect(dialog).toContainText("cannot be undone");
  await expect(dialog.getByRole("button", { name: "Cancel", exact: true })).toBeFocused();
  await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect(deleteButton).toBeFocused();
  await deleteButton.press("Enter");
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(page.locator(".library-vector")).toHaveCount(2);
  expect(state.deleted).toEqual([]);
});

test("confirmed deletion removes steering, preserves other coefficients and selection, and persists", async ({ page }) => {
  const state = await mockBridge(page);
  await page.goto("/");
  await page.getByRole("slider", { name: custom.name, exact: true }).press("ArrowRight");
  await page.getByRole("slider", { name: other.name, exact: true }).press("ArrowRight");
  await page.locator(".vector-select").filter({ hasText: other.name }).click();
  await page.getByRole("button", { name: `Delete ${custom.name}`, exact: true }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete vector", exact: true }).click();
  await expect(page.getByRole("alertdialog")).toHaveCount(0);
  await expect(page.getByRole("button", { name: `Delete ${custom.name}`, exact: true })).toHaveCount(0);
  await expect(page.getByRole("slider", { name: custom.name, exact: true })).toHaveCount(0);
  await expect(page.getByLabel(`${other.name} coefficient`, { exact: true })).toContainText("+0.1");
  await expect(page.locator(".vector-select").filter({ hasText: other.name })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".library-section .count-badge")).toHaveText("1");
  await expect(page.locator(".control-card .count-badge")).toHaveText("1");
  await expect(page.locator(".toast")).toContainText(`${custom.name} deleted`);
  expect(state.deleted).toEqual([custom.id]);

  let coefficients: { id: string; value: number }[] = [];
  await page.route(`${bridge}/api/chat`, route => {
    coefficients = route.request().postDataJSON().coefficients;
    return route.fulfill({ contentType: "application/x-ndjson", body: '{"type":"done"}\n' });
  });
  await page.getByLabel("Message the model", { exact: true }).fill("Hello");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(page.getByRole("button", { name: "Send message", exact: true })).toBeVisible();
  expect(coefficients).toEqual([{ id: other.id, value: 0.1 }]);
  await page.reload();
  await expect(page.locator(".library-vector")).toHaveCount(1);
  await expect(page.locator(".vector-library")).toContainText(other.name);
});

test("deleting the selected and final vectors updates details, focus, and empty states", async ({ page }) => {
  await mockBridge(page);
  await page.goto("/");
  await page.getByRole("button", { name: `Delete ${custom.name}`, exact: true }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete vector", exact: true }).click();
  const remaining = page.locator(".vector-select").filter({ hasText: other.name });
  await expect(remaining).toHaveAttribute("aria-pressed", "true");
  await expect(remaining).toBeFocused();
  await expect(page.locator(".selected-detail")).toContainText("Contrast L2 7.00");
  await page.getByRole("button", { name: `Delete ${other.name}`, exact: true }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete vector", exact: true }).click();
  await expect(page.locator(".library-empty")).toBeVisible();
  await expect(page.locator(".empty-steering")).toBeVisible();
  await expect(page.locator(".selected-detail")).toHaveCount(0);
  await expect(page.locator(".library-section .count-badge")).toHaveText("0");
  await expect(page.getByLabel("Vector name", { exact: true })).toBeFocused();
});

test("failed deletion retains the vector and coefficient and can be retried", async ({ page }) => {
  const state = await mockBridge(page);
  await page.route(`${bridge}/api/vectors/${custom.id}`, route => route.fulfill({ status: 409, json: { detail: "The model is busy. Try again." } }));
  await page.goto("/");
  await page.getByRole("slider", { name: custom.name, exact: true }).press("ArrowRight");
  await page.getByRole("button", { name: `Delete ${custom.name}`, exact: true }).click();
  const dialog = page.getByRole("alertdialog");
  await dialog.getByRole("button", { name: "Delete vector", exact: true }).click();
  await expect(dialog.getByRole("alert")).toHaveText("The model is busy. Try again.");
  await expect(page.locator(".library-vector")).toHaveCount(2);
  await expect(page.getByLabel(`${custom.name} coefficient`, { exact: true })).toContainText("+0.1");
  await expect(page.locator(".toast")).toHaveCount(0);
  await page.unroute(`${bridge}/api/vectors/${custom.id}`);
  await dialog.getByRole("button", { name: "Delete vector", exact: true }).click();
  await expect(dialog).toHaveCount(0);
  expect(state.deleted).toEqual([custom.id]);
});

test("pending deletion blocks repeated requests, dismissal, and competing model operations", async ({ page }) => {
  await mockBridge(page);
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  let requests = 0;
  await page.route(`${bridge}/api/vectors/${custom.id}`, async route => {
    requests++;
    await pending;
    await route.fulfill({ json: { deleted: custom.id } });
  });
  try {
    await page.goto("/");
    await page.getByRole("button", { name: `Delete ${custom.name}`, exact: true }).click();
    const dialog = page.getByRole("alertdialog");
    await dialog.getByRole("button", { name: "Delete vector", exact: true }).click();
    await expect(dialog.getByRole("button", { name: "Deleting…", exact: true })).toBeDisabled();
    await expect(dialog.getByRole("button", { name: "Cancel", exact: true })).toBeDisabled();
    await expect(page.locator(".vector-delete").first()).toBeDisabled();
    await expect(page.getByRole("button", { name: "Extract vector", exact: true, includeHidden: true })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Connect", exact: true, includeHidden: true })).toBeDisabled();
    await expect(page.getByRole("button", { name: "Refresh local models", exact: true, includeHidden: true })).toBeDisabled();
    await page.keyboard.press("Escape");
    await expect(dialog).toBeVisible();
    await expect(page.locator(".library-vector")).toHaveCount(2);
    expect(requests).toBe(1);
    release();
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Extract vector", exact: true })).toBeEnabled();
  } finally { release(); }
});

test("deleting a newly extracted custom vector clears its saved extraction result", async ({ page }) => {
  const state = await mockBridge(page, []);
  await page.route(`${bridge}/api/extract`, route => {
    state.vectors.push(custom);
    return route.fulfill({ json: custom });
  });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Extract vector", exact: true })).toBeEnabled();
  await page.getByLabel("Vector name", { exact: true }).fill(custom.name);
  await page.getByRole("button", { name: "Extract vector", exact: true }).click();
  await expect(page.locator(".extraction-success")).toBeVisible();
  await page.getByRole("button", { name: `Delete ${custom.name}`, exact: true }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete vector", exact: true }).click();
  await expect(page.locator(".extraction-success")).toHaveCount(0);
  await expect(page.locator(".library-empty")).toBeVisible();
});

test("delete controls and confirmation fit desktop and mobile widths", async ({ page }) => {
  await mockBridge(page, [{ ...custom, name: "A very long custom vector name!!!" }]);
  await page.goto("/");
  for (const width of [1440, 1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    const deleteButton = page.getByRole("button", { name: "Delete A very long custom vector name!!!", exact: true });
    await expect(deleteButton).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    if (width === 1440 || width === 390) await page.screenshot({ path: `outputs/vector-deletion-${width}.png`, fullPage: true, animations: "disabled" });
    await deleteButton.click();
    const dialog = page.getByRole("alertdialog");
    const bounds = await dialog.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
    if (width === 390) await page.screenshot({ path: "outputs/vector-deletion-confirmation.png", animations: "disabled" });
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(dialog).toHaveCount(0);
  }
});
