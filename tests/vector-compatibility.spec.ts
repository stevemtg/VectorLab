import { test, expect, type Page } from "@playwright/test";

const legacy = {
  id: "legacy", name: "Saved unit vector", positive: "positive", negative: "negative",
  model_digest: "test-digest", model: "test-model", layers: 4, dimensions: 2,
  difference_norm: 3, layer_norms: [3, 4, 5, 6], method: "unit-normalized per layer",
};
const raw = {
  ...legacy, id: "raw", name: "Raw vector", mode: "final_token", best_layer: 3,
  extraction_layer: 1, auc_layer: 1, layer_aucs: [0.95, 0.8, 0.6, 0.5],
  layer_ratios: [1.0, 0.8, 0.6, 0.4], ratio: 0.6, ratio_target: 0.6,
  samples: { positive: 2, negative: 2 }, method: "raw difference",
};

async function mockBridge(page: Page) {
  await page.route("http://127.0.0.1:8788/api/models", route => route.fulfill({ json: {
    models: [{ name: "test-model", size: 1000, family: "test", parameter_size: "2", digest: "test-digest" }],
    default_model: "test-model",
  } }));
  await page.route("http://127.0.0.1:8788/api/status", route => route.fulfill({ json: {
    connected: true, engine: "native", model: "test-model", digest: "test-digest",
    layers: 5, dimensions: 2, vectors: [legacy, raw, { ...raw, id: "partial", name: "Missing AUC", layer_aucs: undefined }],
  } }));
}

test("saved legacy and raw vectors render, refresh, and stage coefficients without errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await mockBridge(page);
  await page.goto("/");
  const library = page.locator(".vector-library");
  await expect(library).toContainText("Legacy · unit vectors · 4 layers");
  await expect(library).toContainText("L3 · final token pool · train AUC 0.95");
  await expect(library).toContainText("train AUC unavailable");
  await expect(page.locator(".selected-detail")).toContainText("legacy unit vectors across 4 layers");
  await page.getByRole("slider", { name: legacy.name, exact: true }).press("ArrowRight");
  await expect(page.getByLabel(`${legacy.name} coefficient`, { exact: true })).toContainText("+0.1");
  await page.getByRole("button", { name: "Refresh local models" }).click();
  await expect(page.getByLabel(`${legacy.name} coefficient`, { exact: true })).toContainText("0.0");
  await page.reload();
  await expect(library).toContainText("Legacy · unit vectors · 4 layers");
  expect(errors).toEqual([]);
});

for (const oldBridge of [false, true]) {
  test(`extraction handles ${oldBridge ? "legacy bridge" : "raw vector"} responses`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await mockBridge(page);
    let requestBody: Record<string, unknown> = {};
    await page.route("http://127.0.0.1:8788/api/extract", route => {
      requestBody = route.request().postDataJSON();
      return route.fulfill({ json: { ...(oldBridge ? legacy : raw), id: "new", name: "New vector" } });
    });
    await page.goto("/");
    await expect(page.getByText("Real model connected", { exact: true })).toBeVisible();
    await page.getByLabel("Vector name", { exact: true }).fill("New vector");
    await page.getByLabel("Activation pooling mode").click();
    await page.getByRole("option", { name: "Final token", exact: true }).click();
    await page.getByRole("button", { name: "Extract vector", exact: true }).click();
    await expect(page.getByText("New vector extracted", { exact: true })).toBeVisible();
    await expect(page.locator(".extraction-success")).toContainText(oldBridge ? "legacy unit vectors across 4 layers" : "raw vector at L3");
    expect(requestBody.mode).toBe("final_token");
    await expect(page.getByRole("alert")).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}
