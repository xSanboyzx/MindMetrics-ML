import assert from "node:assert/strict";
import { readFile, mkdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const base = process.env.DEMO_URL || "http://127.0.0.1:8791";
const output = new URL("../../.artifacts/", import.meta.url);
await mkdir(output, { recursive: true });
const server = process.env.DEMO_URL
  ? null
  : spawn(process.execPath, ["tools/serve.mjs"], {
      stdio: "pipe",
      windowsHide: true,
    });
let browser;
try {
  for (let i = 0; i < 80; i++) {
    try {
      if ((await fetch(base)).ok) break;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
    if (i === 79) throw new Error("Preview server did not start.");
  }
  browser = await chromium.launch({
    channel: process.env.BROWSER_CHANNEL || undefined,
    headless: true,
  });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    recordVideo: {
      dir: output.pathname.replace(/^\/(\w:)/, "$1"),
      size: { width: 1280, height: 900 },
    },
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const errors = [],
    external = [],
    submissions = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("request", (request) => {
    if (!request.url().startsWith(base) && !request.url().startsWith("data:"))
      external.push(request.url());
    if (request.method() !== "GET") submissions.push(request.url());
  });
  await page.addInitScript(() => {
    window.__cspViolations = [];
    document.addEventListener("securitypolicyviolation", (event) =>
      window.__cspViolations.push(event.violatedDirective),
    );
  });
  await page.goto(base);
  const compare = page.getByRole("button", { name: "Compare predictions" });
  await compare.waitFor();
  await page.waitForFunction(
    () =>
      ![...document.querySelectorAll("button")].find((b) =>
        b.textContent.includes("Compare predictions"),
      ).disabled,
  );
  await page.screenshot({
    path: new URL("overview.png", output).pathname.replace(/^\/(\w:)/, "$1"),
  });
  // Verify the actual shipped WASM runtime against every Python fixture.
  const manifest = JSON.parse(
    await readFile(
      new URL("../dist/.vite/manifest.json", import.meta.url),
      "utf8",
    ),
  );
  const inferenceEntry = manifest["src/inference.ts"].file;
  const fixtures = JSON.parse(
    await readFile(new URL("../../tests/parity.json", import.meta.url), "utf8"),
  );
  const parity = await page.evaluate(
    async ({ entry, fixtures }) => {
      const { predict } = await import("/" + entry);
      const results = await predict(fixtures.inputs);
      return Math.max(
        ...results.map((result, i) => Math.abs(result - fixtures.expected[i])),
      );
    },
    { entry: inferenceEntry, fixtures },
  );
  assert.ok(parity < 1e-4, `Browser parity error: ${parity}`);
  await page.getByRole("link", { name: "Explorer", exact: true }).click();
  await page.waitForTimeout(1000);
  const baseline = await page.getByTestId("adjusted-score").textContent();
  await page
    .getByRole("spinbutton", { name: "Sleep duration", exact: true })
    .fill("9");
  assert.ok(
    await page
      .getByText("Inputs changed. Compare to update this result.")
      .isVisible(),
  );
  await compare.click();
  await page.waitForTimeout(800);
  assert.notEqual(
    await page.getByTestId("adjusted-score").textContent(),
    baseline,
  );
  assert.ok(await page.getByText("7.00 → 9.00", { exact: true }).isVisible());
  await page.evaluate(() =>
    scrollTo(0, document.querySelector("#explore").offsetTop - 30),
  );
  await page.screenshot({
    path: new URL("comparison.png", output).pathname.replace(/^\/(\w:)/, "$1"),
  });
  // Invalid number input must not silently run an old valid value.
  await page
    .getByRole("spinbutton", { name: "Sleep duration", exact: true })
    .fill("99");
  assert.equal(await compare.isDisabled(), true);
  await page.getByRole("button", { name: "Reset changes" }).click();
  assert.equal(await compare.isDisabled(), false);
  assert.equal(
    await page.getByTestId("adjusted-score").textContent(),
    baseline,
  );
  const sleep = page.getByRole("spinbutton", {
    name: "Sleep duration",
    exact: true,
  });
  await sleep.fill("");
  await sleep.pressSequentially("8.25");
  assert.equal(await sleep.inputValue(), "8.25");
  await page.getByRole("button", { name: "Reset changes" }).click();
  await page.getByRole("button", { name: /Deadline week/ }).click();
  await page.waitForTimeout(700);
  assert.equal(
    await page
      .getByRole("spinbutton", { name: "Sleep duration", exact: true })
      .inputValue(),
    "5.5",
  );
  await page.getByRole("link", { name: "Evaluation", exact: true }).click();
  await page.waitForTimeout(700);
  await page.screenshot({
    path: new URL("evaluation.png", output).pathname.replace(/^\/(\w:)/, "$1"),
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("link", { name: "Explorer", exact: true }).click();
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  await page.screenshot({
    path: new URL("mobile.png", output).pathname.replace(/^\/(\w:)/, "$1"),
  });
  assert.deepEqual(await page.evaluate(() => window.__cspViolations), []);
  assert.deepEqual(external, []);
  assert.deepEqual(submissions, []);
  assert.deepEqual(errors, []);
  await context.close();
  // Recoverable missing-model failure.
  const failure = await browser.newPage();
  await failure.route("**/model.onnx", (route) => route.abort());
  await failure.goto(base);
  await failure.getByRole("button", { name: "Retry model" }).waitFor();
  await failure.unroute("**/model.onnx");
  await failure.getByRole("button", { name: "Retry model" }).click();
  await failure.getByTestId("adjusted-score").waitFor();
  await failure.close();
  console.log(
    JSON.stringify(
      {
        browserParityMaxError: parity,
        samples: fixtures.inputs.length,
        desktop: "passed",
        mobile: "passed",
        reset: "passed",
        failureRecovery: "passed",
        externalRequests: external.length,
        inputSubmissions: submissions.length,
        errors,
      },
      null,
      2,
    ),
  );
} finally {
  await browser?.close();
  server?.kill();
}
