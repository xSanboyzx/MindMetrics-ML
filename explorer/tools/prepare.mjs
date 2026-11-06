import { copyFile, mkdir, writeFile } from "node:fs/promises";
const runtime = new URL("../dist/runtime/", import.meta.url);
await mkdir(runtime, { recursive: true });
for (const file of [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]) {
  await copyFile(
    new URL(`../node_modules/onnxruntime-web/dist/${file}`, import.meta.url),
    new URL(file, runtime),
  );
}
await copyFile(
  new URL("../public/onnxruntime-LICENSE.txt", import.meta.url),
  new URL("LICENSE.txt", runtime),
);
const csp =
  "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; worker-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
await writeFile(
  new URL("../dist/_headers", import.meta.url),
  `/*\n  Content-Security-Policy: ${csp}\n  X-Content-Type-Options: nosniff\n  X-Frame-Options: DENY\n  Referrer-Policy: strict-origin-when-cross-origin\n  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()\n  Strict-Transport-Security: max-age=31536000\n  Cache-Control: no-cache\n`,
);
