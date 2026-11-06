import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [react()],
  build: {
    manifest: true,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        inference: fileURLToPath(
          new URL("./src/inference.ts", import.meta.url),
        ),
      },
      preserveEntrySignatures: "strict",
    },
  },
});
