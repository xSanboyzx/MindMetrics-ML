import * as ort from "onnxruntime-web/wasm";

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = `${location.origin}/runtime/`;

let session: Promise<ort.InferenceSession> | undefined;
export function initialize() {
  session ??= ort.InferenceSession.create("/model.onnx", {
    executionProviders: ["wasm"],
  }).catch((error: unknown) => {
    session = undefined;
    throw error;
  });
  return session;
}

export async function predict(rows: number[][]): Promise<number[]> {
  if (
    !rows.length ||
    rows.some(
      (row) =>
        row.length !== 15 || row.some((value) => !Number.isFinite(value)),
    )
  ) {
    throw new Error("Every profile needs 15 finite feature values.");
  }
  const model = await initialize();
  const input = new ort.Tensor("float32", Float32Array.from(rows.flat()), [
    rows.length,
    15,
  ]);
  try {
    const output = await model.run({ features: input });
    try {
      const scores = Array.from(output.score.data as Float32Array);
      if (scores.some((value) => !Number.isFinite(value)))
        throw new Error("Invalid model output.");
      return scores;
    } finally {
      Object.values(output).forEach((tensor) => tensor.dispose());
    }
  } finally {
    input.dispose();
  }
}
