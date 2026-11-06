"""Reproduce checkpoint evaluation, export ONNX, and generate demo metadata."""

import hashlib
import importlib.metadata
import json

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from project.mindmetrics.inference import FEATURES, FEATURE_COLS, PROJECT, load_artifacts, predict

ROOT = PROJECT.parent
PUBLIC = ROOT / "explorer" / "public"


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release():
    torch.set_num_threads(1)
    data_path = PROJECT / "data" / "raw" / "dataset.csv"
    frame = pd.read_csv(data_path)
    x = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = frame["happiness_score"].to_numpy(dtype=np.float32)
    if x.shape != (3500, 15) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Unexpected dataset shape or non-finite data.")
    indices = np.arange(len(x))
    train, temp = train_test_split(indices, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    artifacts = load_artifacts()
    model, mean, scale = artifacts
    fitted = StandardScaler().fit(x[train])
    np.testing.assert_allclose(mean, fitted.mean_, rtol=0, atol=1e-10)
    np.testing.assert_allclose(scale, fitted.scale_, rtol=0, atol=1e-10)
    predictions = predict(x[test], artifacts)
    linear = LinearRegression().fit(fitted.transform(x[train]), y[train])
    baselines = [
        {"name": "Training mean", **metrics(y[test], np.full(len(test), y[train].mean()))},
        {
            "name": "Linear regression",
            **metrics(y[test], linear.predict(fitted.transform(x[test]))),
        },
        {"name": "Residual neural network", **metrics(y[test], predictions)},
    ]

    # Bundle preprocessing into the graph; parity checks bound float32 rounding
    # differences against the original NumPy float64 scaler arithmetic.
    class BrowserModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

        def forward(self, features):
            scaled = (features - self.mean) / self.scale
            return self.model(scaled)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    onnx_path = PUBLIC / "model.onnx"
    torch.onnx.export(
        BrowserModel().eval(),
        (torch.from_numpy(x[test[:2]]),),
        str(onnx_path),
        input_names=["features"],
        output_names=["score"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        verbose=False,
        dynamic_shapes={"features": {0: torch.export.Dim("batch", min=1)}},
    )
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    presets = json.loads((ROOT / "project/mindmetrics/presets.json").read_text())
    preset_x = np.array([p["values"] for p in presets], dtype=np.float32)
    boundaries = np.array(
        [[f["min"] for f in FEATURES], [f["max"] for f in FEATURES]], dtype=np.float32
    )
    parity_x = np.concatenate([x[test], preset_x, boundaries])
    expected = predict(parity_x, artifacts)
    exported = session.run(None, {"features": parity_x})[0]
    error = float(np.max(np.abs(expected - exported)))
    np.testing.assert_allclose(expected, exported, rtol=0, atol=1e-4)
    report = {
        "dataset": {
            "rows": len(frame),
            "features": len(FEATURES),
            "target": "happiness_score",
            "targetRange": [0, 10],
        },
        "split": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "randomState": 42,
        },
        "metrics": baselines,
        "historicalNotebook": {"mae": 1.6368, "rmse": 2.1221, "r2": 0.5312},
        "parity": {"samples": len(parity_x), "maxAbsoluteError": error, "tolerance": 1e-4},
        "hashes": {
            "dataset": digest(data_path),
            "checkpoint": digest(PROJECT / "deployment/model/best_model.pt"),
            "scaler": digest(PROJECT / "deployment/model/scaler_params.npz"),
            "onnx": digest(onnx_path),
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "numpy",
                "scikit-learn",
                "onnx",
                "onnxruntime",
                "onnxscript",
                "pandas",
            ]
        },
        "scatter": [
            {"actual": float(a), "predicted": float(p)}
            for a, p in zip(y[test], predictions, strict=True)
        ],
        "features": FEATURES,
        "presets": presets,
    }
    (PUBLIC / "release.json").write_text(json.dumps(report, indent=2) + "\n")
    # Expected values are used to verify the browser runtime on every held-out
    # example, plus presets and boundary cases. These are synthetic records.
    fixtures = {"inputs": parity_x.tolist(), "expected": expected.tolist()}
    (ROOT / "tests" / "parity.json").write_text(json.dumps(fixtures) + "\n")
    print(
        json.dumps(
            {"metrics": baselines, "parity": report["parity"], "scalerMatchesTrainingOnly": True},
            indent=2,
        )
    )


if __name__ == "__main__":
    release()
