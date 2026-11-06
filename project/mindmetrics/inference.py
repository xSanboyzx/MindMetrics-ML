"""Validated CPU inference; paths are independent of the working directory."""

import json
from pathlib import Path

import numpy as np
import torch

from .model import QoLRegressor

PROJECT = Path(__file__).resolve().parents[1]
FEATURES = json.loads((Path(__file__).with_name("features.json")).read_text())
FEATURE_COLS = [feature["key"] for feature in FEATURES]


def load_artifacts(model_dir: Path | None = None):
    directory = model_dir or PROJECT / "deployment" / "model"
    with np.load(directory / "scaler_params.npz", allow_pickle=False) as params:
        mean, scale = params["mean"].copy(), params["scale"].copy()
    if mean.shape != (len(FEATURES),) or scale.shape != mean.shape:
        raise ValueError("Scaler dimensions do not match the feature contract.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Scaler must contain finite means and positive scales.")
    model = QoLRegressor(len(FEATURES))
    model.load_state_dict(
        torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    model.eval()
    return model, mean, scale


def predict(values, artifacts=None):
    model, mean, scale = artifacts or load_artifacts()
    x = np.asarray(values, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("Expected finite values for all 15 features in contract order.")
    scaled = ((x - mean) / scale).astype(np.float32)
    with torch.inference_mode():
        result = model(torch.from_numpy(scaled)).numpy()
    if not np.isfinite(result).all():
        raise ValueError("The model returned a non-finite prediction.")
    return result
