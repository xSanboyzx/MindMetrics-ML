# MindMetrics residual-network model card

## Intended use

Educational demonstration of tabular regression, preprocessing, evaluation, and browser inference. The public explorer uses fictional scenarios and a synthetic dataset. It has no clinical use, no real-population validation, no calibrated uncertainty, and no causal interpretation.

## Model and data

The committed checkpoint contains a 15 → 128 input projection, two 128-dimensional residual blocks, and a 64 → 32 → 1 head. Blocks use batch normalization, LeakyReLU, and dropout. Evaluation disables dropout and uses stored batch-normalization statistics. Target: `happiness_score`, observed range 0–10.

Data: [Digital Lifestyle Benchmark (2025), Tarek Masryo](https://huggingface.co/datasets/tarekmasryo/digital-lifestyle-benchmark-dataset), CC BY 4.0. See [dataset attribution](DATASET.md).

The original split uses two `train_test_split` calls with `random_state=42`: 2,450 training, 525 validation, 525 test records. StandardScaler parameters match a scaler fitted only on training records. Feature ordering and observed ranges are defined in `project/mindmetrics/features.json`.

## Reproduced checkpoint results

| Model | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: |
| Training-mean baseline | 2.701 | 3.099 | -0.000 |
| Linear regression | 1.701 | 2.208 | 0.492 |
| Committed residual network | 1.637 | 2.122 | 0.531 |

Results reproduce the stored notebook outputs to their reported precision. Baselines are fitted on training data only and evaluated on the same test indices. This is a modest improvement over linear regression, not evidence of performance on real people. Baseline comparisons are retrospective; this release does not tune models against the test set.

Run `python -m tools.release` to regenerate exact results, artifact hashes, ONNX graph, plot data, and parity fixtures. The exported graph bundles scaling with the network. Float32 preprocessing differs slightly from NumPy’s original float64 scaler arithmetic; held-out and browser checks bound the absolute score difference at `1e-4`.

## Known training limitation

The historical notebook assigned `best_state_dict = model.state_dict()` without copying tensor storage. Later optimizer steps could change the saved reference, so the file name `best_model.pt` does **not** prove these are the best validation-epoch weights. The notebook now uses `copy.deepcopy` for future runs. Existing weights and stored historical outputs are preserved, and no retraining or optimal-checkpoint claim is made.

## Further limitations

- Synthetic relationships can reflect the dataset generator, not real-world behavior.
- All 15 inputs are required; no missing-value imputation is learned.
- Range-valid combinations can still be outside the joint training distribution.
- Predictions are unconstrained regression outputs; values outside 0–10 are shown and flagged, not clipped.
- No subgroup fairness or demographic generalization claim is supported.
- Scores labeled anxiety, depression, focus, and productivity are dataset indicators, not clinical instruments.
- The browser downloads public model weights. Entered values are held in memory, with no account, persistence, or inference API. Hosting providers still receive ordinary asset requests.

## Reproducibility

The release pins Python dependencies and frontend dependencies and commits the Node lockfile. CI checks model loading, scaler consistency, Python evaluation, ONNX parity, browser WASM parity on 530 cases, mobile layout, scenario controls, missing-model recovery, and absence of external inference traffic.

The release uses PyTorch 2.15.0's modern dynamo exporter, ONNX opset 18, with ONNX checker and numerical parity gates. Upgrading the original toolchain resolved the exporter conversion failure and incorporated upstream security fixes.
