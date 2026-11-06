# MindMetrics-ML

An academic tabular-regression project, revisited as an interactive **model explorer**. Compare fictional profiles, inspect the actual trained network’s predictions, and see held-out evaluation alongside simple baselines.

**Live demo:** https://mindmetrics-explorer.sanhith-amarathunge.workers.dev

**Portfolio:** https://sanhithamarathunge.net/projects#project-mindmetrics

Originally a two-person CSCI 4050U machine learning course project. Primary implementation and showcase polish by **Sanhith Amarathunge**. Original history and notebooks are preserved.

## What the model demonstrates

The residual PyTorch network uses 15 ordered lifestyle/device/score variables from 3,500 **synthetic** records to predict `happiness_score`. Its committed checkpoint reproduces test MAE **1.637**, RMSE **2.122**, and R² **0.531**, compared with linear regression R² **0.492**.

These results describe the synthetic dataset. They do not establish clinical validity, causality, or performance on real people. Read the [model card](docs/MODEL_CARD.md), [release review](docs/REVIEW.md), and [dataset attribution](docs/DATASET.md).

## Reproduce the verified release

Python **3.13** and Node **22.13+ (22.x)**. Commands below start from the repository root.

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip==26.3 setuptools==84.2.1
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python -m tools.release
cd explorer
npm ci
npm run build
npx playwright install chromium
npm test
npm run preview
```

Open http://127.0.0.1:8791. `npm run dev` is available for UI iteration after generating model artifacts. Build/preview serve the production CSP and local runtime assets; use those for final verification.

`requirements.txt` uses the verified transitive pins in `constraints.txt`. CI also audits the installed Python environment and npm packages.

The release command verifies the training-only scaler, reproduces the original split, evaluates the committed weights and baselines, exports ONNX, and generates `explorer/public/release.json` and test fixtures. No retraining is performed. The browser uses the exported network with bundled preprocessing; it makes no inference API calls and keeps inputs in memory.

### Historical notebooks

Install python -m pip install -r requirements-notebooks.txt for notebook visualization and Jupyter tooling. Rerunning training changes the checkpoint; the release evaluation does not train.

### Local Streamlit companion

From the repository root, with the same Python environment:

```sh
python -m streamlit run project/deployment/app.py --browser.gatherUsageStats=false
```

### Deploy the static explorer

After the verified build, from the repository root:

```sh
npx wrangler deploy
```

Cloudflare serves `explorer/dist`. There is no Python inference server or database. Security headers are generated during build and apply only to this standalone demo. Inputs are not submitted; the host receives normal requests for application/model assets.

## Layout

- `project/mindmetrics/`: shared architecture, ordered feature contract, presets, validated inference.
- `project/notebooks/`: historical EDA and training notebooks. Rerunning training changes artifacts; the best-checkpoint copy bug is fixed for future runs.
- `project/deployment/`: corrected Streamlit companion and original checkpoint/scaler.
- `tools/release.py`: evaluation, export, hashes, plot data, parity gate.
- `explorer/`: React/TypeScript and ONNX Runtime Web interface.
- `tests/`: Python checks and synthetic parity fixtures.

## Data and acknowledgements

Dataset: **Tarek Masryo, Digital Lifestyle Benchmark Dataset (2025)**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Author’s dataset card](https://huggingface.co/datasets/tarekmasryo/digital-lifestyle-benchmark-dataset). The public release includes derived evaluation data and authored fictional scenarios; it does not serve the raw CSV.

Built with Python, PyTorch, NumPy, pandas, scikit-learn, Streamlit, ONNX, React, TypeScript, and ONNX Runtime Web. Third-party model-runtime license is included with the deployed assets. The original project README granted permission to modify and extend this academic project; this release preserves that notice and does not replace third-party licenses.
