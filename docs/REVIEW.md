# Release review

Original baseline: `9affba9` (final MLP update). Original two-person academic project; primary implementation and this showcase release by Sanhith Amarathunge. Original Git history and academic notebook outputs are retained.

## Fixed

- Streamlit defined an incompatible small sequential network, while the saved checkpoint used residual blocks. Both now use the shared architecture with strict checkpoint loading.
- UI ranges disagreed with the CSV for sleep quality, anxiety, depression, focus, productivity, digital dependence, and other fields. The shared feature contract records observed ranges and units.
- Artifact paths depended on shell working directory. They now resolve relative to source files.
- Model loading is CPU-only, `weights_only=True`, strict, and validates scaler dimensions, finiteness, and positive scales. The old pickle scaler is retained as historical material and never loaded.
- Notebook checkpoint selection used mutable state references. Future training uses a deep copy; existing weights are evaluated honestly.
- Dependencies were unpinned and included unused training frameworks. The release specifies its actual runtime/export/test dependencies.
- README overstated personal health interpretation and had incorrect setup paths. Documentation now describes educational use and reproducible commands.

## Added

- Reproduced held-out evaluation and training-only mean/linear baselines.
- ONNX export with bundled preprocessing and parity gate.
- Browser-only model explorer, fictional profiles, scenario comparison/reset, held-out plot, and clear evaluation context.
- Cloudflare static hosting and isolated CSP; model/runtime assets are served locally.
- Python and browser checks in CI, with downloadable verified web artifacts.
- Model card and CC BY 4.0 dataset attribution.

## Future work

If the project becomes actively maintained: repeat seeded training with validation-only model selection, add a stronger tree-based baseline, examine residuals and subgroup performance, and test distribution shifts. Those are new experiments, not prerequisites for presenting this fixed academic checkpoint. Real-world use would require an appropriate dataset, a clearly defined task, and separate validation.

## Verified release checks

Seven Python/Streamlit tests pass; Ruff checks and formatting pass. The original held-out metrics reproduce. ONNX Runtime Web matches PyTorch on 530 synthetic cases within 0.0001 absolute score error. Local and live Chrome checks pass for comparisons, reset, mobile layout, and missing-model recovery, with no external inference requests or input submissions. The verified Python environment audit and frontend npm audit report no known vulnerabilities. Patched transitive Python dependencies are pinned in constraints.txt.

