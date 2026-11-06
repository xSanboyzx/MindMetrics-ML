# Dataset provenance

Digital Lifestyle Benchmark Dataset (2025), by **Tarek Masryo**.

- Author’s dataset card: https://huggingface.co/datasets/tarekmasryo/digital-lifestyle-benchmark-dataset
- License: **Creative Commons Attribution 4.0**, https://creativecommons.org/licenses/by/4.0/
- 3,500 synthetic participant records, 24 columns. These are generated benchmark records, not observations of real people.
- Original academic repository stores the CSV as `project/data/raw/dataset.csv`. It uses 15 numeric columns to predict `happiness_score`; the other columns are excluded.
- This release derives feature ranges, aggregate metrics, a held-out prediction plot, and parity fixtures from that CSV. Fictional presets were authored for the demo rather than copied from participants.
- Artifact SHA-256 hashes are recorded in `explorer/public/release.json`. No raw CSV is served by the public explorer.

The dataset card identifies scores as modeling variables. It does not establish clinical validity for them or supply a causal interpretation. Dataset scales and observed ranges must not be interpreted as validated clinical assessment scales.

Validation: the repository CSV matches the author's published CSV as a parsed pandas DataFrame (all rows, columns, and values). Author CSV SHA-256: a59f935b376cc6098bd3cd2a8eaa993a3e6b71b7eba01a3fec9b700b5e69b120. Verified 2026-11-05.

