# Android reference output (Exynos 2400)

These are reference outputs from a real Exynos 2400 on-device run, provided so a reviewer can compare their own run against known-good values.

- `stablevqa_results.csv` — per-clip scores, per-branch latency, and RSS for a sample run.
- `stablevqa_calibration.csv` — the raw-to-MOS calibration outputs for the same run.

Use these to sanity-check a fresh build: your per-video mean scores should match
these within the paper's tolerance (|ΔScore| ≈ 0.044 MOS). Latencies will differ
by device and thermal state.
