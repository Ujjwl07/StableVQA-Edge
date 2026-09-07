# T4 validation output (StableDB validation split, 391 videos)

Per-clip output of the Linux GPU build (NVIDIA T4, concurrent streams) over
StableDB. The 391 videos of the validation split are the ones with labels
below and are the run behind the paper's accuracy numbers; the analysis script
skips the unlabelled videos automatically.

- `stablevqa_results.csv` — one row per clip (4 per video): score, per-branch
  latency, RSS. Per-branch latencies were measured under concurrent execution
  and are inflated by contention; only `TotalLatency(s)` is a wall-clock figure.
- `stabledb_val_labels.csv` — `video_id,gt_score` for the same 391 videos
  (StableDB MOS from Kou et al., ACM MM 2023). **Confirm redistribution terms
  for these labels before publishing this folder**, exactly as for the models.

Reproduce the paper's accuracy table and the fixed-k control in one command:

```sh
python3 scripts/adaptive_temporal_inference.py \
  samples/t4_validation_output/stablevqa_results.csv \
  --labels samples/t4_validation_output/stabledb_val_labels.csv --sweep
```

Expected (SROCC): four clips 0.9342, first three 0.9292, first two 0.9228,
first clip 0.9095, convergence rule (0.03, 2.0) 0.9247 at 2.38 clips.
The in-binary evaluator reports 0.9344 for four clips; the fourth-decimal
difference is rank-tie handling.
