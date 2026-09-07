#!/usr/bin/env python3
"""Adaptive Temporal Early-Stopping (offline analysis) with fixed-k control.

This reproduces the convergence-based early-stopping rule described in the
paper and, when ground-truth labels are supplied, the accuracy of that rule
against the simplest possible controls: always scoring the first k clips.

It is an OFFLINE analysis. The deployed C++/Android pipeline evaluates a fixed
four clips per video and writes per-clip scores to its results CSV; this script
replays those per-clip scores in order. It is deliberately kept separate from
the real-time inference path so the deployed tool's behaviour is unchanged and
fully auditable.

Convergence criterion (paper Eq. 1), evaluated after clip k >= 2 over the
running per-clip scores m_1..m_k:

    stop if   |mean_k - mean_{k-1}| / |mean_{k-1}|  <  tau_rel
         or   stddev(scores_1..k)                   <  tau_std      (OR rule)

Defaults: tau_rel = 0.03, tau_std = 2.0 (the paper's selected operating point).

Usage:
    python3 adaptive_temporal_inference.py results.csv
    python3 adaptive_temporal_inference.py results.csv --tau-rel 0.03 --tau-std 2.0
    python3 adaptive_temporal_inference.py results.csv --sweep
    python3 adaptive_temporal_inference.py results.csv --labels labels.csv
    python3 adaptive_temporal_inference.py results.csv --labels labels.csv --sweep

The results CSV must contain the columns VideoName, Clip, Score (as written by
every platform implementation in this repository). The optional labels CSV must
contain video_id and gt_score columns; video ids are matched with or without a
file extension. With labels, the script reports SROCC / PLCC / RMSE for the
fixed-k schedules (k = 1..4) and for the adaptive rule, using the same
z-score rescaling as the C++ evaluator before computing PLCC and RMSE.

No third-party dependencies.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import OrderedDict


def _strip_ext(name: str) -> str:
    return os.path.splitext(name.strip())[0]


def load_clip_scores(csv_path: str) -> "OrderedDict[str, list[float]]":
    """Return {video_name: [clip1_score, clip2_score, ...]} in clip order."""
    by_video: "OrderedDict[str, list[tuple[int, float]]]" = OrderedDict()
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"VideoName", "Clip", "Score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"CSV is missing required column(s): {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            name = row["VideoName"].strip()
            try:
                clip = int(row["Clip"])
                score = float(row["Score"])
            except (TypeError, ValueError):
                continue
            by_video.setdefault(name, []).append((clip, score))

    ordered: "OrderedDict[str, list[float]]" = OrderedDict()
    for name, pairs in by_video.items():
        pairs.sort(key=lambda p: p[0])  # sort by clip index
        ordered[name] = [s for _, s in pairs]
    return ordered


def load_labels(csv_path: str) -> "dict[str, float]":
    """Return {video_id_without_extension: gt_score}."""
    labels: "dict[str, float]" = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        if not {"video_id", "gt_score"} <= fields:
            raise SystemExit(
                f"labels CSV needs columns video_id and gt_score; found {sorted(fields)}"
            )
        for row in reader:
            try:
                labels[_strip_ext(row["video_id"])] = float(row["gt_score"])
            except (TypeError, ValueError):
                continue
    return labels


def _running_std(values: "list[float]") -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n  # population std, as in paper
    return math.sqrt(var)


def adaptive_prediction(scores: "list[float]", tau_rel: float, tau_std: float) -> "tuple[float, int]":
    """(video score, clips used) under the convergence rule for one video."""
    total = len(scores)
    if total <= 1:
        return (scores[0] if scores else 0.0), total

    seen: "list[float]" = [scores[0]]
    prev_mean = scores[0]
    for k in range(2, total + 1):
        seen.append(scores[k - 1])
        cur_mean = sum(seen) / len(seen)

        rel_change = abs(cur_mean - prev_mean) / abs(prev_mean) if prev_mean != 0 else 0.0
        std_k = _running_std(seen)

        if rel_change < tau_rel or std_k < tau_std:  # OR rule (paper Eq. 1)
            return cur_mean, k
        prev_mean = cur_mean
    return sum(scores) / total, total


def clips_used(scores: "list[float]", tau_rel: float, tau_std: float) -> int:
    return adaptive_prediction(scores, tau_rel, tau_std)[1]


# ---------------------------------------------------------------------------
# Correlation metrics (mirror the C++ evaluator in platform/gpu/StableVqaGpu.cpp)
# ---------------------------------------------------------------------------
def _ranks(x: "list[float]") -> "list[float]":
    idx = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _pearson(a: "list[float]", b: "list[float]") -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


def srocc(pred: "list[float]", gt: "list[float]") -> float:
    return _pearson(_ranks(pred), _ranks(gt))


def _rescale(pred: "list[float]", gt: "list[float]") -> "list[float]":
    n = len(pred)
    mp, mg = sum(pred) / n, sum(gt) / n
    sp = math.sqrt(sum((x - mp) ** 2 for x in pred) / n)
    sg = math.sqrt(sum((x - mg) ** 2 for x in gt) / n)
    if sp == 0:
        return [mg] * n
    return [(x - mp) / sp * sg + mg for x in pred]


def plcc_rmse(pred: "list[float]", gt: "list[float]") -> "tuple[float, float]":
    rs = _rescale(pred, gt)
    rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(rs, gt)) / len(gt))
    return _pearson(rs, gt), rmse


# ---------------------------------------------------------------------------
def evaluate(videos: "OrderedDict[str, list[float]]", tau_rel: float, tau_std: float) -> dict:
    counts = [clips_used(s, tau_rel, tau_std) for s in videos.values()]
    fixed = [len(s) for s in videos.values()]
    n = len(counts)
    if n == 0:
        raise SystemExit("No videos with per-clip scores found in the CSV.")

    mean_clips = sum(counts) / n
    fixed_mean = sum(fixed) / n
    saving = 1.0 - (mean_clips / fixed_mean) if fixed_mean else 0.0

    dist = {}
    for c in counts:
        dist[c] = dist.get(c, 0) + 1

    return {
        "num_videos": n,
        "mean_clips": mean_clips,
        "fixed_mean": fixed_mean,
        "saving": saving,
        "exit_distribution": dict(sorted(dist.items())),
    }


def accuracy_table(videos, labels, tau_rel, tau_std, grid=None):
    """Print SROCC/PLCC/RMSE for fixed-k schedules and the adaptive rule."""
    ids = [v for v in videos if _strip_ext(v) in labels]
    if len(ids) < 2:
        raise SystemExit("Fewer than two videos matched the labels file; check video ids.")
    skipped = len(videos) - len(ids)
    gt = [labels[_strip_ext(v)] for v in ids]
    max_k = max(len(videos[v]) for v in ids)

    print(f"Videos with labels     : {len(ids)}" + (f"  ({skipped} unlabelled skipped)" if skipped else ""))
    print(f"{'schedule':<30} {'clips':>6} {'SROCC':>8} {'PLCC':>8} {'RMSE':>8}")

    def row(name, preds, mean_clips):
        s = srocc(preds, gt)
        p, r = plcc_rmse(preds, gt)
        print(f"{name:<30} {mean_clips:6.2f} {s:8.4f} {p:8.4f} {r:8.3f}")

    for k in range(1, max_k + 1):
        row(f"fixed k={k} (first {k} clip{'s' if k > 1 else ''})",
            [sum(videos[v][:k]) / len(videos[v][:k]) for v in ids], float(k))

    for tr, ts in (grid or [(tau_rel, tau_std)]):
        res = [adaptive_prediction(videos[v], tr, ts) for v in ids]
        row(f"adaptive tau_rel={tr} tau_std={ts}",
            [m for m, _ in res], sum(k for _, k in res) / len(res))


SWEEP_GRID = [
    (0.02, 1.0),
    (0.03, 1.5),
    (0.03, 2.0),
    (0.05, 3.0),
    (0.08, 3.0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Per-clip results CSV written by the deployed pipeline")
    ap.add_argument("--tau-rel", type=float, default=0.03, help="relative-change threshold (default 0.03)")
    ap.add_argument("--tau-std", type=float, default=2.0, help="running-std threshold (default 2.0)")
    ap.add_argument("--sweep", action="store_true", help="run the paper's threshold sweep")
    ap.add_argument("--labels", help="CSV with video_id,gt_score columns; enables SROCC/PLCC/RMSE and the fixed-k control")
    args = ap.parse_args()

    videos = load_clip_scores(args.csv)

    if args.labels:
        labels = load_labels(args.labels)
        accuracy_table(videos, labels, args.tau_rel, args.tau_std,
                       grid=SWEEP_GRID if args.sweep else None)
        return 0

    if args.sweep:
        print(f"{'tau_rel':>8} {'tau_std':>8} {'mean_clips':>11} {'saving':>8}")
        for tr, ts in SWEEP_GRID:
            r = evaluate(videos, tr, ts)
            print(f"{tr:8.3f} {ts:8.2f} {r['mean_clips']:11.3f} {r['saving']*100:7.1f}%")
        return 0

    r = evaluate(videos, args.tau_rel, args.tau_std)
    print(f"Videos analysed        : {r['num_videos']}")
    print(f"Fixed clips per video  : {r['fixed_mean']:.2f}")
    print(f"Adaptive mean clips    : {r['mean_clips']:.3f}  (tau_rel={args.tau_rel}, tau_std={args.tau_std})")
    print(f"Per-video compute saving: {r['saving']*100:.1f}%")
    print(f"Exit distribution (clips: #videos): {r['exit_distribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
