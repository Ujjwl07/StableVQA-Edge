#!/usr/bin/env python3
"""Release gate: the pre-release regression suite for the StableVQA C++ deployment.

Every check described in the paper's assurance section is implemented here so
that a reviewer, or CI, can reproduce them with a single command. The script
has no third-party dependencies and exits non-zero if any enabled gate fails.

Gates
-----
1. models      SHA-256 of every shipped ONNX file against MODEL_MANIFEST.sha256.
2. schema      Every video has the expected clip count; no missing/NaN scores.
3. accuracy    SROCC / PLCC / KROCC / RMSE against ground-truth labels, with a
               minimum-SROCC threshold. Predictions are linearly rescaled to the
               label distribution before PLCC/RMSE, matching the evaluator built
               into the C++ binaries.
4. equivalence Per-clip and per-video score agreement against a reference run
               from another platform, plus the decision-flip rate: how often the
               two platforms disagree about which of two videos is more stable.
               This is the figure that matters when the tool is used to compare
               stabilization candidates, and it is reported at a configurable
               margin because near-ties are where platforms disagree.
5. pairwise    On synchronized benchmarks, the stabilized video of each pair must
               score above its handheld counterpart.

Usage
-----
  # full accuracy gate
  python3 scripts/release_gate.py --results run.csv \
      --labels samples/t4_validation_output/stabledb_val_labels.csv

  # cross-platform equivalence against a reference platform
  python3 scripts/release_gate.py --results mac.csv --reference t4.csv

  # model integrity only
  python3 scripts/release_gate.py --models onnx_models

  # synchronized pairwise ranking (pairs.csv: unstable_id,stable_id)
  python3 scripts/release_gate.py --results deepstab.csv --pairs pairs.csv

Any combination of gates may be run at once. Gates whose inputs are not supplied
are skipped and reported as SKIP, never as PASS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys

# ---------------------------------------------------------------------------
# statistics (mirrors the evaluator in platform/gpu/StableVqaGpu.cpp)
# ---------------------------------------------------------------------------


def _ranks(x):
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


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


def srocc(pred, gt):
    return _pearson(_ranks(pred), _ranks(gt))


def krocc(pred, gt):
    n = len(pred)
    nc = nd = t1 = t2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pred[i] - pred[j]
            dy = gt[i] - gt[j]
            if dx == 0:
                t1 += 1
            if dy == 0:
                t2 += 1
            if dx and dy:
                nc += 1 if dx * dy > 0 else 0
                nd += 1 if dx * dy < 0 else 0
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - t1) * (n0 - t2))
    return (nc - nd) / den if den else float("nan")


def rescale(pred, gt):
    """Linear z-score map of predictions onto the label distribution."""
    n = len(pred)
    mp, mg = sum(pred) / n, sum(gt) / n
    sp = math.sqrt(sum((x - mp) ** 2 for x in pred) / n)
    sg = math.sqrt(sum((x - mg) ** 2 for x in gt) / n)
    if sp == 0:
        return [mg] * n
    return [(x - mp) / sp * sg + mg for x in pred]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _strip(name):
    return os.path.splitext(name.strip())[0]


def load_clip_scores(path):
    """{video_id: [clip scores in clip order]} from any platform's results CSV."""
    rows = list(csv.reader(open(path, newline="")))
    if not rows:
        raise SystemExit(f"{path}: empty file")
    hdr = rows[0]
    for col in ("Score", "MOS"):
        if col in hdr:
            ci = hdr.index(col)
            break
    else:
        raise SystemExit(f"{path}: no Score or MOS column; found {hdr}")
    clip_i = hdr.index("Clip") if "Clip" in hdr else None
    by = {}
    for r in rows[1:]:
        if len(r) <= ci or not r[0].strip() or r[0].startswith("SUMMARY"):
            continue
        try:
            score = float(r[ci])
        except ValueError:
            continue
        if math.isnan(score):
            continue
        order = 0
        if clip_i is not None and len(r) > clip_i:
            try:
                order = int(r[clip_i])
            except ValueError:
                order = 0
        by.setdefault(_strip(r[0]), []).append((order, score))
    return {v: [s for _, s in sorted(p, key=lambda t: t[0])] for v, p in by.items()}


def load_labels(path):
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        f = set(r)
        if not {"video_id", "gt_score"} <= f:
            raise SystemExit(f"{path}: needs video_id,gt_score columns; found {sorted(f)}")
        try:
            out[_strip(r["video_id"])] = float(r["gt_score"])
        except (TypeError, ValueError):
            continue
    return out


def load_pairs(path):
    pairs = []
    for r in csv.reader(open(path, newline="")):
        if len(r) >= 2 and r[0].strip() and not r[0].lower().startswith("unstable"):
            pairs.append((_strip(r[0]), _strip(r[1])))
    return pairs


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class Report:
    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, gate, status, detail):
        if status == "FAIL":
            self.failed = True
        self.rows.append((gate, status, detail))

    def emit(self):
        w = max(len(g) for g, _, _ in self.rows) if self.rows else 12
        print("\n" + "=" * 72)
        print("StableVQA release gate")
        print("=" * 72)
        for gate, status, detail in self.rows:
            print(f"  [{status:4}] {gate:<{w}}  {detail}")
        print("=" * 72)
        print("RESULT: " + ("FAIL" if self.failed else "PASS"))
        return 1 if self.failed else 0


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_models(rep, model_dir):
    manifest = os.path.join(model_dir, "MODEL_MANIFEST.sha256")
    if not os.path.isfile(manifest):
        rep.add("models", "FAIL", f"no manifest at {manifest}")
        return
    bad, n = [], 0
    for line in open(manifest):
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        name = name.strip()
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            bad.append(f"{name}: missing")
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        n += 1
        if h.hexdigest() != digest:
            bad.append(f"{name}: digest mismatch")
    if bad:
        rep.add("models", "FAIL", "; ".join(bad))
    else:
        rep.add("models", "PASS", f"{n} model files match the manifest")


def gate_schema(rep, res, expected_clips):
    counts = {}
    for v, s in res.items():
        counts[len(s)] = counts.get(len(s), 0) + 1
    wrong = {k: c for k, c in counts.items() if k != expected_clips}
    detail = f"{len(res)} videos, clip counts {dict(sorted(counts.items()))}"
    if wrong:
        rep.add("schema", "FAIL", detail + f"; expected {expected_clips} clips per video")
    else:
        rep.add("schema", "PASS", detail)


def gate_accuracy(rep, res, labels, min_srocc):
    ids = [v for v in res if v in labels and res[v]]
    if len(ids) < 2:
        rep.add("accuracy", "FAIL", f"only {len(ids)} videos matched the labels file")
        return
    pred = [sum(res[v]) / len(res[v]) for v in ids]
    gt = [labels[v] for v in ids]
    s = srocc(pred, gt)
    rs = rescale(pred, gt)
    p = _pearson(rs, gt)
    k = krocc(pred, gt)
    rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(rs, gt)) / len(gt))
    detail = (f"n={len(ids)}  SROCC={s:.4f}  PLCC={p:.4f}  KROCC={k:.4f}  RMSE={rmse:.3f}"
              f"  (min SROCC {min_srocc:.4f})")
    rep.add("accuracy", "PASS" if s >= min_srocc else "FAIL", detail)


def gate_equivalence(rep, res, ref, tol_mos, margin):
    com = [v for v in res if v in ref and len(res[v]) == len(ref[v]) and res[v]]
    if len(com) < 2:
        rep.add("equivalence", "FAIL", f"only {len(com)} videos common to both runs")
        return
    clip_d = [abs(a - b) for v in com for a, b in zip(res[v], ref[v])]
    a = [sum(res[v]) / len(res[v]) for v in com]
    b = [sum(ref[v]) / len(ref[v]) for v in com]
    vid_d = [abs(x - y) for x, y in zip(a, b)]
    rho = srocc(a, b)
    flips = near = near_flips = 0
    for i in range(len(com)):
        for j in range(i + 1, len(com)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da * db < 0:
                flips += 1
            if abs(da) < margin:
                near += 1
                if da * db < 0:
                    near_flips += 1
    total = len(com) * (len(com) - 1) // 2
    ok = max(vid_d) <= tol_mos
    detail = (f"n={len(com)}  per-clip |d| mean={sum(clip_d)/len(clip_d):.3f} max={max(clip_d):.3f}"
              f"  per-video |d| mean={sum(vid_d)/len(vid_d):.3f} max={max(vid_d):.3f}"
              f"  (tolerance {tol_mos:.3f})")
    rep.add("equivalence", "PASS" if ok else "FAIL", detail)
    rep.add("rank agreement", "PASS" if rho >= 0.999 else "FAIL",
            f"Spearman={rho:.4f}  orderings flipped {flips}/{total} ({100*flips/total:.3f}%)")
    if near:
        pct = 100 * near_flips / near
        rep.add("near-tie flips", "INFO",
                f"pairs within {margin:.1f} MOS: {near_flips}/{near} flip ({pct:.1f}%)"
                f"  -> compare candidates on one platform, or treat gaps < {margin:.1f} MOS as ties")


def gate_pairwise(rep, res, pairs):
    have = [(u, s) for u, s in pairs if u in res and s in res]
    if not have:
        rep.add("pairwise", "FAIL", "no pair had both videos in the results file")
        return
    wrong = []
    margins = []
    for u, s in have:
        su = sum(res[u]) / len(res[u])
        ss = sum(res[s]) / len(res[s])
        margins.append(ss - su)
        if ss <= su:
            wrong.append(f"{s}<={u}")
    detail = (f"{len(have)-len(wrong)}/{len(have)} pairs ranked correctly"
              f"  mean margin {sum(margins)/len(margins):.1f} MOS")
    rep.add("pairwise", "PASS" if not wrong else "FAIL",
            detail + ("" if not wrong else "; " + ", ".join(wrong[:5])))


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="per-clip results CSV from a deployment target")
    ap.add_argument("--labels", help="ground-truth CSV with video_id,gt_score")
    ap.add_argument("--reference", help="results CSV from another platform, for equivalence")
    ap.add_argument("--pairs", help="CSV of unstable_id,stable_id synchronized pairs")
    ap.add_argument("--models", help="directory holding the ONNX files and MODEL_MANIFEST.sha256")
    ap.add_argument("--clips", type=int, default=4, help="expected clips per video (default 4)")
    ap.add_argument("--min-srocc", type=float, default=0.92,
                    help="accuracy gate threshold (default 0.92)")
    ap.add_argument("--tolerance-mos", type=float, default=1.0,
                    help="max per-video cross-platform difference allowed (default 1.0)")
    ap.add_argument("--tie-margin", type=float, default=1.0,
                    help="score gap below which two videos count as near-ties (default 1.0)")
    args = ap.parse_args()

    if not any([args.results, args.models]):
        ap.error("give at least --results or --models")

    rep = Report()
    if args.models:
        gate_models(rep, args.models)

    if args.results:
        res = load_clip_scores(args.results)
        if not res:
            raise SystemExit(f"{args.results}: no usable rows")
        gate_schema(rep, res, args.clips)
        if args.labels:
            gate_accuracy(rep, res, load_labels(args.labels), args.min_srocc)
        else:
            rep.add("accuracy", "SKIP", "no --labels given")
        if args.reference:
            gate_equivalence(rep, res, load_clip_scores(args.reference),
                             args.tolerance_mos, args.tie_margin)
        else:
            rep.add("equivalence", "SKIP", "no --reference given")
        if args.pairs:
            gate_pairwise(rep, res, load_pairs(args.pairs))
        else:
            rep.add("pairwise", "SKIP", "no --pairs given")

    return rep.emit()


if __name__ == "__main__":
    sys.exit(main())
