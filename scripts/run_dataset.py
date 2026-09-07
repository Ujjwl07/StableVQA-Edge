#!/usr/bin/env python3
"""Run a StableVQA native binary over a dataset without changing its platform path."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Folder of videos")
    parser.add_argument("--binary", type=Path, required=True, help="Built stablevqa executable")
    parser.add_argument("--platform", choices=("x86", "macos", "gpu"), required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("onnx_models"))
    parser.add_argument("--output", type=Path, default=Path("stablevqa_results.csv"))
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"dataset is not a directory: {args.dataset}")
    if not args.binary.is_file():
        parser.error(f"binary not found: {args.binary}")
    if not args.model_dir.is_dir():
        parser.error(f"model directory not found: {args.model_dir}")

    if args.platform == "x86":
        video_list = args.output.with_suffix(".videos.txt")
        videos = sorted(p for p in args.dataset.iterdir()
                        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"})
        video_list.write_text("".join(f"{p.resolve()}\\n" for p in videos), encoding="utf-8")
        command = [str(args.binary), str(video_list), str(args.output)]
    else:
        # Both Mac and CUDA programs accept a video directory then a model directory.
        command = [str(args.binary), str(args.dataset), str(args.model_dir)]

    print("Running:", " ".join(command))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
