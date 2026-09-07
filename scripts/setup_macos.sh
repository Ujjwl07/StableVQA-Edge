#!/usr/bin/env bash
# Build StableVQA from any cloned/copied location on an Apple Silicon Mac.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_DIR="$ROOT_DIR/third_party/onnxruntime-macos"
BUILD_DIR="$ROOT_DIR/build-macos"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This setup script is for macOS."
  exit 1
fi
if [[ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" || ! -d "$ORT_DIR/lib" ]]; then
  echo "Bundled macOS ONNX Runtime is missing: $ORT_DIR"
  echo "Use a complete clone/release archive; do not copy only selected source files."
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake is required. Install it with: brew install cmake"
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew and OpenCV are required. Install Homebrew, then run: brew install opencv"
  exit 1
fi
if ! brew list --versions opencv >/dev/null 2>&1; then
  echo "OpenCV is required. Install it with: brew install opencv"
  exit 1
fi

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DSTABLEVQA_PLATFORM=MACOS \
  -DONNXRUNTIME_ROOT="$ORT_DIR"
BUILD_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" --config Release -j "$BUILD_JOBS"

echo
echo "Build complete: $BUILD_DIR/stablevqa"
echo "Run: $BUILD_DIR/stablevqa /path/to/video-folder $ROOT_DIR/onnx_models"
