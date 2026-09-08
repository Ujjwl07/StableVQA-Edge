# StableVQA deployment package

This standalone package runs the deployed five-model INT8 StableVQA pipeline.
It keeps the measured implementation for each target rather than forcing a
single generic code path: CUDA/TensorRT stream placement on Linux GPU, Apple
Silicon branch concurrency, Intel/AMD physical-core and P-core affinity, and
Exynos XNNPACK/MLAS selection and core pinning.

## Layout

```text
StableVQA-Edge/
├── platform/       native implementations selected by CMake
├── include/        x86 topology/affinity tuning
├── onnx_models/    five bundled, checksum-verified deployed INT8 models
├── android/        Android Studio app and JNI implementation
├── standalone_tool/ reproducible Windows one-folder PyInstaller recipe
├── scripts/        dataset launchers and x86 comparison runners
├── tests/          CPU tuning unit test
└── CMakeLists.txt  desktop build entry point
```

## Models

The five required models are bundled under `onnx_models/`; no placeholder URL
or download step is required. They are byte-identical to the copies embedded in
the Android app. Verify the files before benchmarking:

```sh
./onnx_models/verify_models.sh
```

The model exports derive from StableVQA (Kou et al., ACM MM 2023). We thank the
StableVQA authors for making their code and models publicly available. These
exports are used unmodified in learned weights. **Confirm redistribution terms
for the underlying StableVQA weights before publishing this package or a release
asset** — if redistribution is not permitted, replace the committed models with a
script that fetches the original weights and performs the export locally.

## Build from source

Install OpenCV and an ONNX Runtime binary distribution, then configure from the
package root. `ONNXRUNTIME_ROOT` must contain `include/` and `lib/`.

| Target | Configure command | Native execution provider and preserved strategy |
| --- | --- | --- |
| Linux x86-64 CPU | `cmake -S . -B build -DSTABLEVQA_PLATFORM=X86 -DONNXRUNTIME_ROOT=/path/to/ort` | ONNX Runtime CPU/MLAS; physical cores, hybrid P-core affinity, `SVQA_THREADS` override |
| Windows x86-64 | `cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DSTABLEVQA_PLATFORM=X86 -DONNXRUNTIME_ROOT=C:/ort` | ONNX Runtime CPU/MLAS; Windows topology detection and affinity |
| macOS Apple Silicon | `cmake -S . -B build -DSTABLEVQA_PLATFORM=MACOS` | CPU EP; bundled ONNX Runtime 1.23.2; 3×3-thread concurrent backbone/deblur/flow pools, 8-thread motion |
| Linux NVIDIA GPU | `cmake -S . -B build -DSTABLEVQA_GPU=ON -DONNXRUNTIME_ROOT=/path/to/ort-gpu` | CUDA EP, optional TensorRT and optional dual GPU controls in `StableVqaGpu.cpp` |

Then build with `cmake --build build --config Release`. The source trees used
ONNX Runtime 1.18.0 (GPU), 1.23.2 (macOS), and Android 1.19.2. The Linux GPU
path must use an ONNX Runtime GPU package compatible with its installed CUDA
and cuDNN; those versions are machine-specific and were not recorded in this
workspace, so they must be captured when validating the GPU release.

Run from the package root so the default `onnx_models/` directory resolves:

```sh
# macOS and CUDA programs accept a video folder and optional model directory
./build/stablevqa /path/to/videos onnx_models

# x86 program accepts a text file containing one video path per line
./build/stablevqa videos.txt stablevqa_results.csv
```

`scripts/run_dataset.py` makes those input conventions uniform. For example:

```sh
python3 scripts/run_dataset.py /path/to/videos --binary build/stablevqa \
  --platform macos --model-dir onnx_models
```

The results CSV is written by the selected native implementation. GPU and macOS
also write `stablevqa_calibration.csv`. No sample video was included in the
source workspace, so this package cannot truthfully provide a fixed expected
score; add a redistributable clip and expected result before a public release.

## Release gate

`scripts/release_gate.py` runs the full pre-release regression suite described in
the paper: model-manifest verification, clip-schema checks, accuracy against
ground truth (SROCC/PLCC/KROCC/RMSE), cross-platform score equivalence against a
reference run, and pairwise ranking on synchronized benchmarks. It has no
third-party dependencies and exits non-zero if a gate fails, so it can be wired
into CI.

```sh
# accuracy + model integrity
python3 scripts/release_gate.py --models onnx_models \
  --results samples/t4_validation_output/stablevqa_results.csv \
  --labels  samples/t4_validation_output/stabledb_val_labels.csv

# cross-platform equivalence of a new build against the reference T4 run
python3 scripts/release_gate.py --results my_run.csv \
  --reference samples/t4_validation_output/stablevqa_results.csv \
  --tolerance-mos 2.0 --tie-margin 1.0
```

Gates whose inputs are not supplied are reported as SKIP, never as PASS.

## Validation outputs

`samples/t4_validation_output/` holds the per-clip T4 output over the 391-video
StableDB validation split together with the ground-truth scores, so the paper's
accuracy numbers can be recomputed without a GPU (see the folder README).
`samples/android_expected_output/` holds a reference on-device run.

## Android

The Android tree is runnable and contains the Kotlin UI, JNI native pipeline,
Gradle files, and models. See [android/README_ANDROID.md](android/README_ANDROID.md).

## Fresh macOS setup

For a clean Apple Silicon MacBook, follow the complete
[RUN_ON_MACOS.md](RUN_ON_MACOS.md) guide. After installing Homebrew dependencies,
the portable one-command build is `./scripts/setup_macos.sh`; it resolves the
cloned repository directory and bundled macOS runtime automatically.

## Adaptive temporal inference

The deployed pipeline evaluates a fixed four clips per video and writes per-clip
scores to its results CSV. The paper's adaptive early-stopping method is provided
as an **offline analysis** over those per-clip scores, kept separate from the
real-time path so the deployed binary stays unchanged and auditable:

```sh
# single operating point (paper default: tau_rel=0.03, tau_std=2.0)
python3 scripts/adaptive_temporal_inference.py results.csv

# full threshold sweep (reproduces the paper's Table 11)
python3 scripts/adaptive_temporal_inference.py results.csv --sweep
```

The script implements the paper's OR-rule convergence criterion (Eq. 1) exactly.
With `--labels`, it also reports SROCC/PLCC/RMSE for the rule **and for the
fixed-k controls** (always score the first k clips), which is how the paper's
fixed-k comparison was produced:

```sh
python3 scripts/adaptive_temporal_inference.py \
  samples/t4_validation_output/stablevqa_results.csv \
  --labels samples/t4_validation_output/stabledb_val_labels.csv --sweep
```

The rule is not part of the deployed path. To make early exit part of the
real-time path instead, the same rule can be applied inside the per-clip loop in
`platform/*/StableVqa*.cpp`.

## Other platform guides

- [Windows x86-64](RUN_ON_WINDOWS.md)
- [Linux / NVIDIA GPU](RUN_ON_LINUX_GPU.md)
- [Android / Exynos](RUN_ON_ANDROID.md)

## Standalone end-user tool

The standalone tool covers the Windows x86-64 deployment with zero setup once
its documented bundle is built; all other platforms use the source build (and
the Android app for mobile). Prebuilt binaries are published as release assets rather than committed, so
`standalone_tool/` carries the reproducible one-folder PyInstaller recipe. See [standalone_tool/README_TOOL.md](standalone_tool/README_TOOL.md).

## Getting the models

The five subgraphs are published as a release asset rather than committed, so a
clone stays small. Download and unpack them into the repository root:

```bash
curl -L -o models.zip https://github.com/Ujjwl07/StableVQA-Edge/releases/download/v1.0.0/stablevqa-edge-models-v1.0.0.zip
unzip -q models.zip && rm models.zip
```

This creates `onnx_models/` with the ten `.onnx` files. For the Android build,
copy the five `*_quant.onnx` files into
`android/app/src/main/assets/` before assembling the APK.

## Tested platforms

| Target | Package content | Status |
| --- | --- | --- |
| Linux x86 CPU | CMake source | Tested |
| Windows x86 CPU | CMake source and packaging recipe | Tested |
| macOS Apple Silicon | CMake source | Tested |
| Linux NVIDIA GPU | CMake source | Tested |
| Android / Exynos | Full Gradle/JNI application | Tested |

## Baseline StableVQA Citation
```bibtex
  @inproceedings{kou2023stablevqa,
  title={Stablevqa: A deep no-reference quality assessment model for video stability},
  author={Kou, Tengchuan and Liu, Xiaohong and Sun, Wei and Jia, Jun and Min, Xiongkuo and Zhai, Guangtao and Liu, Ning},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={1066--1076},
  year={2023}
}
```
