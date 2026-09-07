# Windows x86-64 standalone tool

This is a reproducible **one-folder PyInstaller recipe**, not a prebuilt binary.
The supplied workspace contains no historical PyInstaller `dist/` artifact, so
claiming one is available would be inaccurate. Build `stablevqa.exe` with the
top-level MSVC x86 command, place it in this directory, and run
`pyinstaller --noconfirm stablevqa_tool.spec` on Windows. The spec bundles that
native executable and the five byte-verified models from `../onnx_models/`.

Run the generated `dist/StableVQATool/StableVQATool.exe <video-list.txt>
[output.csv]`. It writes the C++ CSV output in the bundle directory. It requires
Windows x86-64 with AVX2 (roughly 2013+ Intel or 2015+ AMD), matching the
`/favor:INTEL64` / native-optimized source configuration.

| Target | Standalone tool | Alternative |
| --- | --- | --- |
| Windows x86-64 | Yes, after producing the documented one-folder bundle | — |
| Linux x86-64 | No | Build source with `STABLEVQA_PLATFORM=X86` |
| macOS | No | Build source with `STABLEVQA_PLATFORM=MACOS` |
| NVIDIA CUDA | No | Build source with `STABLEVQA_GPU=ON` |
| Android / Exynos | No | Build `android/` app |

The bundle must use the model files in this repository; verify them with
`onnx_models/verify_models.ps1` before packaging. A release asset can be added
once it has been built and tested on the intended Windows hardware.
