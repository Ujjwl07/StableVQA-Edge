# Build and validation guide

## Common desktop prerequisites

- CMake 3.20 or later and a C++17 compiler.
- OpenCV development package discoverable by `find_package(OpenCV)`.
- ONNX Runtime binary package with matching headers and library. Pass its root
  with `-DONNXRUNTIME_ROOT=...`.
- Run the executable from the package root, or pass `onnx_models` explicitly
  to the Mac/GPU programs.

## Linux CPU

```sh
cmake -S . -B build-cpu -DSTABLEVQA_PLATFORM=X86 -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build-cpu -j
./build-cpu/test_cpu_tuning
./build-cpu/stablevqa videos.txt stablevqa_results.csv
```

Use `SVQA_THREADS=N` to compare a fixed worker count. On Windows, the supplied
`scripts/run*.bat` files reproduce the P-core, physical-core, fixed-eight, and
all-logical comparisons.

## macOS

```sh
cmake -S . -B build-macos -DSTABLEVQA_PLATFORM=MACOS -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build-macos -j
./build-macos/stablevqa /path/to/videos onnx_models
```

The source intentionally enables the measured M2 concurrent mode. Do not use a
generic thread count when collecting comparable latency measurements.

## Linux NVIDIA GPU

```sh
cmake -S . -B build-gpu -DSTABLEVQA_GPU=ON -DONNXRUNTIME_ROOT=/opt/onnxruntime-gpu
cmake --build build-gpu -j
./build-gpu/stablevqa /path/to/videos onnx_models
```

Use an ONNX Runtime GPU package containing CUDA provider support. CUDA and
cuDNN must match the selected package. The GPU source retains its optional
TensorRT, concurrent-stream, and dual-GPU toggles; benchmark a configuration
only after explicitly recording those values.

## Windows

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 `
  -DSTABLEVQA_PLATFORM=X86 -DONNXRUNTIME_ROOT=C:\onnxruntime
cmake --build build-win --config Release
.\build-win\Release\stablevqa.exe videos.txt stablevqa_results.csv
```

Ensure `onnxruntime.dll` and OpenCV DLLs are available through `PATH` or copied
beside the executable. The native code supports UTF-16 path conversion where
required and executes the same tuned MLAS path as Linux x86.

## Android

Open `android/` in Android Studio or run `./gradlew assembleDebug` from that
directory. Details and pins are in `android/README_ANDROID.md`.

## Release gate

Before publishing, run `onnx_models/verify_models.sh`, build every advertised
target from a clean build directory, and run one common video through each.
Record scores, toolchain versions, provider loading, and the exact GPU toggle
state. The package has source and models but no bundled sample video or
prebuilt Windows release asset.
