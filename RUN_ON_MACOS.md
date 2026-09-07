# Run StableVQA on a new MacBook

These instructions are for Apple Silicon Macs (M1/M2/M3/M4). They use the
native macOS implementation: three independent heavy branches run concurrently
with three threads each, while the motion branch uses eight threads. Keep these
defaults unchanged when comparing latency to this project’s Mac measurements.

## 1. Copy and open the package

Unzip `StableVQA-Edge.zip`, open Terminal, then enter the extracted
folder:

```sh
cd /path/to/StableVQA-Edge
```

All commands below must be run from this folder unless stated otherwise.

## 2. Install build tools and OpenCV

Install the Apple Command Line Tools if macOS asks for them, then install
Homebrew from https://brew.sh if it is not already installed. Run:

```sh
xcode-select --install
brew install cmake opencv
```

Confirm that Apple Silicon is being used:

```sh
uname -m
```

It should print `arm64`. An Intel Mac can use this CMake target too, but its
latency will not match the M2-specific measurements.

## 3. Use the bundled ONNX Runtime

The package now includes the tested ONNX Runtime 1.23.2 C/C++ runtime at
`third_party/onnxruntime-macos/`. **Do not create an empty directory in
`$HOME/Library/onnxruntime` and point CMake at it**; an empty directory has no
headers or `.dylib` library and causes the exact error shown above.

Before building, inspect the bundled runtime with this command:

```sh
ls -l third_party/onnxruntime-macos/include/onnxruntime_cxx_api.h \
  third_party/onnxruntime-macos/lib/libonnxruntime.dylib
```

It should print these two existing files (the following is an explanation, not
a command to run):

```text
third_party/onnxruntime-macos/include/onnxruntime_cxx_api.h
third_party/onnxruntime-macos/lib/libonnxruntime*.dylib
```

No Python `pip install onnxruntime` package is needed; this project uses the
bundled C/C++ headers and dynamic library.

## 4. Verify the bundled models

```sh
./onnx_models/verify_models.sh
```

All five commands must print `OK`. Stop if any checksum differs.

## 5. Build the macOS executable

For a Git clone or extracted release archive, run the self-locating setup
script. It derives every project path from its own location and always uses the
runtime bundled in the clone:

```sh
./scripts/setup_macos.sh
```

The only external prerequisites are Apple Command Line Tools, Homebrew, CMake,
and OpenCV. The script tells you exactly which missing dependency to install.

If you prefer the manual equivalent, run:

```sh
cmake -S . -B build-macos \
  -DSTABLEVQA_PLATFORM=MACOS
cmake --build build-macos --config Release -j "$(sysctl -n hw.ncpu)"
```

If you intentionally use another compatible ONNX Runtime installation, pass it
explicitly with `-DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime`. If CMake
says it cannot find OpenCV, run `brew --prefix opencv` and configure again with
`-DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"`.

## 6. Prepare input videos

Place one or more `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm` files in one
folder. For example:

```text
/Users/you/Videos/stability-test/
├── clip-01.mp4
└── clip-02.mov
```

The program selects four evenly spaced clips from each video and evaluates 32
frames per clip. It accepts a folder, not an individual video file.

## 7. Run inference

```sh
./build-macos/stablevqa /Users/you/Videos/stability-test onnx_models
```

Expected startup output includes:

```text
mode      : CONCURRENT (backbone|deblur|flow)
threads   : trio=3  motion=8
blur      : ON
Models loaded.
```

After the run, the current directory contains `stablevqa_results.csv` and
`stablevqa_calibration.csv`. The first contains per-clip scores, latency, and
memory fields; the second contains branch calibration values.

## 8. Troubleshooting

- **`Library not loaded: libonnxruntime...dylib`**: delete `build-macos`,
  re-run the Step 5 configure command, then rebuild. Do not point CMake at an
  empty `$HOME/Library/onnxruntime` directory.
- **`Model file not found`**: run from the package root and retain the final
  `onnx_models` argument.
- **No videos found**: pass the containing folder, not a single video path.
- **Cannot open a video**: re-encode the file to a codec supported by the local
  OpenCV/FFmpeg installation, then retry.
- **Latency differs from the project result**: verify `arm64`, use the listed
  concurrency defaults, let warm-up finish, and compare the same input video.
