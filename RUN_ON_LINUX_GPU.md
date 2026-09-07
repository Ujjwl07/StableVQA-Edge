# Run StableVQA on Linux with an NVIDIA GPU

This guide builds and runs the CUDA implementation of StableVQA on a Linux x86-64 system with an NVIDIA GPU.

The Linux GPU implementation uses ONNX Runtime's CUDA Execution Provider and retains the project's optional TensorRT fallback, concurrent-stream controls, and dual-GPU placement controls in `platform/gpu/StableVqaGpu.cpp`.

## 1. Prerequisites

### Hardware and drivers

- Linux x86-64
- NVIDIA GPU with a working NVIDIA driver
- `nvidia-smi` must succeed

Verify:

```bash
nvidia-smi
```

### Software

Install:

- CUDA **12.x**
- cuDNN **9.x**
- CMake **3.20+**
- A C++17 compiler
- OpenCV development files
- `curl` or `wget`
- `tar`

The shipped Linux GPU configuration uses ONNX Runtime **1.20.1 GPU**, which matches the CUDA 12.x + cuDNN 9.x stack. ONNX Runtime's compatibility table lists the 1.20.x GPU packages under CUDA 12.x and cuDNN 9.x.

For Debian/Ubuntu, install the general build tools and OpenCV with:

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev curl tar
```

Install CUDA and cuDNN using NVIDIA's official installation instructions for your Linux distribution.

> **Important:** Do not use an ONNX Runtime GPU package built for CUDA 11.x/cuDNN 8.x with this documented configuration. CUDA and cuDNN major-version mismatches can prevent the CUDA Execution Provider from loading.

## 2. Clone and verify the repository

Clone the repository and enter it:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
cd StableVQA-Edge
```

Make sure the model files are present:

```bash
ls -lh onnx_models/
```

The repository expects these five quantized models:

```text
backbone_quant.onnx
deblur_net_quant.onnx
flow_model_quant.onnx
motion_analyzer_quant.onnx
quality_head_quant.onnx
```

Verify their checksums before running:

```bash
chmod +x onnx_models/verify_models.sh
./onnx_models/verify_models.sh
```

All five model checksums should pass.

If the script is not executable, the `chmod +x` command above fixes the permission.

## 3. Verify CUDA

Confirm that the NVIDIA driver and CUDA compiler are available:

```bash
nvidia-smi
nvcc --version
```

For this configuration, `nvcc` should report CUDA 12.x.

Also confirm that CUDA runtime libraries are available:

```bash
ls /usr/local/cuda/lib64/libcudart.so*
ls /usr/local/cuda/lib64/libcublas.so*
ls /usr/local/cuda/lib64/libcublasLt.so*
```

If CUDA is installed somewhere other than `/usr/local/cuda`, use that installation's `lib64` directory in the runtime library path described below.

## 4. Install the ONNX Runtime GPU C/C++ package

StableVQA is a C++ application, so you need the **ONNX Runtime C/C++ GPU binary package**, not only the Python `onnxruntime-gpu` package.

This guide pins the tested Linux GPU runtime to ONNX Runtime **1.20.1**:

```bash
export ORT_VERSION=1.20.1
export ORT_ROOT="$PWD/onnxruntime-gpu"
```

Download and extract it:

```bash
curl -L \
  -o "onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"

tar -xzf "onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"

rm -rf "$ORT_ROOT"
mv "onnxruntime-linux-x64-gpu-${ORT_VERSION}" "$ORT_ROOT"
```

Verify that the required C++ headers and libraries exist:

```bash
test -f "$ORT_ROOT/include/onnxruntime_cxx_api.h"
test -f "$ORT_ROOT/lib/libonnxruntime.so"
test -f "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so"
```

You can also inspect the CUDA provider dependencies:

```bash
ldd "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so" | grep "not found"
```

**Expected result:** no output.

If libraries are reported as `not found`, fix the CUDA/cuDNN installation or library search path before building StableVQA.

## 5. Configure and build StableVQA

From the repository root:

```bash
rm -rf build-gpu

cmake -S . -B build-gpu \
  -DSTABLEVQA_GPU=ON \
  -DONNXRUNTIME_ROOT="$ORT_ROOT"

cmake --build build-gpu --config Release -j "$(nproc)"
```

The CMake project searches the supplied ONNX Runtime root for:

```text
include/onnxruntime_cxx_api.h
lib/libonnxruntime.so
```

and links the application against ONNX Runtime.

## 6. Set the runtime library path

At runtime, Linux must be able to locate ONNX Runtime and the CUDA libraries.

For a standard CUDA installation under `/usr/local/cuda`:

```bash
export LD_LIBRARY_PATH="$ORT_ROOT/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

If cuDNN is installed in a directory that is not already known to the dynamic linker, add that directory as well.

Verify the ONNX Runtime CUDA provider's dependencies:

```bash
ldd "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so" | grep "not found"
```

No output should be produced.

## 7. Verify models and run inference

Put supported videos in one directory.

Supported extensions include:

```text
.mp4
.avi
.mov
.mkv
.webm
```

Run StableVQA from the repository root:

```bash
./build-gpu/stablevqa /path/to/video-folder onnx_models
```

Example:

```bash
./build-gpu/stablevqa ./data/videos onnx_models
```

At startup, the program should report that the provider is CUDA, for example:

```text
provider  : CUDA
gpu mode  : SINGLE (all branches->GPU0)
```

The GPU program writes:

```text
stablevqa_results.csv
stablevqa_calibration.csv
```

to the current working directory.

## 8. Multiple GPUs and optimization modes

The GPU implementation contains optional controls for GPU placement and optimization in:

```text
platform/gpu/StableVqaGpu.cpp
```

The relevant controls are:

```text
USE_CONCURRENT   (shipped: false) — false runs the three heavy branches one
                                    after another and reports per-branch
                                    latencies that are NOT inflated by stream
                                    contention. Set to true to dispatch
                                    backbone/blur/flow on separate CUDA streams;
                                    this is the configuration behind the paper's
                                    headline per-clip latency (~0.541 s).
USE_TENSORRT     (shipped: false) — optional TensorRT EP before CUDA.
USE_DUAL_GPU     (shipped: false) — split branches across two GPUs
                                    (paper: ~0.335 s per clip on 2x T4).
```

Model precision is selected by the filename constants at the top of the same
file (`FLOW_MODEL`, `BACKBONE_MODEL`, `DEBLUR_MODEL`, `MOTION_MODEL`,
`HEAD_MODEL`). Both `*_quant.onnx` (static INT8) and `*_fp16.onnx` exports of
every branch are bundled in `onnx_models/`. The paper's T4 configuration is
INT8 for the flow model and FP16 for the other four branches; record the
filenames actually loaded (they are printed at start-up) with any published
number.

If you change any of these options:

1. Record the new values.
2. Rebuild the project.
3. Treat the result as a different benchmark configuration.

Do not compare latency numbers from different optimization configurations without documenting the differences.

## 9. Correctness check

For a correctness validation, run the same video through the x86 CPU and Linux GPU builds and compare their output scores.

Small numerical differences can occur between CPU and GPU execution. Use the tolerance documented by the project when determining whether the results agree.

## 10. Troubleshooting

### `Permission denied` when running `verify_models.sh`

Run:

```bash
chmod +x onnx_models/verify_models.sh
./onnx_models/verify_models.sh
```

Or invoke it directly with Bash:

```bash
bash onnx_models/verify_models.sh
```

### `ONNXRUNTIME_ROOT is not a valid ONNX Runtime C/C++ directory`

Check:

```bash
ls "$ORT_ROOT/include/onnxruntime_cxx_api.h"
ls "$ORT_ROOT/lib/libonnxruntime.so"
```

If either file is missing, the ONNX Runtime package was not extracted correctly or the wrong directory was supplied.

### `libonnxruntime_providers_cuda.so` fails to load

Run:

```bash
ldd "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so" | grep "not found"
```

If CUDA libraries are missing, check:

```bash
ls /usr/local/cuda/lib64/
```

and set:

```bash
export LD_LIBRARY_PATH="$ORT_ROOT/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

Do **not** create fake compatibility symlinks such as mapping `libcublasLt.so.12` to `libcublasLt.so.11`. CUDA and cuDNN major-version mismatches can cause ABI incompatibilities or crashes.

### `nvidia-smi` works but CUDA inference fails

`nvidia-smi` only verifies that the NVIDIA driver can communicate with the GPU. It does not guarantee that the CUDA/cuDNN libraries required by ONNX Runtime are installed and discoverable.

Check:

```bash
nvcc --version
ldd "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so" | grep "not found"
```

### CMake says `Makefile: No rule to make target`

This normally means the CMake configuration failed before generating build files.

Remove the failed build directory and configure again:

```bash
rm -rf build-gpu

cmake -S . -B build-gpu \
  -DSTABLEVQA_GPU=ON \
  -DONNXRUNTIME_ROOT="$ORT_ROOT"
```

Only run `cmake --build` after configuration completes successfully.

## 11. Reproducibility

For reproducible GPU benchmarks, record at least:

```bash
nvidia-smi
nvcc --version
cmake --version
gcc --version
```

and the ONNX Runtime version:

```bash
cat "$ORT_ROOT/VERSION_NUMBER"
```

Also record:

```text
USE_CONCURRENT
USE_TENSORRT
USE_DUAL_GPU
```

from `platform/gpu/StableVqaGpu.cpp`.

This is important because CUDA, cuDNN, ONNX Runtime, GPU model, driver version, and StableVQA optimization toggles can all affect runtime performance.

## 12. Expected directory layout

After setup, the relevant repository layout should look like:

```text
StableVQA-Edge/
├── CMakeLists.txt
├── cmake/
│   └── FindONNXRuntime.cmake
├── onnx_models/
│   ├── backbone_quant.onnx
│   ├── deblur_net_quant.onnx
│   ├── flow_model_quant.onnx
│   ├── motion_analyzer_quant.onnx
│   ├── quality_head_quant.onnx
│   └── verify_models.sh
├── platform/
│   └── gpu/
│       └── StableVqaGpu.cpp
├── onnxruntime-gpu/
│   ├── include/
│   └── lib/
└── build-gpu/
    └── stablevqa
```

The `onnxruntime-gpu/` directory is a downloaded third-party runtime and should **not** be committed to the repository. Users download it during setup.

## 13. Quick start

For a Linux system that already has a working NVIDIA driver, CUDA 12.x, cuDNN 9.x, CMake, a C++17 compiler, and OpenCV development files:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
cd StableVQA-Edge

chmod +x onnx_models/verify_models.sh
./onnx_models/verify_models.sh

export ORT_VERSION=1.20.1
export ORT_ROOT="$PWD/onnxruntime-gpu"

curl -L \
  -o "onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"

tar -xzf "onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
rm -rf "$ORT_ROOT"
mv "onnxruntime-linux-x64-gpu-${ORT_VERSION}" "$ORT_ROOT"

export LD_LIBRARY_PATH="$ORT_ROOT/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cmake -S . -B build-gpu \
  -DSTABLEVQA_GPU=ON \
  -DONNXRUNTIME_ROOT="$ORT_ROOT"

cmake --build build-gpu --config Release -j "$(nproc)"

./build-gpu/stablevqa /path/to/video-folder onnx_models
```

## Compatibility note

This README intentionally pins the documented Linux GPU runtime to ONNX Runtime **1.20.1** rather than telling users to install an arbitrary latest GPU runtime. The project has a C++ integration and optional TensorRT path, so changing the ONNX Runtime version should be treated as a compatibility change and revalidated.

For the current ONNX Runtime CUDA compatibility matrix, consult the official ONNX Runtime CUDA Execution Provider documentation before changing the pinned version.
