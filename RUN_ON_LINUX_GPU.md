# Run StableVQA on Linux with an NVIDIA GPU

This guide builds and runs the CUDA implementation of StableVQA on a Linux x86-64 system with an NVIDIA GPU.

The Linux GPU implementation uses ONNX Runtime's CUDA Execution Provider and retains the project's optional TensorRT fallback and concurrent-stream controls in `platform/gpu/StableVqaGpu.cpp`.

## 1. Prerequisites

### Hardware and drivers

- Linux x86-64
- NVIDIA GPU with a working NVIDIA driver

Confirm the GPU and the CUDA toolkit version:

```bash
nvidia-smi
nvcc --version
```

Note the CUDA **major** version that `nvcc` reports. Any CUDA 12.x release works with the runtime pinned below; the configuration in this guide was validated on CUDA 12.8.

### Software

- CUDA **12.8**
- cuDNN **9.8.0**
- CMake **3.20+**
- A C++17 compiler
- OpenCV development files
- `wget` or `curl`, and `unzip`

The shipped Linux GPU configuration uses ONNX Runtime **1.20.1 GPU**, which is built for CUDA 12.8 and cuDNN 9.8.0.

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev wget unzip
```

Install CUDA and cuDNN using NVIDIA's official instructions for your distribution.

## 2. Clone the repository

```bash
git clone https://github.com/Ujjwl07/StableVQA-Edge.git
cd StableVQA-Edge
```

## 3. Download the model weights

The ONNX weights are distributed as a GitHub Release asset rather than committed to the repository, so a fresh clone does **not** contain them. Download and unpack them into the repository root:

```bash
wget -q https://github.com/Ujjwl07/StableVQA-Edge/releases/download/v1.0.0/stablevqa-edge-models-v1.0.0.zip
unzip -q stablevqa-edge-models-v1.0.0.zip && rm stablevqa-edge-models-v1.0.0.zip
```

This populates `onnx_models/` with ten files, a static-INT8 (`*_quant.onnx`) and an FP16 (`*_fp16.onnx`) export of each of the five subgraphs:

```text
backbone_quant.onnx        backbone_fp16.onnx
deblur_net_quant.onnx      deblur_net_fp16.onnx
flow_model_quant.onnx      flow_model_fp16.onnx
motion_analyzer_quant.onnx motion_analyzer_fp16.onnx
quality_head_quant.onnx    quality_head_fp16.onnx
```

Verify their checksums against the shipped manifest:

```bash
chmod +x ./onnx_models/verify_models.sh
./onnx_models/verify_models.sh
```

## 4. Install the ONNX Runtime GPU C/C++ package

StableVQA is a C++ application, so it needs the ONNX Runtime **C/C++** GPU binary package, not the Python `onnxruntime-gpu` wheel.

```bash
wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz
rm -rf onnxruntime-gpu
mv onnxruntime-linux-x64-gpu-1.20.1 onnxruntime-gpu
```

Sanity-check the extraction before building:

```bash
ls onnxruntime-gpu/include/onnxruntime_cxx_api.h
ls onnxruntime-gpu/lib/libonnxruntime_providers_cuda.so
ldd onnxruntime-gpu/lib/libonnxruntime_providers_cuda.so | grep "not found"
```

The `ldd` line should print nothing. If it lists missing libraries, this ONNX Runtime build does not match your installed CUDA toolkit; choose the release that matches instead of proceeding.

## 5. Configure and build

From the repository root:

```bash
rm -rf build-gpu
cmake -S . -B build-gpu \
  -DSTABLEVQA_GPU=ON \
  -DONNXRUNTIME_ROOT="$(pwd)/onnxruntime-gpu"
cmake --build build-gpu --config Release -j "$(nproc)"
```

If CUDA is installed outside the default loader path, export it before running:

```bash
export LD_LIBRARY_PATH="$(pwd)/onnxruntime-gpu/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

## 6. Run inference

Place videos in one directory. Supported extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

```bash
./build-gpu/stablevqa /path/to/video-folder onnx_models
```

At start-up the program reports the bound provider and the models it loaded:

```text
provider  : CUDA
gpu mode  : SINGLE (all branches->GPU0)
```

Results are written to the current working directory:

```text
stablevqa_results.csv
stablevqa_calibration.csv
```

## 7. Precision and scheduling options

The compile-time controls are at the top of `platform/gpu/StableVqaGpu.cpp`:

```text
USE_CONCURRENT   (shipped: false) — false runs the three heavy branches
                                    sequentially and reports per-branch
                                    latencies that are not inflated by stream
                                    contention. Set true to dispatch
                                    backbone/blur/flow on separate CUDA
                                    streams (~492 ms per clip on a T4,
                                    against ~513 ms sequential).
USE_TENSORRT     (shipped: false) — optional TensorRT EP ahead of CUDA.
```

Precision is selected per branch by the filename constants in the same file (`FLOW_MODEL`, `BACKBONE_MODEL`, `DEBLUR_MODEL`, `MOTION_MODEL`, `HEAD_MODEL`). The shipped T4 configuration is static INT8 for the flow branch and FP16 for the other four:

```text
FLOW_MODEL     = flow_model_quant.onnx
BACKBONE_MODEL = backbone_fp16.onnx
DEBLUR_MODEL   = deblur_net_fp16.onnx
MOTION_MODEL   = motion_analyzer_fp16.onnx
HEAD_MODEL     = quality_head_fp16.onnx
```

Changing any of these produces a different benchmark configuration. Rebuild, record the new values, and record the model filenames printed at start-up alongside any published number. Do not compare latencies across configurations without documenting the difference.

## 8. Correctness check

Run the same video through the x86 CPU and Linux GPU builds and compare the scores. Small CPU/GPU numerical differences are expected; the project's cross-platform tolerance is 1.3 MOS per video, enforced by:

```bash
python3 scripts/release_gate.py --results gpu_run.csv --reference cpu_run.csv
```

## 9. Reproducibility

For reproducible GPU benchmarks, record:

```bash
nvidia-smi
nvcc --version
cmake --version
gcc --version
cat onnxruntime-gpu/VERSION_NUMBER
```

along with `USE_CONCURRENT`, `USE_TENSORRT`, and the five model filename constants from `platform/gpu/StableVqaGpu.cpp`. CUDA, cuDNN, ONNX Runtime, GPU model, driver version, and the build-time toggles all affect runtime performance.

## 10. Expected directory layout

```text
StableVQA-Edge/
├── CMakeLists.txt
├── cmake/
│   └── FindONNXRuntime.cmake
├── onnx_models/                      # populated by step 3
│   ├── backbone_fp16.onnx
│   ├── backbone_quant.onnx
│   ├── deblur_net_fp16.onnx
│   ├── deblur_net_quant.onnx
│   ├── flow_model_fp16.onnx
│   ├── flow_model_quant.onnx
│   ├── motion_analyzer_fp16.onnx
│   ├── motion_analyzer_quant.onnx
│   ├── quality_head_fp16.onnx
│   ├── quality_head_quant.onnx
│   ├── MODEL_MANIFEST.sha256
│   └── verify_models.sh
├── platform/
│   └── gpu/
│       └── StableVqaGpu.cpp
├── onnxruntime-gpu/                  # downloaded in step 4, not committed
│   ├── include/
│   └── lib/
└── build-gpu/
    └── stablevqa
```

## 11. Quick start

For a system that already has an NVIDIA driver, CUDA 12.8, cuDNN 9.8.0, CMake, a C++17 compiler, and OpenCV development files:

```bash
git clone https://github.com/Ujjwl07/StableVQA-Edge.git
cd StableVQA-Edge

wget -q https://github.com/Ujjwl07/StableVQA-Edge/releases/download/v1.0.0/stablevqa-edge-models-v1.0.0.zip
unzip -q stablevqa-edge-models-v1.0.0.zip && rm stablevqa-edge-models-v1.0.0.zip
chmod +x ./onnx_models/verify_models.sh
./onnx_models/verify_models.sh

wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz
rm -rf onnxruntime-gpu
mv onnxruntime-linux-x64-gpu-1.20.1 onnxruntime-gpu

rm -rf build-gpu
cmake -S . -B build-gpu \
  -DSTABLEVQA_GPU=ON \
  -DONNXRUNTIME_ROOT="$(pwd)/onnxruntime-gpu"
cmake --build build-gpu --config Release -j "$(nproc)"

./build-gpu/stablevqa /path/to/video-folder onnx_models
```

## Compatibility note

This guide pins the Linux GPU runtime to ONNX Runtime **1.20.1** rather than tracking the latest release. The project has a C++ integration and an optional TensorRT path, so changing the ONNX Runtime version is a compatibility change and should be revalidated with `scripts/release_gate.py`. Consult the official ONNX Runtime CUDA Execution Provider documentation for the current compatibility matrix before repinning.
