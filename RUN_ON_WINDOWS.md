# Run StableVQA on Windows x86-64

This guide builds and runs the optimized Intel/AMD CPU path for StableVQA. The program automatically detects the CPU architecture at runtime, utilizes physical cores (excluding SMT siblings), and pins Intel hybrid architectures (12th Gen+) to P-cores for maximum throughput. It requires an AVX2-capable x86-64 CPU (Intel Haswell 2013+ / AMD Zen 2017+).

---

## 1. Install Prerequisites

### A. Visual Studio 2022 / Build Tools
Install **Visual Studio 2022** (Community, Professional, or Build Tools) with the **Desktop development with C++** workload:
- Required components: **MSVC v143 - VS 2022 C++ x64/x86 build tools** and the **Windows 10/11 SDK**.
- [Download Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)

### B. CMake & Git
Install CMake (>= 3.20) and Git:
```powershell
winget install Kitware.CMake Git.Git
```
*(Or download from [cmake.org](https://cmake.org/download/) and [git-scm.com](https://git-scm.com/download/win). Ensure they are added to your system `PATH`.)*

### C. ONNX Runtime (C/C++ Windows x64 CPU)
1. Download **`onnxruntime-win-x64-1.18.0.zip`** from [Microsoft ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.0).
2. Extract the archive to `C:\libraries\onnxruntime-win-x64` (so that `include\onnxruntime_cxx_api.h` and `lib\onnxruntime.lib` exist).

### D. OpenCV (Pre-built Windows x64)
1. Download **`opencv-4.10.0-windows.exe`** (or 4.x) from [OpenCV Releases](https://github.com/opencv/opencv/releases/tag/4.10.0).
2. Run the self-extracting archive and extract to `C:\libraries\opencv`.
*(Note: OpenCV pre-built binaries for Visual Studio 2019/2022 are located in `build\x64\vc16\`)*.

---

## 2. Verify Models

Open PowerShell in the repository root and verify that all five bundled quantized models match their SHA256 checksums:

```powershell
.\onnx_models\verify_models.ps1
```

All five models must report `OK`:
```text
OK backbone_quant.onnx
OK deblur_net_quant.onnx
OK flow_model_quant.onnx
OK motion_analyzer_quant.onnx
OK quality_head_quant.onnx
```

---

## 3. Configure and Build

Open **x64 Native Tools Command Prompt for VS 2022** (or in standard PowerShell, run `& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`), navigate to the cloned directory, and run:

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 `
  -DSTABLEVQA_PLATFORM=X86 `
  -DONNXRUNTIME_ROOT="C:\libraries\onnxruntime-win-x64" `
  -DOpenCV_DIR="C:\libraries\opencv\build\x64\vc16\lib"

cmake --build build-win --config Release --parallel
```

> **Tip**: CMake automatically copies `onnxruntime.dll` beside `stablevqa.exe` upon building.

---

## 4. Set Up Runtime DLLs

Make sure the OpenCV runtime binaries are in `PATH` or copied into the executable directory:

```powershell
# Option A: Add to PATH for the current session
$env:PATH = "C:\libraries\opencv\build\x64\vc16\bin;C:\libraries\onnxruntime-win-x64\lib;$env:PATH"

# Option B: Copy DLLs directly into the Release folder
Copy-Item "C:\libraries\opencv\build\x64\vc16\bin\*.dll" "build-win\Release\" -Force
Copy-Item "C:\libraries\onnxruntime-win-x64\lib\*.dll" "build-win\Release\" -Force
```

---

## 5. Verify CPU Topology

Run the standalone CPU tuning unit test to verify topology detection and thread selection on your hardware:

```powershell
.\build-win\Release\test_cpu_tuning.exe
```

Expected output confirms detection of physical cores and hybrid P-cores (e.g., Intel 12th+ Gen):
```text
=== detectCpu() on THIS machine ===
vendor=Intel  brand="12th Gen Intel(R) Core(TM) i5-12500H"  physical=12  logical=16  hybrid=yes  P-cores=4
would choose: threads=4  profile=Intel-hybrid (P-cores only)

7 passed, 0 failed
```

---

## 6. Prepare Input & Run StableVQA

The x86 program accepts a text file containing **one video path per line** (absolute or relative):

Create `videos.txt`:
```text
C:\Users\you\Videos\sample1.mp4
C:\Users\you\Videos\sample2.mov
```

Run the pipeline from the repository root (so `onnx_models\` is located automatically):

```powershell
.\build-win\Release\stablevqa.exe videos.txt stablevqa_results.csv
```

### Output
The output CSV (`stablevqa_results.csv`) records:
- Quality MOS score per clip (12.5%, 37.5%, 62.5%, 87.5% temporal anchors)
- Detailed latency breakdown per model (`Backbone`, `Deblur`, `Flow`, `Motion`, `QualityHead`)
- Memory usage (RSS) and RAM delta
- Overall video mean quality score and standard deviation

---

## 7. Tuning & Comparison Scripts

Pre-configured benchmark scripts in `scripts\` evaluate affinity and thread strategies:
- `scripts\run1_ours.bat` — Recommended auto-detected physical/P-core topology
- `scripts\run2_pe_allphysical.bat` — All physical cores (disabling P-core restriction)
- `scripts\run3_fixed8.bat` — Fixed 8-worker thread pool
- `scripts\run4_all_logical.bat` — All logical SMT threads

Environment variables for manual experiments:
- **Set thread count**: `$env:SVQA_THREADS = "8"`
- **Disable CPU affinity**: `$env:SVQA_NO_AFFINITY = "1"`
- **Reset**: `Remove-Item Env:SVQA_THREADS; Remove-Item Env:SVQA_NO_AFFINITY`

---

## 8. Troubleshooting

| Issue | Cause & Solution |
|---|---|
| `The requested API version [18] is not available...` | Windows loaded an older `onnxruntime.dll` from `C:\Windows\System32`. **Solution**: Copy `C:\libraries\onnxruntime-win-x64\lib\onnxruntime.dll` directly into `build-win\Release\` so it takes precedence. |
| `Could not find a package configuration file provided by "OpenCV"` | Ensure `-DOpenCV_DIR` points to `.../build/x64/vc16/lib` where `OpenCVConfig.cmake` resides. |
| `Cannot open video list file` | Run `stablevqa.exe` from the repository root or provide a valid path to `videos.txt`. |
| `Skipping (cannot open): video.mp4` | OpenCV could not decode the video format. Ensure `opencv_videoio_ffmpeg*.dll` is in `PATH` or beside `stablevqa.exe`. |