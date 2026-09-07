# StableVQA Android application

This is the runnable Android application used for the Exynos 2400 path, not a
desktop CMake adapter. It embeds the same five SHA-256-verified INT8 ONNX
models as the desktop package in `app/src/main/assets/`.

The JNI pipeline in `app/src/main/cpp/native-lib.cpp` preserves the measured
Exynos configuration: CPUs 4–9 are selected, six intra-op threads are used,
XNNPACK is attempted per model then safely falls back to MLAS for unsupported
quantized attention graphs, and concurrent heavy branches remain disabled
because they did not improve latency on this SoC. The Kotlin activity decodes
four clips per video, records memory/battery measurements, and writes CSV data
to Downloads.

Build with Android Studio or `./gradlew assembleDebug` using JDK 11, Android
SDK 36, NDK 21.4.7075529, CMake 3.22.1, and ONNX Runtime Android 1.19.2. The
Gradle task extracts ONNX Runtime headers and ABI libraries from the declared
AAR before native compilation. Supported ABIs are configured by the installed
ONNX Runtime AAR; validate on the target Exynos device before reporting timing.
