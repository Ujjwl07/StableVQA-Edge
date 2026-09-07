# Run StableVQA on Android / Exynos

This is the full Android application, not a desktop binary. It includes the
Kotlin user interface, JNI inference engine, Gradle build files, and the same
five INT8 model assets used by the desktop package.

> **Models.** The five `*_quant.onnx` files are not in the repository. Download them from the [release](https://github.com/Ujjwl07/StableVQA-Edge/releases/download/v1.0.0/stablevqa-edge-models-v1.0.0.zip) and copy them into `android/app/src/main/assets/` before building.

## 1. Install tools

- Android Studio with Android SDK 36.
- Android NDK **21.4.7075529** and CMake **3.22.1** through Android Studio’s
  SDK Manager.
- JDK 11.
- A physical Android device is recommended. The tuned deployment profile is
  Samsung Exynos 2400; other ARM devices can run it but will have different
  latency and thermal behavior.

## 2. Open and build

Open the package’s `android/` folder in Android Studio. Allow Gradle sync to
finish; it downloads ONNX Runtime Android 1.19.2 and extracts the C/C++ headers
and ABI libraries before native compilation.

Alternatively, from the repository root:

```sh
cd android
./gradlew assembleDebug
```

The debug APK is written under `android/app/build/outputs/apk/debug/`.

## 3. Install and run

Connect an Android device with USB debugging enabled, select it in Android
Studio, and press **Run**, or use:

```sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

When the app reports that models are ready, tap **Select Folder** and grant
access to a folder containing supported videos. The app processes four clips
per video and writes `stablevqa_results.csv` to Downloads.

## 4. Preserved Exynos optimization profile

The JNI code pins work to CPUs 4–9, uses six intra-op threads, attempts
XNNPACK for supported INT8 models, and safely falls back to the CPU/MLAS path
for unsupported quantized attention graphs. The heavy-branch concurrency toggle
is intentionally off by default because it did not reduce latency on the
measured Exynos 2400 configuration.

For full architecture and dependency detail, also see
[android/README_ANDROID.md](android/README_ANDROID.md).
