cmake_minimum_required(VERSION 3.22.1)
project("video_stability_engine")

# 1. Bring in the manually extracted ONNX Runtime headers
include_directories(${ONNXRUNTIME_DIR}/headers)

# 2. Tell CMake exactly where the pre-built ONNX .so file is
#    for the current CPU architecture
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
    ${ONNXRUNTIME_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so)

# 3. Build our C++ pipeline
add_library(
        video_stability_engine
        SHARED
        native-lib.cpp)

find_library(
        log-lib
        log)

# 4. Link the custom C++ code to the ONNX Runtime library
target_link_libraries(
        video_stability_engine
        onnxruntime
        android
        ${log-lib})