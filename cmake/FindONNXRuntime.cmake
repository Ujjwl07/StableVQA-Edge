# Locates an ONNX Runtime binary package. Set ONNXRUNTIME_ROOT explicitly, or
# place an `onnxruntime/` folder beside this top-level CMakeLists.txt.
if(NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_LIST_DIR}/../onnxruntime")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h" OR
   NOT EXISTS "${ONNXRUNTIME_ROOT}/lib")
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT is not a valid ONNX Runtime C/C++ directory: '${ONNXRUNTIME_ROOT}'.\n"
    "Expected: include/onnxruntime_cxx_api.h and lib/libonnxruntime*.dylib.\n"
    "For the bundled macOS runtime use: ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-macos")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include" ENV ONNXRUNTIME_ROOT
  PATH_SUFFIXES include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib" ENV ONNXRUNTIME_ROOT
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
  set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
endif()
