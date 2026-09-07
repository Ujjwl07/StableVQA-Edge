# Runtime source selection

The desktop programs deliberately remain separate source implementations under
`platform/`. They use materially different measured execution strategies:

- `gpu/StableVqaGpu.cpp`: CUDA streams, optional TensorRT, and optional dual-GPU placement.
- `macos/StableVqaMac.cpp`: Apple Silicon CPU execution with three concurrent branch pools.
- `windows/StableVqaX86.cpp`: Intel/AMD topology detection, physical-core selection, and hybrid P-core affinity.

The top-level CMake project selects exactly one of these at configure time. This
avoids replacing platform-specific latency optimizations with a lowest-common-
denominator implementation.
