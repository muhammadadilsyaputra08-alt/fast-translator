cmake_minimum_required(VERSION 3.22.1)
project(fastai)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CRITICAL: without this, the NDK/CMake toolchain applies NO optimization
# flags at all for a debug-variant APK build (effectively -O0). This was
# true for every benchmark measured before this fix -- the 12.77 tok/s
# baseline, and the ~9.86 tok/s NEON numbers, were BOTH unoptimized builds.
# Hand-written NEON intrinsics use more temporaries/registers than the
# simple scalar loop, so without an optimizer to manage register
# allocation, NEON can end up SLOWER than scalar -- which is exactly what
# was observed. Forcing -O2 here applies regardless of whether the Android
# app build type is debug or release, since the native math kernels should
# always be optimized even when testing via the (unsigned-friendly) debug
# APK. See notes/NEON_OPTIMIZATION_FINDINGS.md for the full investigation.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -DNDEBUG") # opt in even for the "Debug" CMake config
add_compile_options(-O2)

option(ENABLE_MEDUSA "Enable Medusa multi-token decoding" ON)

# Path to the shared core engine (Fase 1), which lives outside the Android
# module so the same code can be unit-tested on desktop with plain g++.
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core)

add_library(fastai SHARED
    jni_bridge.cpp
    ${CORE_DIR}/src/fllm_parser.cpp
    ${CORE_DIR}/src/bitnet_kernel.cpp
    ${CORE_DIR}/src/kv_cache.cpp
    ${CORE_DIR}/src/medusa.cpp
    ${CORE_DIR}/src/engine.cpp
    ${CORE_DIR}/src/transformer.cpp
    ${CORE_DIR}/src/transformer_mixed.cpp
    ${CORE_DIR}/src/tokenizer.cpp
    ${CORE_DIR}/src/gbnf_parser.cpp
    ${CORE_DIR}/src/gbnf_runtime.cpp
    ${CORE_DIR}/src/neon_selftest.cpp
    ${CORE_DIR}/src/thread_bench.cpp
    ${CORE_DIR}/src/vulkan_bench.cpp
)

target_include_directories(fastai PRIVATE
    ${CORE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_MEDUSA)
    target_compile_definitions(fastai PRIVATE FAST_ENABLE_MEDUSA=1)
endif()

find_library(log-lib log)
find_library(vulkan-lib vulkan) # system lib, ships with the NDK/AOSP since API 24

target_link_libraries(fastai PRIVATE ${log-lib} ${vulkan-lib})
