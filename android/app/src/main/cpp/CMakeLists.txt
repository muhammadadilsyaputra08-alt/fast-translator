cmake_minimum_required(VERSION 3.22.1)
project(fastai)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)

target_include_directories(fastai PRIVATE
    ${CORE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_MEDUSA)
    target_compile_definitions(fastai PRIVATE FAST_ENABLE_MEDUSA=1)
endif()

find_library(log-lib log)

target_link_libraries(fastai PRIVATE ${log-lib})
