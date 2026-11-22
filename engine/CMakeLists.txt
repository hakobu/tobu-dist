# tobu-fnx: compute engine for Qwen3.8-Flash-Next on gfx1151 (edition qwen38-flash-next-gfx1151).
# src/ and gufo/ contain parts from gufo (gufo-org/gufo 2845939b, MIT) and flash-next-iq-gufo-extension
# (storm1er 9ce08ce, MIT), src/kernels/rocm/mmq/ from llama.cpp/ggml (MIT); THIRD_PARTY_NOTICES.md.
# Compiler flags per file as in the gufo build (compile_commands.json, 2026-11-19), a prerequisite for identical
# outputs. Build in the container tobu-gufo-build:0.34.0:
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/rocm && ninja -C build
cmake_minimum_required(VERSION 3.24)
# Before project(): otherwise enable_language(HIP) picks a default arch and gfx11 intrinsics (WMMA, sudot4) are missing.
set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "HIP architecture")
project(tobu_fnx LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipblaslt REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)   # Vision encoder: identity (SHA-256), TOB-37
find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp REQUIRED)
find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp REQUIRED)
find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp REQUIRED)

set(E ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(G ${CMAKE_CURRENT_SOURCE_DIR}/gufo)
set(MMQ ${E}/kernels/rocm/mmq)

add_compile_definitions(ENGINE_ENABLE_HIP=1)
# libstdc++ before HIP's __noinline__ macro (as in gufo).
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include \"${G}/src/core/hip/libstdcxx_noinline.hpp\">")
include_directories(${E} ${G}/include ${G} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${HIPCUB_INCLUDE_DIR} ${ROCPRIM_INCLUDE_DIR} ${ROCWMMA_INCLUDE_DIR})

# ---- Qwen kernels (gufo family recipe: -O2, Wave64 per file)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/qwen_kernels.txt QWEN_KERNELS)
list(TRANSFORM QWEN_KERNELS PREPEND ${E}/)
add_library(fnx_qwen_hip STATIC ${QWEN_KERNELS})
set_source_files_properties(${QWEN_KERNELS} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${E}/qwen/kernels/small_batch_wave64.hip ${E}/qwen/kernels/attention_decode_verify_wave64.hip
  PROPERTIES LANGUAGE HIP COMPILE_OPTIONS -mwavefrontsize64)
set_source_files_properties(${E}/qwen/kernels/prefill_quant_wave64.hip
  PROPERTIES LANGUAGE HIP COMPILE_OPTIONS "-mwavefrontsize64;-fno-slp-vectorize")
set_source_files_properties(${E}/qwen/kernels/small_batch_quant16_wave64.hip
  PROPERTIES LANGUAGE HIP COMPILE_OPTIONS
  "-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
set_source_files_properties(${E}/qwen/kernels/prefill_quant_gemm.hip
  PROPERTIES LANGUAGE HIP COMPILE_OPTIONS -fno-slp-vectorize)
target_link_libraries(fnx_qwen_hip PUBLIC hip::host roc::hipblas roc::hipblaslt)

# ---- llama.cpp MMQ/MMVQ (decode GEMVs): C++17, -O3 -ffast-math
set(MMQ_SOURCES ${MMQ}/fniq_ggml_stubs.hip.cpp ${MMQ}/fniq_mmq.hip.cpp ${MMQ}/mmid.hip.cpp ${MMQ}/mmvq.hip.cpp
  ${MMQ}/moe_wave64.hip.cpp ${MMQ}/quantize.hip.cpp)
add_library(fnx_mmq STATIC ${MMQ_SOURCES})
set_source_files_properties(${MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${MMQ}/moe_wave64.hip.cpp PROPERTIES LANGUAGE HIP COMPILE_OPTIONS -mwavefrontsize64)
set_target_properties(fnx_mmq PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_compile_definitions(fnx_mmq PRIVATE GGML_HIP_NO_VMM)
target_compile_options(fnx_mmq PRIVATE -O3 -ffast-math -fno-finite-math-only -Wno-unused-value
  -Wno-unused-command-line-argument)
target_include_directories(fnx_mmq SYSTEM PUBLIC ${MMQ})
target_link_libraries(fnx_mmq PUBLIC hip::host roc::hipblas)

# ---- Flash-Next kernels + host (executor, engine, loader)
add_library(fnx_engine STATIC
  ${E}/kernels/rocm/kernels.hip.cpp ${E}/kernels/rocm/w8a8_wave64.hip
  ${E}/engine.cpp ${E}/weights.cpp ${E}/config.cpp ${E}/ngram.cpp ${E}/rope_scaling.cpp
  ${E}/kernels/rocm/executor.cpp ${E}/kernels/rocm/batch.cpp ${E}/kernels/rocm/device_model.cpp
  ${E}/kernels/rocm/blaslt.cpp ${E}/qwen/text/tokenizer.cpp
  ${G}/src/core/gguf_reader.cpp ${G}/src/core/platform/mapped_file.cpp ${G}/src/core/hip/host_allocations.cpp
  ${G}/src/core/hip/weight_upload.cpp ${G}/src/core/quant/ggml_dequant.cpp ${G}/src/core/sampling.cpp
  ${G}/src/core/sampling_logit_row.cpp ${G}/src/core/platform/fs_posix.cpp ${G}/src/core/platform/page_pin_linux.cpp
  ${G}/src/core/platform/pages_posix.cpp
  ${E}/qwen/vision/prompt.cpp ${G}/src/core/crypto/sha256.cpp
  ${G}/src/core/hip/hipblaslt_plan_database.cpp ${G}/src/core/logging/logging.cpp
  ${G}/src/core/platform/os_info.cpp ${G}/src/core/platform/publish.cpp
  ${G}/src/core/json_constraint.cpp ${G}/src/core/json_schema_lexeme.cpp ${G}/src/core/json_schema_regex.cpp)
set_source_files_properties(${E}/kernels/rocm/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(${E}/kernels/rocm/w8a8_wave64.hip PROPERTIES LANGUAGE HIP COMPILE_OPTIONS -mwavefrontsize64)
set_source_files_properties(${G}/src/core/sampling.cpp PROPERTIES COMPILE_OPTIONS -fopenmp)
target_compile_definitions(fnx_engine PRIVATE QFN_ROCM_BUILD=1)
target_compile_options(fnx_engine PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-O3> $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
  $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only> $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
target_link_libraries(fnx_engine PUBLIC fnx_qwen_hip fnx_mmq hip::host roc::hipblas roc::hipblaslt
  ICU::uc ICU::i18n Threads::Threads OpenMP::OpenMP_CXX OpenSSL::Crypto)

# tobu: recompute the state identifier (state_id.h) on every build (changes only when a state-determining source
# changes; the file is then rewritten and tobu-fnx recompiled)
add_custom_target(tobu_state_id
  COMMAND ${CMAKE_COMMAND} -DROOT=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/gen/state_id.h
          "-DEXTRA=${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_HIP_COMPILER_VERSION}-${CMAKE_HIP_ARCHITECTURES}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/state_id.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/gen/state_id.h
  COMMENT "tobu: Zustands-Kennung")
add_executable(tobu-fnx tobu/tobu_fnx.cpp)
add_dependencies(tobu-fnx tobu_state_id)
target_include_directories(tobu-fnx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(tobu-fnx PRIVATE fnx_engine)
