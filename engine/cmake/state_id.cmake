# tobu (fingerprint by state): hash over the sources that determine the content of a session snapshot
# (kernels, model computation, prefill flow, snapshot header with lookup/MTP state). Not included: tobu/ (protocol, server
# binding), sampling/JSON/logging/platform. Same hash -> snapshots from another build are valid.
# Usage: cmake -DROOT=<engine> -DOUT=<state_id.h> -DEXTRA=<compiler/target> -P state_id.cmake
set(PATTERNS
  src/engine.cpp src/engine.hpp src/config.cpp src/config.hpp src/weights.cpp src/weights.hpp
  src/rope_scaling.cpp src/rope_scaling.hpp src/ggml_q2_0.hpp src/ngram.cpp src/ngram.hpp
  src/mtp_policy.hpp src/mtp_costs.hpp src/continuation_hooks.hpp
  src/kernels/* src/qwen/* gufo/src/core/hip/* gufo/src/core/quant/* gufo/src/core/speculative/*
  gufo/src/core/gguf_reader.cpp gufo/src/core/gguf_reader.hpp gufo/src/core/model_config.hpp
  CMakeLists.txt cmake/state_id.cmake)
set(FILES)
foreach(p ${PATTERNS})
  file(GLOB_RECURSE f RELATIVE ${ROOT} ${ROOT}/${p})
  list(APPEND FILES ${f})
endforeach()
list(REMOVE_DUPLICATES FILES)
list(SORT FILES)
list(FILTER FILES EXCLUDE REGEX "[.]md$")   # Docs (e.g. VENDOR.md) determine no state
set(ALL "toolchain:${EXTRA}\n")   # Compiler/version/GPU target: different kernels -> different state
# The normalized content is hashed: comments removed, whitespace collapsed to one space (TOB-61). Pure comment/
# formatting changes thus keep the state_id (and the SSD prefix caches); every code/number change changes it.
# String literals are kept (the first alternative catches them before "//" or "#"), otherwise the file is hashed raw.
set(RE_STR "(\"([^\"\\\\\n]|\\\\.)*\"|'([^'\\\\\n]|\\\\.)*')")
set(RE_C "${RE_STR}|//[^\n]*|/[*]([^*]|[*]+[^*/])*[*]+/")
set(RE_CMAKE "(\"([^\"\\\\]|\\\\.)*\")|#[[]=*[[]([^]]|][^]])*]=*]|#[^\n]*")
foreach(f ${FILES})
  get_filename_component(name ${f} NAME)
  string(REGEX MATCH "[.][^.]*$" ext "${name}")
  set(re "")
  if(ext MATCHES "^[.](c|cc|cpp|cxx|h|hh|hpp|hxx|hip|cu|cuh|inc)$")
    set(re "${RE_C}")
  elseif(name STREQUAL "CMakeLists.txt" OR ext STREQUAL ".cmake")
    set(re "${RE_CMAKE}")
  endif()
  if(re)
    file(READ ${ROOT}/${f} c)
    # Without backreference "\\1" (TOB-62): CMake < 3.29 aborts on "\\1" if the group did not participate in the match
    # (comment match: "out-of-range escape"). Hence match by match: keep string literals, drop comments.
    # Same result as REGEX REPLACE in CMake >= 3.29, hence the same state_id.
    set(rest "${c}")
    set(c "")
    while(1)
      string(REGEX MATCH "${re}" m "${rest}")
      if(m STREQUAL "")
        break()
      endif()
      string(FIND "${rest}" "${m}" pos)   # first occurrence of the text = position of the match (leftmost)
      string(SUBSTRING "${rest}" 0 ${pos} vor)
      string(APPEND c "${vor}")
      string(SUBSTRING "${m}" 0 1 m0)
      if(m0 STREQUAL "\"" OR m0 STREQUAL "'")
        string(APPEND c "${m}")
      endif()
      string(LENGTH "${m}" n)
      math(EXPR pos "${pos} + ${n}")
      string(SUBSTRING "${rest}" ${pos} -1 rest)
    endwhile()
    string(APPEND c "${rest}")
    string(REGEX REPLACE "[ \t\r\n]+" " " c "${c}")
    string(STRIP "${c}" c)
    string(SHA256 h "${c}")
  else()
    file(SHA256 ${ROOT}/${f} h)
  endif()
  string(APPEND ALL "${f}:${h}\n")
endforeach()
string(SHA256 H "${ALL}")
string(SUBSTRING ${H} 0 32 H16)
set(BYTES "")
foreach(i RANGE 0 30 2)
  string(SUBSTRING ${H16} ${i} 2 b)
  string(APPEND BYTES "0x${b}, ")
endforeach()
list(LENGTH FILES N)
set(CONTENT "// erzeugt von cmake/state_id.cmake (${N} Dateien) – nicht von Hand ändern\n#pragma once\nstatic constexpr unsigned char kTobuStateId[16] = {${BYTES}};\n#define TOBU_STATE_ID_HEX \"${H16}\"\n")
if(EXISTS ${OUT})
  file(READ ${OUT} OLD)
endif()
if(NOT "${OLD}" STREQUAL "${CONTENT}")
  file(WRITE ${OUT} "${CONTENT}")
endif()
