# tobu: optional routed-kernel test and bench (GPU, gfx1151), off by default. Included by build.sh via
# -DCMAKE_PROJECT_tobu_fnx_INCLUDE (runs right after project()), deliberately not from CMakeLists.txt: that file
# belongs to the state_id, and these targets change no kernel. Enable with -DTOBU_FNX_TESTS=ON; then build only
# the needed targets, e.g. ninja tobu-fnx tobu_fnx_tests.
option(TOBU_FNX_TESTS "Build the routed kernel test and bench (tests/)" OFF)

function(_tobu_fnx_tests)
  if(NOT TOBU_FNX_TESTS)
    return()
  endif()
  foreach(name routed_pair_iq_test routed_pair_iq_bench routed_down_iq_test routed_down_iq_bench)
    add_executable(tobu_fnx_${name} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/tests/${name}.cpp)
    target_link_libraries(tobu_fnx_${name} PRIVATE fnx_engine)
    target_compile_options(tobu_fnx_${name} PRIVATE -O2)
  endforeach()
  add_custom_target(tobu_fnx_tests DEPENDS tobu_fnx_routed_pair_iq_test tobu_fnx_routed_pair_iq_bench
    tobu_fnx_routed_down_iq_test tobu_fnx_routed_down_iq_bench)
endfunction()
# At the end of CMakeLists.txt: fnx_engine exists by then.
cmake_language(DEFER CALL _tobu_fnx_tests)
