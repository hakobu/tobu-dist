# tobu (TOB-63): bit-identical build of tobu-fnx, independent of build and source directory (checksum across machines).
# Included by build.sh via -DCMAKE_PROJECT_INCLUDE (runs right after project()). Deliberately not in CMakeLists.txt:
# that file belongs to the state_id, and these flags change no kernel, only paths in debug info/__FILE__ and the CUID name.
#
# Causes of the difference between two build directories (b1/b2, same tree, same size, ~1.1 MB of differing bytes):
#  1. DWARF (host and device code in the HIP offload bundle): DW_AT_comp_dir = build directory, file paths absolute.
#     -> -ffile-prefix-map (debug info and __FILE__) to fixed names; the Clang driver passes it on to host and
#        device compilation. If the bundle is compressed, a different path changes the whole bundle.
#  2. HIP CUID (__hip_cuid_<id>, in host and device code): Clang derives it from the source path + all arguments,
#     i.e. also from the absolute paths of the prefix maps. -> fixed -cuid per source from the relative path.
#  3. -grecord-command-line would embed the command line (with paths) -> explicitly off.
# No __DATE__/__TIME__ in the sources; SOURCE_DATE_EPOCH is therefore not needed.
# Debugging: gdb "set substitute-path /tobu/engine <engine>" and "set substitute-path /tobu/build <build-dir>".

set(_tobu_repro_src "${CMAKE_CURRENT_SOURCE_DIR}")
set(_tobu_repro_bin "${CMAKE_CURRENT_BINARY_DIR}")
# Order: build directory last. Clang >= 17 takes the last matching map, older versions the longest; so with
# build/ inside engine/ /tobu/build wins in both cases.
add_compile_options(
  "-ffile-prefix-map=${_tobu_repro_src}=/tobu/engine"
  "-ffile-prefix-map=${_tobu_repro_bin}=/tobu/build"
  -gno-record-gcc-switches)

function(_tobu_repro_cuid)
  get_property(targets DIRECTORY "${_tobu_repro_src}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(t ${targets})
    get_target_property(srcs ${t} SOURCES)
    if(NOT srcs)
      continue()
    endif()
    foreach(s ${srcs})
      get_filename_component(abs "${s}" ABSOLUTE BASE_DIR "${_tobu_repro_src}")
      file(RELATIVE_PATH rel "${_tobu_repro_src}" "${abs}")
      string(MD5 h "${t}:${rel}")
      string(SUBSTRING "${h}" 0 16 h)
      # Only for HIP compilations (CXX files stay untouched)
      set_property(SOURCE "${abs}" DIRECTORY "${_tobu_repro_src}" APPEND PROPERTY COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:HIP>:-cuid=${h}>")
    endforeach()
  endforeach()
endfunction()
# Only at the end of CMakeLists.txt: by then all targets and their sources are fixed.
cmake_language(DEFER CALL _tobu_repro_cuid)
