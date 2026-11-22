#!/bin/bash
# Build tobu-fnx in the container: engine/build.sh [build-dir]        (default engine/build; image: TOBU_IMAGE)
#   docker build -t tobu-build:gfx1151 docker/     (once)
# Kernel tests/benchmarks (tests/, cmake/fnx_tests.cmake, need the GPU): TOBU_CMAKE_ARGS=-DTOBU_FNX_TESTS=ON,
# targets tobu_fnx_tests (not part of "all").
# Bit-identical across build directories and machines: cmake/reproducible.cmake (prefix maps, fixed HIP CUID).
# The release binary (and production) is built with the flags below; with them this script reproduces its sha256.
BASE_ARGS="-DTOBU_FNX_TESTS=ON -DCMAKE_HIP_FLAGS=-Rpass-analysis=kernel-resource-usage"
set -e
H=$(cd "$(dirname "$0")" && pwd); B=$(mkdir -p "${1:-$H/build}" && cd "${1:-$H/build}" && pwd)
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$H:$H" -v "$B:$B" "${TOBU_IMAGE:-tobu-build:gfx1151}" \
  bash -c "cmake -S $H -B $B -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_PROJECT_INCLUDE=$H/cmake/reproducible.cmake -DCMAKE_PROJECT_tobu_fnx_INCLUDE=$H/cmake/fnx_tests.cmake $BASE_ARGS ${TOBU_CMAKE_ARGS:-} >/dev/null && ninja -C $B -k 0 > $B/build.log 2>&1" || { tail -30 "$B/build.log"; exit 1; }
echo "built: $B/tobu-fnx (log: $B/build.log)"
