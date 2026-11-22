# Build (and run) image for tobu-fnx on gfx1151 (AMD Strix Halo). Everything happens in the container, nothing on the host.
#   docker build -t tobu-build:gfx1151 docker/
# The release binary was built from this base (ROCm 10.1.0, Ubuntu 24.04). Same base digest + same sources = same binary
# (engine/cmake/reproducible.cmake fixes prefix maps and the HIP CUID), and the same binary is what keeps greedy
# outputs bit-identical to the published hashes.
# Base pinned by digest (= rocm/dev-ubuntu-24.04:10.1.0-full as of 2026-11-22; same layers as the release build image).
FROM rocm/dev-ubuntu-24.04:10.1.0-full@sha256:5ed1362ea542a928651e4c710b44024ebe98870f74159cb70f0265aec8ef0abe
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config git ca-certificates curl libicu-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# Sanity check: the ROCm libraries the engine links against. ROCm 10.1 packages them as amdrocm-*, so check the
# installed headers and libraries rather than Debian package names.
RUN for d in hipblas hipblaslt rocblas hipcub rocprim rocwmma; do \
      test -e /opt/rocm/core-10.1/include/$d || { echo "missing ROCm headers: $d"; exit 1; }; done && \
    for l in libhipblas.so libhipblaslt.so librocblas.so; do \
      test -e /opt/rocm/core-10.1/lib/$l || { echo "missing ROCm library: $l"; exit 1; }; done
ENV LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib
