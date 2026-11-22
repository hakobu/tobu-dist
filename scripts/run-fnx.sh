#!/usr/bin/env bash
# Start the tobu-fnx engine inside its container (ROCm, gfx1151). The server calls this as
#   scripts/run-fnx.sh <path/to/tobu-fnx> --model <gguf> [--mtp <gguf>] [--ple <gguf>] [--mmproj <gguf>] ...
# and talks to the engine over stdin/stdout (framed protocol), so the container runs with -i and no network.
#
# Environment:
#   TOBU_IMAGE        container image (default tobu-build:gfx1151)
#   TOBU_DOCKER       container CLI (default docker)
#   TOBU_MODELS       model directory, mounted read-only if it exists (default $HOME/models)
#   TOBU_GGUF, TOBU_MTP   their directories are mounted read-only as well (models may live elsewhere)
#   TOBU_FNX_DISK_DIR SSD prefix store; mounted read-write (the engine writes the snapshot files)
#   TOBU_FNX_OOM_ADJ  --oom-score-adj of the container (default 0)
#   TOBU_CONTAINER_LD_LIBRARY_PATH   LD_LIBRARY_PATH inside the container (default /opt/rocm/core-10.1/lib, where ROCm 10.1 keeps HIP/hipBLAS)
# Engine settings forwarded into the container: TOBU_FNX_PREFILL_CHUNK (default 16384), TOBU_FNX_MTP_GATE,
# TOBU_FNX_DRAFT_VOCAB_WINDOW, TOBU_FNX_IMAGE_MB, TOBU_FNX_VISION_IDLE_S, TOBU_FNX_CORPUS_MIN, TOBU_FNX_IQ_PAIR,
# TOBU_FNX_IQ_DOWN, TOBU_FNX_MMPROJ, TOBU_FNX_XSTORE_TOKENS, TOBU_FNX_DISK_DIR, TOBU_FNX_DISK_MB, TOBU_FNX_CACHE_MB.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$ROOT/scripts/tobu-env.sh" && tobu_load_env "$ROOT"   # TOBU_IMAGE/TOBU_MODELS from install.sh
IMAGE=${TOBU_IMAGE:-tobu-build:gfx1151}
DOCKER=${TOBU_DOCKER:-docker}
MODELS=${TOBU_MODELS:-$HOME/models}

# Large prefill chunk when the engine is alone (16,384 instead of 4,096: +6-10 % cold prefill, bit-identical;
# about 4 GiB more scratch memory).
export TOBU_FNX_PREFILL_CHUNK=${TOBU_FNX_PREFILL_CHUNK:-16384}

[ $# -ge 1 ] || { echo "usage: $0 <tobu-fnx binary> [engine args...]" >&2; exit 2; }
BIN=$1
[ -x "$BIN" ] || { echo "run-fnx.sh: engine binary not found or not executable: $BIN (build it with engine/build.sh)" >&2; exit 1; }
BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")

SEEN=$'\n'
MOUNTS=()
mount_dir() { # $1 = directory, $2 = ro|rw
  local d=$1
  [ -n "$d" ] && [ -d "$d" ] || return 0
  d=$(cd "$d" && pwd -P)
  case "$SEEN" in *$'\n'"$d"$'\n'*) return 0 ;; esac
  SEEN+="$d"$'\n'
  MOUNTS+=(-v "$d:$d:$2")
}
mount_file_dir() { # directory of a file argument, read-only (and of its symlink target, so linked models work)
  [ -n "$1" ] && [ -e "$1" ] || return 0
  mount_dir "$(dirname "$1")" ro
  [ -L "$1" ] && mount_dir "$(dirname "$(readlink -f "$1")")" ro
  return 0
}

# The SSD prefix store is written by the engine: read-write (and created, so the mount works)
if [ -n "${TOBU_FNX_DISK_DIR:-}" ]; then
  mkdir -p "$TOBU_FNX_DISK_DIR"
  mount_dir "$TOBU_FNX_DISK_DIR" rw
fi
mount_dir "$ROOT" ro
mount_dir "$(dirname "$BIN")" ro
mount_dir "$MODELS" ro
mount_file_dir "${TOBU_GGUF:-}"
mount_file_dir "${TOBU_MTP:-}"
mount_file_dir "${TOBU_FNX_MMPROJ:-}"
# every file given to the engine (--model, --mtp, --ple, --mmproj): its directory read-only
prev=""
for a in "${@:2}"; do
  case "$prev" in --model|--mtp|--ple|--mmproj) mount_file_dir "$a" ;; esac
  prev=$a
done

gid_of() { # group id: getent if available, else the owner group of the device
  local g
  g=$(getent group "$1" 2>/dev/null | cut -d: -f3 || true)
  [ -n "$g" ] && { echo "$g"; return; }
  [ -e "$2" ] && stat -c %g "$2" 2>/dev/null || true
}
GROUPS_ARGS=()
for spec in "video:/dev/dri" "render:/dev/kfd"; do
  g=$(gid_of "${spec%%:*}" "${spec#*:}")
  [ -n "$g" ] && GROUPS_ARGS+=(--group-add "$g")
done

ENV_ARGS=(-e HOME=/tmp)
ENV_ARGS+=(-e "LD_LIBRARY_PATH=${TOBU_CONTAINER_LD_LIBRARY_PATH:-/opt/rocm/core-10.1/lib}")
for v in TOBU_FNX_PREFILL_CHUNK TOBU_FNX_MTP_GATE TOBU_FNX_DRAFT_VOCAB_WINDOW TOBU_FNX_IMAGE_MB TOBU_FNX_VISION_IDLE_S \
         TOBU_FNX_CORPUS_MIN TOBU_FNX_IQ_PAIR TOBU_FNX_IQ_DOWN TOBU_FNX_MMPROJ TOBU_FNX_XSTORE_TOKENS \
         TOBU_FNX_DISK_DIR TOBU_FNX_DISK_MB TOBU_FNX_CACHE_MB; do
  [ -n "${!v+x}" ] && ENV_ARGS+=(-e "$v")
done

exec "$DOCKER" run --rm -i --label tobu=fnx --oom-score-adj "${TOBU_FNX_OOM_ADJ:-0}" --network none \
  --device /dev/kfd --device /dev/dri ${GROUPS_ARGS[@]+"${GROUPS_ARGS[@]}"} \
  --ulimit memlock=-1 --security-opt seccomp=unconfined --ipc=host -u "$(id -u):$(id -g)" \
  "${ENV_ARGS[@]}" "${MOUNTS[@]}" -w "$ROOT" "$IMAGE" "$BIN" "${@:2}"
