#!/bin/bash
# tobu install for AMD Strix Halo (Ryzen AI Max+ 395, gfx1151, 128 GB): binary + model with checksums, then start the
# OpenAI-compatible server on 127.0.0.1:8080.
#
#   ./install.sh                 download + verify + start
#   ./install.sh --no-start      download + verify only
#   ./install.sh --build         build the engine from source in Docker instead of downloading the release binary
#   ./install.sh --head-p        optional: use MTP head P (Hugging Face, pinned revision, sha256-verified) instead of the
#                                unsloth head; same output, faster decode. Not the default.
#
# What it does NOT do: no sudo, no system packages, no services, no changes outside this folder and TOBU_MODELS.
# It checks the prerequisites and tells you what is missing (Docker with GPU access, python3, curl, ~95 GB free disk).
# Env: TOBU_MODELS (model folder, default ~/models/Qwen3.8-Flash-Next-UD-IQ4_XS), TOBU_RELEASE (release tag),
#      TOBU_REPO (GitHub repo of the release asset), TOBU_IMAGE (default tobu-build:gfx1151),
#      HF_TOKEN (optional; sent with the --head-p download if set), TOBU_PY (Python with server/requirements.txt).
# The chosen settings (model folder and files, MTP head, image) are saved in .tobu.env, so ./tobu serve and ./tobu bench
# use them without exporting anything again; a later ./install.sh run reuses TOBU_MODELS and TOBU_IMAGE from there.
set -euo pipefail
# Everything runs inside main(), so a truncated download (curl | bash) cannot execute half a script.
main() {
# Also works as a one-liner: curl -fsSL https://raw.githubusercontent.com/hakobu/tobu-dist/main/install.sh | bash
# Outside a tobu-dist checkout (no docker/, engine/, scripts/ next to this script, or piped from curl), clone the repo
# into ./tobu-dist (or $TOBU_DIR) and run the script from there with the same arguments.
is_checkout() { [ -d "$1/docker" ] && [ -d "$1/engine" ] && [ -d "$1/scripts" ]; }
SRC=${BASH_SOURCE[0]:-}
H=$([ -n "$SRC" ] && [ -f "$SRC" ] && cd "$(dirname "$SRC")" && pwd || pwd)
if ! is_checkout "$H"; then
  D=${TOBU_DIR:-$PWD/tobu-dist}
  command -v git >/dev/null || { echo "error: git not found – please install it" >&2; exit 1; }
  if is_checkout "$D"; then
    echo "==> using existing checkout $D (not updated; run: git -C $D pull)"
  elif [ -e "$D" ]; then
    echo "error: $D exists but is not a tobu-dist checkout (set TOBU_DIR to another folder)" >&2; exit 1
  else
    echo "==> cloning tobu-dist into $D"
    git clone --depth 1 https://github.com/hakobu/tobu-dist "$D"
  fi
  exec bash "$D/install.sh" "$@"
fi
USER_PY=${TOBU_PY:-}
. "$H/scripts/tobu-env.sh" && tobu_load_env "$H" TOBU_MODELS TOBU_IMAGE TOBU_PY   # settings of an earlier run
M=${TOBU_MODELS:-$HOME/models/Qwen3.8-Flash-Next-UD-IQ4_XS}
REPO=${TOBU_REPO:-hakobu/tobu-dist}
TAG=${TOBU_RELEASE:-v0.8.0}
IMAGE=${TOBU_IMAGE:-tobu-build:gfx1151}
BIN=$H/engine/build/tobu-fnx

# Model: unsloth/Qwen3.8-Flash-Next-GGUF, pinned revision (the files all published numbers were measured with).
HF=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6
MODEL_FILES=(
  "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf 5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"
  "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf 577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"
  "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"
  "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf 5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6"
)
QWEN_LICENSE=https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE
# Optional MTP head P (public Hugging Face repo hakobuAi/tobu-mtp-head-p, pinned revision; no token needed).
HEADP_URL=https://huggingface.co/hakobuAi/tobu-mtp-head-p/resolve/cd5add7e1f0ba947287863a68fa9c3305e741f83/mtp-tobu-p-iq4xs.gguf
HEADP_SHA256=c0d60e1c2374f292caed86f94018ec1c2fda82556a5b4ee04867bf4f47ec29f0
BIN_SHA256=d245dd1e3848418547435726307bffb334c0d4de1a8bbe7c46a0ed9293c72346   # release v0.8.0 = engine/build.sh output

START=1; BUILD=0; HEAD_P=0
for a in "$@"; do
  case "$a" in
    --no-start) START=0 ;;
    --build) BUILD=1 ;;
    --head-p) HEAD_P=1 ;;
    -h|--help) sed -n 2,17p "$H/install.sh"; exit 0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

# ---- prerequisites (checked, never installed) ----------------------------------------------------------------------
for c in docker python3 curl sha256sum; do command -v $c >/dev/null || die "$c not found – please install it"; done
docker info >/dev/null 2>&1 || die "docker is not usable by $(id -un) (is the daemon running, are you in the docker group?)"
[ -e /dev/kfd ] && [ -d /dev/dri ] || die "/dev/kfd or /dev/dri missing – amdgpu kernel driver with ROCm (KFD) support needed"
id -nG | tr ' ' '\n' | grep -qx render || echo "note: you are not in the 'render' group; the container gets it via --group-add, which is fine"
grep -q gfx1151 /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null || \
  grep -qi "gfx_target_version 110501" /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null || \
  echo "warning: no gfx1151 GPU found in the KFD topology – this edition is built for Strix Halo only"
mem=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
[ "$mem" -ge 100 ] || echo "warning: ${mem} GB RAM visible; the model needs ~90 GB of unified memory (set the GPU share in BIOS/kernel)"
# Python packages: an explicit TOBU_PY is used as is; otherwise a private venv in the checkout ($H/.venv), which
# scripts/serve.sh and ./tobu pick up automatically. Checked with -I, exactly as the server runs.
if [ -n "${TOBU_PY:-}" ]; then
  PY=$TOBU_PY
  "$PY" -I -c 'import numpy, tokenizers, jinja2' 2>/dev/null || die "$PY lacks numpy, tokenizers or jinja2 (pip install -r $H/server/requirements.txt)"
else
  PY=$H/.venv/bin/python
  if ! { [ -x "$PY" ] && "$PY" -I -c 'import numpy, tokenizers, jinja2' 2>/dev/null; }; then
    say "creating the Python environment $H/.venv (server/requirements.txt)"
    rm -rf "$H/.venv"
    if python3 -m venv "$H/.venv" >/dev/null 2>&1; then
      "$PY" -m pip install -q -r "$H/server/requirements.txt" || die "pip install -r server/requirements.txt failed"
    elif command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
      # no ensurepip (Ubuntu/Debian without python3-venv): uv builds the venv without sudo
      UV=$(command -v uv || echo "$HOME/.local/bin/uv")
      rm -rf "$H/.venv"; say "python3 -m venv is not available here; using uv ($UV)"
      "$UV" venv -q "$H/.venv" && VIRTUAL_ENV=$H/.venv "$UV" pip install -q -r "$H/server/requirements.txt" \
        || { rm -rf "$H/.venv"; die "uv could not create the Python environment"; }
    else
      rm -rf "$H/.venv"
      v=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
      die "python3 -m venv failed: the venv module is missing. Install it with: sudo apt install python3-venv (or python$v-venv),
       or install uv (https://docs.astral.sh/uv/), then run ./install.sh again"
    fi
    "$PY" -I -c 'import numpy, tokenizers, jinja2' || die "Python packages still missing after pip install"
  fi
fi
export TOBU_PY=$PY

# ---- image -----------------------------------------------------------------------------------------------------------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  say "building the runtime/build image $IMAGE (ROCm 10.1.0, ~10 GB download, once)"
  if docker buildx version >/dev/null 2>&1; then docker buildx build --load -t "$IMAGE" "$H/docker"
  else docker build -t "$IMAGE" "$H/docker"; fi
fi

# ---- engine binary ---------------------------------------------------------------------------------------------------
if [ "$BUILD" = 1 ]; then
  say "building tobu-fnx from source (10–20 min)"
  TOBU_IMAGE=$IMAGE "$H/engine/build.sh" "$H/engine/build"
  if [ "$(sha "$BIN")" = "$BIN_SHA256" ]; then say "source build reproduces the release binary (sha256 ${BIN_SHA256:0:16}…)"
  else echo "note: this build differs from the release binary (expected sha256 ${BIN_SHA256:0:16}…); outputs may not match the published hashes"; fi
elif [ ! -x "$BIN" ] || [ "$(sha "$BIN")" != "$BIN_SHA256" ]; then
  say "downloading tobu-fnx ($TAG)"
  mkdir -p "$(dirname "$BIN")"
  curl -fL --retry 3 -o "$BIN.part" "https://github.com/$REPO/releases/download/$TAG/tobu-fnx"
  [ "$(sha "$BIN.part")" = "$BIN_SHA256" ] || { rm -f "$BIN.part"; die "tobu-fnx checksum mismatch"; }
  chmod +x "$BIN.part" && mv "$BIN.part" "$BIN"
fi
say "engine: $BIN ($(sha "$BIN" | cut -c1-16)…)"

# ---- model (~87 GiB + 2.6 GiB MTP head) -------------------------------------------------------------------------------
mkdir -p "$M"
need=0
for e in "${MODEL_FILES[@]}"; do set -- $e; [ -f "$M/$(basename "$1")" ] || need=1; done
if [ "$need" = 1 ]; then
  free=$(df -Pk "$M" | awk 'NR==2{print int($4/1048576)}')
  [ "$free" -ge 95 ] || die "only ${free} GB free in $M, ~95 GB needed"
  say "model weights: unsloth/Qwen3.8-Flash-Next-GGUF, licensed under the Qwen Community License 1.0"
  echo "    $QWEN_LICENSE"
  echo "    The weights keep their publishers' terms and are not bundled with tobu; downloading them means accepting that license."
fi
for e in "${MODEL_FILES[@]}"; do
  set -- $e; f="$M/$(basename "$1")"
  if [ -f "$f" ] && [ -f "$f.ok" ] && [ "$(cat "$f.ok")" = "$2" ]; then continue; fi
  if [ -f "$f" ]; then             # present (e.g. downloaded before): accept it if the checksum matches
    say "checking existing $(basename "$1")"
    if [ "$(sha "$f")" = "$2" ]; then echo "$2" > "$f.ok"; continue; fi
    die "$f exists but its checksum does not match the pinned revision – move it away and run again"
  fi
  say "downloading $(basename "$1")"
  curl -fL --retry 5 -C - -o "$f.part" "$HF/$1"
  say "verifying $(basename "$1")"
  [ "$(sha "$f.part")" = "$2" ] || die "checksum mismatch for $(basename "$1") (delete $f.part and retry)"
  mv "$f.part" "$f" && echo "$2" > "$f.ok"
done
say "model: $M"

MTP=$M/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
if [ "$HEAD_P" = 1 ]; then
  f=$M/mtp-tobu-p-iq4xs.gguf
  if ! { [ -f "$f" ] && [ -f "$f.ok" ] && [ "$(cat "$f.ok")" = "$HEADP_SHA256" ]; }; then
    if [ -f "$f" ] && [ "$(sha "$f")" = "$HEADP_SHA256" ]; then echo "$HEADP_SHA256" > "$f.ok"
    else
      free=$(df -Pk "$M" | awk 'NR==2{print int($4/1048576)}')
      [ "$free" -ge 3 ] || die "only ${free} GB free in $M, ~3 GB needed for head P"
      say "downloading MTP head P (2.8 GB, Qwen Community License 1.0)"
      # the token goes to curl on stdin (-K -), never on the command line
      rc=0
      { [ -n "${HF_TOKEN:-}" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; true; } |
        curl -K - -fL --retry 5 -C - -o "$f.part" "$HEADP_URL" || rc=$?
      case "$rc" in
        0) ;;
        22) die "head P download refused by Hugging Face (HTTP 401/403/404; if HF_TOKEN is set, check that it is valid)" ;;
        *) die "head P download failed (curl exit $rc: network or proxy problem; run again to resume)" ;;
      esac
      [ "$(sha "$f.part")" = "$HEADP_SHA256" ] || { rm -f "$f.part"; die "head P checksum mismatch (download deleted, run again)"; }
      mv "$f.part" "$f" && echo "$HEADP_SHA256" > "$f.ok"
    fi
  fi
  MTP=$f
  say "MTP head: head P"
fi

export TOBU_GGUF=$M/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf TOBU_MTP=$MTP
# ---- settings for ./tobu serve / ./tobu bench (.tobu.env; the environment still wins) -------------------------------
{ echo "TOBU_MODELS=$M"; echo "TOBU_GGUF=$TOBU_GGUF"; echo "TOBU_MTP=$TOBU_MTP"; echo "TOBU_IMAGE=$IMAGE"
  [ -n "$USER_PY" ] && echo "TOBU_PY=$USER_PY"; true; } > "$H/.tobu.env"
say "settings saved in $H/.tobu.env (model, MTP head, image)"
[ "$START" = 1 ] || { say "installed. Start with: ./tobu serve start"; exit 0; }
say "starting the server (first load ~1–2 min)"
"$H/scripts/serve.sh" start
cat <<EOF

tobu is running: OpenAI-compatible API at http://127.0.0.1:8080/v1 (model name: any), chat page at http://127.0.0.1:8080/chat
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
Stop: ./tobu serve stop      Benchmark: ./tobu bench (about 9 min; --quick: about 3 min)
EOF
}

main "$@"
