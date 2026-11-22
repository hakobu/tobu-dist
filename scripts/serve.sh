#!/usr/bin/env bash
# Start/stop the tobu server (Qwen3.8-Flash-Next on gfx1151, engine in a container via scripts/run-fnx.sh).
#
#   scripts/serve.sh start | stop | status
#
# Environment (defaults in brackets):
#   TOBU_GGUF   first shard of the model [$HOME/models/Qwen3.8-Flash-Next-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf]
#   TOBU_MTP    MTP draft head [$HOME/models/Qwen3.8-Flash-Next-UD-IQ4_XS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf];
#               missing file -> start without MTP (slower, same output)
#   TOBU_SLOTS [4]  TOBU_HOST [127.0.0.1]  TOBU_PORT [8080]  TOBU_PY [<repo>/.venv/bin/python if present, else python3]  TOBU_MAX_LEN [65536]
#   TOBU_STATE  directory for PID file and log [$HOME/.local/state/tobu]; shared by all checkouts of this user, so
#               give a second checkout its own TOBU_STATE (or TOBU_PORT) to run both
#   Settings saved by install.sh in <checkout>/.tobu.env are used unless the variable is set in the environment.
#   TOBU_ARGS   extra server flags, e.g. "--slots 2 --ban-scripts cjk --greedy-harness claude-cli"
#   TOBU_START_TIMEOUT  seconds to wait for /health 200 [900]
#   TOBU_DOCKER [docker]; image/mount settings: see scripts/run-fnx.sh
# The PID file holds the server's own PID (no pkill -f patterns, which also hit unrelated shells).
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$ROOT/scripts/tobu-env.sh" && tobu_load_env "$ROOT"   # TOBU_GGUF/TOBU_MTP/TOBU_IMAGE/TOBU_PY from install.sh
MODEL_DIR_DEFAULT=$HOME/models/Qwen3.8-Flash-Next-UD-IQ4_XS
export TOBU_GGUF=${TOBU_GGUF:-$MODEL_DIR_DEFAULT/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
export TOBU_MTP=${TOBU_MTP:-$MODEL_DIR_DEFAULT/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
HOST=${TOBU_HOST:-127.0.0.1}
PORT=${TOBU_PORT:-8080}
PY=${TOBU_PY:-$([ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || echo python3)}
LEN=${TOBU_MAX_LEN:-65536}; SLOTS=${TOBU_SLOTS:-4}
STATE=${TOBU_STATE:-$HOME/.local/state/tobu}
TIMEOUT=${TOBU_START_TIMEOUT:-900}
DOCKER=${TOBU_DOCKER:-docker}
PID=$STATE/tobu-server-$PORT.pid
LOG=$STATE/tobu-server-$PORT.log
URL_HOST=$HOST; [ "$HOST" = "0.0.0.0" ] && URL_HOST=127.0.0.1

running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }

health() { # 0 if /health answers 200
  if command -v curl >/dev/null 2>&1; then
    curl -sf -o /dev/null --max-time 5 "http://$URL_HOST:$PORT/health"
  else
    "$PY" -c 'import sys, urllib.request
try:
    sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=5).status == 200 else 1)
except Exception:
    sys.exit(1)' "http://$URL_HOST:$PORT/health"
  fi
}

engine_containers() { "$DOCKER" ps -q --filter label=tobu=fnx 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "already running (PID $(cat "$PID"), http://$URL_HOST:$PORT)"; exit 0; fi
    [ -f "$TOBU_GGUF" ] || { echo "model not found: $TOBU_GGUF (set TOBU_GGUF)" >&2; exit 1; }
    mtp=()
    if [ -f "$TOBU_MTP" ]; then
      mtp=(--fnx-mtp "$TOBU_MTP")
    else
      echo "note: MTP head not found ($TOBU_MTP) - starting without MTP (slower, same output)"
    fi
    mkdir -p "$STATE"
    echo "===== start $(date -u +%FT%TZ) =====" >> "$LOG"
    # shellcheck disable=SC2086  # TOBU_ARGS: extra flags, split on purpose
    nohup "$PY" -I "$ROOT/server/server.py" --engine flashnext --gguf "$TOBU_GGUF" ${mtp[@]+"${mtp[@]}"} \
      --max-len "$LEN" --slots "$SLOTS" --host "$HOST" --port "$PORT" --arm C ${TOBU_ARGS:-} </dev/null >>"$LOG" 2>&1 &
    echo $! > "$PID"
    t0=$(date +%s)
    while :; do
      if health; then echo "ready (PID $(cat "$PID"), http://$URL_HOST:$PORT, log $LOG)"; exit 0; fi
      if ! running; then echo "server exited, see $LOG" >&2; tail -n 20 "$LOG" >&2; rm -f "$PID"; exit 1; fi
      if [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT" ]; then
        echo "not ready after ${TIMEOUT} s - stopping (log $LOG)" >&2
        "$0" stop >/dev/null
        exit 1
      fi
      sleep 2
    done ;;
  stop)
    if [ -f "$PID" ]; then
      p=$(cat "$PID")
      kill "$p" 2>/dev/null
      # the server writes hot prefix entries to the SSD store on SIGTERM (TOBU_FNX_DISK_STOP_S, default 20 s)
      for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
      kill -9 "$p" 2>/dev/null
      rm -f "$PID"
    else
      echo "no PID file ($PID)"
    fi
    # The engine container (--rm) ends when the server closes its stdin, but only after freeing its GPU memory.
    # Wait for it, otherwise an old and a new engine share the GPU.
    for _ in $(seq 1 90); do [ -z "$(engine_containers)" ] && break; sleep 1; done
    left=$(engine_containers)
    if [ -n "$left" ]; then
      echo "engine container still running after 90 s - killing it"
      # shellcheck disable=SC2086
      "$DOCKER" kill $left >/dev/null
      for _ in $(seq 1 30); do [ -z "$(engine_containers)" ] && break; sleep 1; done
    fi
    echo "stopped" ;;
  status)
    if running; then
      echo "running (PID $(cat "$PID"), http://$URL_HOST:$PORT)"
      health && echo "health: ok" || echo "health: not ready"
    else
      echo "not running"; exit 3
    fi ;;
  *)
    echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
