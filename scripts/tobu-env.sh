# Settings chosen at install time, stored by install.sh in <checkout>/.tobu.env (one KEY=value per line):
# TOBU_MODELS, TOBU_GGUF, TOBU_MTP, TOBU_IMAGE, TOBU_PY. Variables already set in the environment win.
# Sourced by ./tobu, scripts/serve.sh, scripts/run-fnx.sh (and read by bench/tobu_bench.py).
#   tobu_load_env <checkout> [KEY ...]   (no keys: all TOBU_* keys in the file)
tobu_load_env() {
  local f=$1/.tobu.env k v; shift
  [ -f "$f" ] || return 0
  while IFS='=' read -r k v; do
    case "$k" in TOBU_[A-Z_]*) ;; *) continue ;; esac
    if [ $# -gt 0 ]; then case " $* " in *" $k "*) ;; *) continue ;; esac; fi
    [ -n "${!k:-}" ] || export "$k=$v"
  done < "$f"
}
