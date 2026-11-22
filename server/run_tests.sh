#!/usr/bin/env bash
# All server tests that run without GPU, model files or a running server.
#
#   server/run_tests.sh            exit 1 if a test fails
#   server/run_tests.sh --quick    skip the slow statistical sampling test (test_sampling.py, ~4 min)
#
# Environment: PYTHON (default python3). Needs numpy, tokenizers, jinja2; pillow for test_media.py.
# The tests are standalone scripts (no pytest); passed means exit 0.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1
TESTS=(test_fnx_adapter test_server_fnx test_chat_auth test_api test_tool_stream test_deferred_tools test_tob59 test_tob2 test_tob12
       test_tob30 test_tob32 test_tob33 test_ban_scripts test_media)
[ $QUICK = 1 ] || TESTS+=(test_sampling)
OK=0; FAIL=0; FAILED=()
cd "$HERE" || exit 1
for t in "${TESTS[@]}"; do
  t0=$SECONDS
  if out=$("$PY" -I "$t.py" 2>&1); then
    OK=$((OK + 1)); printf 'ok      %-24s %3ss  (%s checks)\n' "$t" "$((SECONDS - t0))" "$(grep -cE '^ok' <<<"$out")"
  else
    FAIL=$((FAIL + 1)); FAILED+=("$t"); printf 'FAILED  %-24s\n' "$t"
    printf '%s\n' "$out" | grep -vE '^ok' | tail -n 25 | sed 's/^/    | /'
  fi
done
echo "result: $OK ok, $FAIL failed"
[ $FAIL = 0 ]
