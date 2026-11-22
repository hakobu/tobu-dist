"""TOB-30: a request during warm-up must not block the server (no GPU, no model, no network).

    python test_tob30.py          (venv with numpy, tokenizers)

Reproduced like server.main(): HTTP server runs, B.warm is cleared, startup then puts warm-up and
clear_cache via WORKER.submit(...).result() onto the single worker thread. Before TOB-30 a chat request from
that time waited in the worker for B.warm -> startup waited for the same thread -> deadlock.

  1. /v1/models and /health before warm-up ends: 503 (OpenAI and Anthropic form), Retry-After.
  2. POST /v1/chat/completions and /v1/messages (also streamed) right after start: immediately 503, Retry-After.
  3. The startup step via the worker (like warm-up / clear_cache) then runs through (no deadlock).
  4. After B.warm.set(): /v1/models 200, requests 200.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_tob2 import ANSWERS, make_tokenizer  # noqa: E402

FAILS: list[str] = []


def check(msg, ok, info=None):
    ok = bool(ok)
    print(("ok   " if ok else "FAIL ") + msg + ("" if ok or info is None else f"  ({info})"), flush=True)
    if not ok:
        FAILS.append(msg)


def call(url, body=None, headers=None, timeout=10):
    """-> (status, headers, body as dict or text); timeout -> (None, {}, error)."""
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            b = resp.read()
            st, hd = resp.status, dict(resp.headers)
    except urllib.error.HTTPError as e:
        b, st, hd = e.read(), e.code, dict(e.headers)
    except OSError as e:                 # timeout: the server hangs
        return None, {}, repr(e)
    try:
        return st, hd, json.loads(b)
    except json.JSONDecodeError:
        return st, hd, b.decode(errors="replace")


def start(tokfile: Path, answers: str):
    """Like server.main() up to before warm-up: backend built in the worker, warm cleared, HTTP running."""
    import server

    a = server.build_parser().parse_args(["--engine", "keyed:" + answers, "--tokenizer", str(tokfile), "--port", "0",
                                          "--max-len", "32768"])
    B = server.WORKER.submit(server.Backend, a).result(timeout=60)
    B.warm.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(B))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return server, B, httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_start(tokfile: Path, answers: str):
    server, B, httpd, base = start(tokfile, answers)
    oa = {"model": "tobu", "max_tokens": 16, "temperature": 0, "messages": [{"role": "user", "content": "Hallo"}]}
    an = {"model": "tobu", "max_tokens": 16, "temperature": 0, "messages": [{"role": "user", "content": "Hallo"}]}
    AV = {"anthropic-version": "2023-06-01"}
    try:
        s, h, j = call(base + "/v1/models")
        check("1 /v1/models vor dem Aufwärmen: 503 mit Retry-After", s == 503 and h.get("Retry-After"), (s, j))
        s, h, j = call(base + "/v1/models", headers=AV)
        check("1 /v1/models (Anthropic) vor dem Aufwärmen: 503 overloaded_error",
              s == 503 and isinstance(j, dict) and j.get("type") == "error"
              and j["error"]["type"] == "overloaded_error", (s, j))
        s, _, j = call(base + "/health")
        check("1 /health vor dem Aufwärmen: 503 loading", s == 503 and j.get("status") == "loading", (s, j))

        res = {}

        def fire(name, path, body, headers=None):
            t0 = time.perf_counter()
            res[name] = (*call(base + path, body, headers), time.perf_counter() - t0)

        ts = [threading.Thread(target=fire, args=x) for x in (
            ("oa", "/v1/chat/completions", oa), ("oa_stream", "/v1/chat/completions", {**oa, "stream": True}),
            ("an", "/v1/messages", an, AV), ("an_stream", "/v1/messages", {**an, "stream": True}, AV))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(15)
        for name, (s, h, j, dt) in sorted(res.items()):
            check(f"2 {name} direkt nach dem Start: 503 mit Retry-After in {dt:.2f} s",
                  s == 503 and h.get("Retry-After") and dt < 5, (s, j))
        s, h, j, _ = res.get("an", (None, {}, None, 0))
        check("2 Anthropic-Fehlerform (type error, overloaded_error)",
              isinstance(j, dict) and j.get("type") == "error" and j["error"]["type"] == "overloaded_error", j)
        s, h, j, _ = res.get("oa", (None, {}, None, 0))
        check("2 OpenAI-Fehlerform (error.message)", isinstance(j, dict) and "message" in j.get("error", {}), j)

        # 3. startup step via the same worker as in the real start (warm-up, clear_cache before the slots)
        try:
            server.WORKER.submit(lambda: "ok").result(timeout=10)
            ok = True
        except Exception as e:           # noqa: BLE001  TimeoutError: worker hangs on a request
            ok = False
            check("3 Worker frei nach Anfragen während des Aufwärmens (kein Deadlock)", False, repr(e))
        if ok:
            check("3 Worker frei nach Anfragen während des Aufwärmens (kein Deadlock)", True)
        try:
            server.WORKER.submit(B.warmup).result(timeout=60)
            check("3 Aufwärmen über den Worker läuft durch", True)
        except Exception as e:           # noqa: BLE001
            check("3 Aufwärmen über den Worker läuft durch", False, repr(e))

        B.warm.set()
        s, _, j = call(base + "/v1/models")
        check("4 /v1/models nach dem Aufwärmen: 200", s == 200 and j["data"][0]["id"] == B.model, (s, j))
        s, _, j = call(base + "/health")
        check("4 /health nach dem Aufwärmen: 200", s == 200 and j.get("status") == "ok", (s, j))
        s, _, j = call(base + "/v1/chat/completions", oa, timeout=60)
        check("4 Chat nach dem Aufwärmen: 200", s == 200 and j["choices"][0]["finish_reason"] == "stop", (s, j))
        s, _, j = call(base + "/v1/messages", an, AV, timeout=60)
        check("4 Messages nach dem Aufwärmen: 200", s == 200 and j.get("type") == "message", (s, j))
    finally:
        B.warm.set()                     # old state: release hanging jobs so the test ends
        httpd.shutdown()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        make_tokenizer(tokfile)
        ans = Path(td) / "answers.json"
        ans.write_text(json.dumps(ANSWERS, ensure_ascii=False), encoding="utf-8")
        test_start(tokfile, str(ans))
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
