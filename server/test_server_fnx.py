"""Server + flashnext engine binding over HTTP, against the stand-in fixtures/fake_fnx.py - no GPU, no model.

    python test_server_fnx.py          (Python with numpy, tokenizers, jinja2)

Backend like server.main(), the engine is FlashNextEngine.from_proc(fake_fnx.py) with the byte tokenizer from test_tob2.
Checked:
  1. greedy: arm C (engine drafts, several tokens per round) == arm A with X-Tobu-Cache: off (cold start, single steps)
     == stream == Anthropic path (same prompt hash, same output tokens).
  2. Prefix memory: the same request again -> hit (cached_tokens > 0, cache_src ram), same output.
  3. /health, /props, /tobu/metrics (last request, no contents), /chat (page with CSP).
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_tob2 import make_tokenizer, post  # noqa: E402

FAKE = HERE / "fixtures" / "fake_fnx.py"
FAILS: list[str] = []


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}{(' – ' + str(info)[:600]) if info and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def serve(tokfile: Path):
    import server
    from engine_flashnext import FlashNextEngine

    a = server.build_parser().parse_args(["--engine", "toy", "--tokenizer", str(tokfile), "--port", "0",
                                          "--no-warmup", "--max-len", "4096"])
    B = server.Backend(a)                        # tokenizer, template, grammar as usual; swap the engine afterwards
    vocab = B.tok.t.get_vocab_size()
    B.eng = FlashNextEngine.from_proc([sys.executable, "-I", str(FAKE), "--vocab", str(vocab), "--context", "4096"],
                                      hf_tok=B.tok.t, template=None, max_len=4096)
    B.warm.set()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(B))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, B, f"http://127.0.0.1:{httpd.server_address[1]}"


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read(), r.headers


def main():
    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        make_tokenizer(tokfile)
        httpd, B, base = serve(tokfile)
        try:
            msgs = [{"role": "system", "content": "Du bist ein Test. " * 20},
                    {"role": "user", "content": "Zähle bitte weiter: eins zwei drei vier"}]
            oa = {"model": "tobu", "messages": msgs, "max_tokens": 60, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}}
            s0, ref = post(base + "/v1/chat/completions", oa, {"X-Tobu-Arm": "A", "X-Tobu-Cache": "off"})
            s1, c = post(base + "/v1/chat/completions", oa)
            ok = (s0, s1) == (200, 200)
            check("1 HTTP 200 (Arm A kalt, Arm C)", ok, (s0, s1, ref, c))
            if not ok:
                return
            check("1 Arm C == Arm A kalt (gleiche Ausgabe-Tokens)",
                  c["tobu"]["out_hash"] == ref["tobu"]["out_hash"] and c["tobu"]["prompt_hash"] == ref["tobu"]["prompt_hash"],
                  (c["tobu"], ref["tobu"]))
            check("1 Arm C rechnet mit Engine-Entwürfen (mehr als 1 Token je Runde), Arm A nicht",
                  c["timings"]["tokens_per_round"] > 1.0 and ref["timings"]["tokens_per_round"] <= 1.0,
                  (c["timings"]["tokens_per_round"], ref["timings"]["tokens_per_round"]))
            s2, raw = post(base + "/v1/chat/completions", {**oa, "stream": True}, raw=True)
            txt = "".join((json.loads(ln[5:])["choices"] or [{}])[0].get("delta", {}).get("content") or ""
                          for ln in raw.splitlines() if ln.startswith("data:") and "[DONE]" not in ln)
            check("1 Stream == Nicht-Stream (Text)", s2 == 200 and txt == c["choices"][0]["message"]["content"],
                  (txt[:200], c["choices"][0]["message"]["content"][:200]))
            an = {"model": "tobu", "system": msgs[0]["content"], "messages": msgs[1:], "max_tokens": 60,
                  "temperature": 0}
            s3, j = post(base + "/v1/messages", an, {"anthropic-version": "2023-06-01"})
            check("1 Anthropic-Weg: gleicher Prompt, gleiche Tokens",
                  s3 == 200 and j["tobu"]["prompt_hash"] == c["tobu"]["prompt_hash"]
                  and j["tobu"]["out_hash"] == c["tobu"]["out_hash"], j.get("tobu"))
            check("2 Prefix-Speicher: Treffer bei der Wiederholung, gleiche Ausgabe",
                  j["tobu"]["cached_tokens"] > 0 and j["tobu"]["timings"]["cache_src"] == "ram", j["tobu"].get("timings"))
            s, body, _ = get(base + "/health")
            h = json.loads(body)
            check("3 /health ok", s == 200 and h["status"] == "ok", h)
            s, body, _ = get(base + "/props")
            check("3 /props n_ctx", s == 200 and json.loads(body)["n_ctx"] == 4096, body[:200])
            s, body, _ = get(base + "/tobu/metrics")
            m = json.loads(body)
            check("3 /tobu/metrics: letzte Anfrage, keine Inhalte",
                  s == 200 and m["requests"] and m["requests"][-1]["api"] == "anthropic" and "Zähle" not in body.decode(),
                  m.get("requests", [])[-1:])
            s, body, hd = get(base + "/chat")
            check("3 /chat: Seite mit CSP", s == 200 and b"<html" in body.lower() and "default-src 'none'"
                  in (hd.get("Content-Security-Policy") or ""), hd)
        finally:
            httpd.shutdown()
            B.eng.close()


if __name__ == "__main__":
    main()
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
