"""Browser chat with --token (review S1): the page never contains the token, login once via ?token=, then cookie.

    python test_chat_auth.py          (Python with numpy, tokenizers, jinja2)

Backend like server.main() with the toy engine and the byte tokenizer from test_tob2. Checked:
  1. /chat and / without token -> 401; wrong ?token= -> 401; API without token -> 401.
  2. /chat?token=<token> -> 303 to /chat, Set-Cookie tobu_session (HttpOnly, SameSite=Strict, Path=/), the cookie
     value is not the token.
  3. /chat with the cookie -> 200, the page does not contain the token.
  4. API (/chat/info, /v1/models, POST /v1/chat/completions) with the cookie -> 200; wrong cookie -> 401; cookie
     with a foreign Origin or Sec-Fetch-Site: cross-site -> 401; Bearer still works.
  5. --token-file: file created with mode 0600.
  6. without --token: /chat is open as before (host check only), page without token meta.
"""

from __future__ import annotations

import http.client
import json
import os
import stat
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_tob2 import make_tokenizer  # noqa: E402

TOKEN = "test-chat-token-123"          # placeholder for --token (not a real key)
FAILS: list[str] = []


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}{(' – ' + str(info)[:600]) if info and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def serve(tokfile: Path, *extra: str):
    import server

    a = server.build_parser().parse_args(["--engine", "toy", "--tokenizer", str(tokfile), "--port", "0",
                                          "--no-warmup", "--max-len", "4096", *extra])
    B = server.Backend(a)
    B.warm.set()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(B))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, B, httpd.server_address[1]


def req(port, method, path, headers=None, body=None):
    """One request without following redirects: (status, headers, body bytes)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        data = json.dumps(body).encode() if body is not None else None
        h = {"Content-Type": "application/json"} if data is not None else {}
        c.request(method, path, body=data, headers={**h, **(headers or {})})
        r = c.getresponse()
        return r.status, r.headers, r.read()
    finally:
        c.close()


def main():
    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        make_tokenizer(tokfile)

        httpd, B, port = serve(tokfile, "--token", TOKEN)
        try:
            for path in ("/chat", "/"):
                s, _, b = req(port, "GET", path)
                check(f"1 {path} ohne Token -> 401", s == 401 and TOKEN.encode() not in b, (s, b[:200]))
            s, _, b = req(port, "GET", "/chat?token=falsch")
            check("1 /chat?token=falsch -> 401", s == 401 and TOKEN.encode() not in b, (s, b[:200]))
            s, _, _ = req(port, "GET", "/chat/info")
            check("1 /chat/info ohne Token -> 401", s == 401, s)

            s, h, _ = req(port, "GET", "/chat?token=" + TOKEN)
            sc = h.get("Set-Cookie") or ""
            parts = [x.strip() for x in sc.split(";")]
            cookie = parts[0] if parts and parts[0].startswith("tobu_session=") else ""
            value = cookie.split("=", 1)[1] if cookie else ""
            check("2 /chat?token= -> 303 nach /chat", s == 303 and h.get("Location") == "/chat", (s, dict(h)))
            check("2 Set-Cookie tobu_session; HttpOnly; SameSite=Strict; Path=/",
                  bool(cookie) and "HttpOnly" in parts and "SameSite=Strict" in parts and "Path=/" in parts, sc)
            check("2 Cookie enthält nicht das Token", bool(value) and TOKEN not in value and value != TOKEN, sc)

            ck = {"Cookie": cookie}
            s, h, b = req(port, "GET", "/chat", ck)
            check("3 /chat mit Cookie -> 200 (Seite)", s == 200 and b"<html" in b.lower(), (s, b[:200]))
            check("3 Seite enthält das Token nicht", TOKEN.encode() not in b and b"tobu-key" not in b, b[:300])

            s, _, b = req(port, "GET", "/chat/info", ck)
            check("4 /chat/info mit Cookie -> 200", s == 200, (s, b[:200]))
            s, _, b = req(port, "GET", "/v1/models", ck)
            check("4 /v1/models mit Cookie -> 200", s == 200, (s, b[:200]))
            body = {"model": "tobu", "messages": [{"role": "user", "content": "Hallo"}], "max_tokens": 8,
                    "temperature": 0}
            same = {**ck, "Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin"}
            s, _, b = req(port, "POST", "/v1/chat/completions", same, body)
            check("4 POST /v1/chat/completions mit Cookie (same-origin) -> 200", s == 200, (s, b[:300]))
            s, _, b = req(port, "GET", "/chat/info", {"Cookie": "tobu_session=falsch"})
            check("4 falsches Cookie -> 401", s == 401, (s, b[:200]))
            s, _, b = req(port, "GET", "/chat/info", {"Cookie": "tobu_session=" + TOKEN})
            check("4 rohes Token als Cookie -> 401", s == 401, (s, b[:200]))
            s, _, b = req(port, "POST", "/v1/chat/completions", {**ck, "Origin": "http://127.0.0.1:3000"}, body)
            check("4 Cookie mit fremdem Origin -> 401", s == 401, (s, b[:200]))
            s, _, b = req(port, "POST", "/v1/chat/completions", {**ck, "Sec-Fetch-Site": "cross-site"}, body)
            check("4 Cookie mit Sec-Fetch-Site: cross-site -> 401", s == 401, (s, b[:200]))
            s, _, b = req(port, "POST", "/v1/chat/completions", {"Authorization": "Bearer " + TOKEN}, body)
            check("4 Bearer weiterhin -> 200", s == 200, (s, b[:200]))
            s, _, b = req(port, "POST", "/v1/chat/completions", {"Authorization": "Bearer falsch"}, body)
            check("4 falscher Bearer -> 401", s == 401, (s, b[:200]))
            s, _, _ = req(port, "GET", "/health")
            check("4 /health ohne Token weiter offen", s == 200, s)
        finally:
            httpd.shutdown()

        tf = Path(td) / "sub" / "token"
        httpd, B, port = serve(tokfile, "--token-file", str(tf))
        try:
            mode = stat.S_IMODE(os.stat(tf).st_mode)
            check("5 --token-file: Datei 0600 mit Token", mode == 0o600 and tf.read_text().strip() == B.api_token,
                  oct(mode))
            s, h, _ = req(port, "GET", "/chat?token=" + B.api_token)
            check("5 --token-file: Login über ?token= -> 303", s == 303 and "tobu_session=" in (h.get("Set-Cookie") or ""),
                  s)
        finally:
            httpd.shutdown()

        httpd, B, port = serve(tokfile)
        try:
            s, h, b = req(port, "GET", "/chat")
            check("6 ohne --token: /chat offen (200), ohne Cookie-Zwang", s == 200 and not h.get("Set-Cookie"), s)
            s, _, b = req(port, "GET", "/chat", {"Host": "evil.example"})
            check("6 ohne --token: fremder Host -> 403", s == 403, s)
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    main()
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
