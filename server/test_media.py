"""TOB-37: media.py – grid (= Gufo ResizeImage), decoding/limits, SSRF protection and fetch (phase 4).

No outbound network: a local HTTP server delivers images/redirects; for the positive cases the
IP check is opened in the test (the real check rejects 127.0.0.1 – first case). Needs Pillow.
python3 -I test_media.py
"""

from __future__ import annotations

import io
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import media  # noqa: E402

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}" + (f" – {str(detail)[:200]}" if not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def png(w=64, h=48) -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (w, h), (200, 10, 10)).save(b, "PNG")
    return b.getvalue()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path
        if p == "/img.png":
            body, ct = png(), "image/png"
        elif p == "/text":
            body, ct = b"hallo", "text/plain"
        elif p == "/big.png":
            body, ct = b"\0" * (3 * 2**20), "image/png"
        elif p.startswith("/redir/"):
            n = int(p.split("/")[2])
            self.send_response(302)
            self.send_header("Location", f"/redir/{n - 1}" if n > 1 else "/img.png")
            self.end_headers()
            return
        elif p == "/to-private":
            self.send_response(302)
            self.send_header("Location", "http://10.0.0.1/img.png")
            self.end_headers()
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    # grid: known values (engine comparison in test_vision.py)
    check("grid 640x480 -> 15x20", media.grid(640, 480) == (15, 20))
    check("grid 4000x3000 @1MP -> 27x36", media.grid(4000, 3000, 0, 1 << 20) == (27, 36))
    check("grid 4000x3000 low -> 12x16", media.grid(4000, 3000, 1, 1 << 20) == (12, 16))
    try:
        media.grid(10, 3000)
        check("grid Seitenverhältnis > 200 -> Fehler", False)
    except media.MediaError:
        check("grid Seitenverhältnis > 200 -> Fehler", True)
    # decoding, limits
    im = media.decode(png(100, 50))
    check("decode PNG -> RGB8, Raster", im.width == 100 and len(im.rgb) == 100 * 50 * 3 and im.tokens > 0, im)
    for name, raw in (("kein Bild", b"nope"), ("zu groß", b"\0" * 100)):
        try:
            media.decode(raw, 0, {**media.limits(), "max_bytes": 50})
            check(f"decode {name} -> Fehler", False)
        except media.MediaError:
            check(f"decode {name} -> Fehler", True)
    lim = media.limits()
    check("Standard: 32 Bilder, 32768 Bild-Tokens je Anfrage", (lim["max_count"], lim["max_tokens"]) == (32, 32768), lim)
    small = media.Image(**{**im.__dict__, "gh": 16, "gw": 16}) if hasattr(im, "__dict__") else im
    try:
        media.check_request([small] * 32)
        check("32 Bilder -> ok", True)
    except media.MediaError as e:
        check("32 Bilder -> ok", False, e)
    try:
        media.check_request([small] * 33)
        check("33 Bilder -> Fehler", False)
    except media.MediaError as e:
        check("33 Bilder -> Fehler nennt TOBU_IMAGE_MAX_COUNT und den Grund (Bilder je Runde neu)",
              "TOBU_IMAGE_MAX_COUNT" in str(e) and "33 > 32" in str(e) and "resend" in str(e), e)
    try:
        media.check_request([small] * 4, {**lim, "max_tokens": 10})
        check("Bild-Tokens über der Grenze -> Fehler", False)
    except media.MediaError as e:
        check("Bild-Tokens über der Grenze -> Fehler nennt TOBU_IMAGE_MAX_TOKENS", "TOBU_IMAGE_MAX_TOKENS" in str(e), e)
    os.environ["TOBU_IMAGE_MAX_COUNT"] = "4"
    try:
        media.check_request([small] * 5)
        check("TOBU_IMAGE_MAX_COUNT=4 -> 5 Bilder Fehler", False)
    except media.MediaError:
        check("TOBU_IMAGE_MAX_COUNT=4 -> 5 Bilder Fehler (Einstellung wirkt)", True)
    finally:
        os.environ.pop("TOBU_IMAGE_MAX_COUNT", None)
    # SSRF
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    os.environ["TOBU_IMAGE_FETCH"] = "1"
    try:
        media.from_url(base + "/img.png")
        check("SSRF: 127.0.0.1 abgelehnt", False)
    except media.MediaError as e:
        check("SSRF: 127.0.0.1 abgelehnt", "non-public" in str(e), e)
    real = media.public_ip
    media.public_ip = lambda ip: ip == "127.0.0.1" or real(ip)   # test: local server counts as public
    try:
        im = media.from_url(base + "/img.png")
        check("Abruf: Bild geladen", im.width == 64 and im.height == 48, im)
        im = media.from_url(base + "/redir/3")
        check("Abruf: 3 Weiterleitungen ok", im.width == 64)
        for name, path, want in (("4 Weiterleitungen", "/redir/4", "redirects"), ("text/plain", "/text", "not an image"),
                                 ("404", "/nix", "HTTP 404"), ("Weiterleitung auf 10.0.0.1", "/to-private", "non-public")):
            try:
                media.from_url(base + path)
                check(f"Abruf: {name} -> Fehler", False)
            except media.MediaError as e:
                check(f"Abruf: {name} -> Fehler ({want})", want in str(e), e)
        try:
            media.fetch(base + "/big.png", {**media.limits(), "max_bytes": 2**20})
            check("Abruf: über max_bytes -> Abbruch", False)
        except media.MediaError as e:
            check("Abruf: über max_bytes -> Abbruch", "too large" in str(e), e)
    finally:
        media.public_ip = real
        os.environ.pop("TOBU_IMAGE_FETCH", None)
    try:
        media.from_url(base + "/img.png")
        check("ohne TOBU_IMAGE_FETCH: http aus", False)
    except media.MediaError as e:
        check("ohne TOBU_IMAGE_FETCH: http aus", "disabled" in str(e), e)
    srv.shutdown()
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
