"""TOB-37: image input for tobu-fnx (Qwen3.8-Flash-Next, vision tower via --mmproj).

The server decodes (Pillow, EXIF orientation, RGB8) and checks limits; scaling happens in the engine (Gufo
ResizeImage, bicubic, grid 32 px). `grid()` computes the same dimensions in advance (tokens per image = gh*gw) – the engine
reports them back on IMAGE, the adapter compares.

Sources: `data:image/...;base64,...` (OpenAI image_url) and Anthropic `source.type == "base64"`. http(s) URLs only with
TOBU_IMAGE_FETCH=1 (phase 4, SSRF protection) – until then 400 with a reason. No `file:`.

Limits (environment, defaults): TOBU_IMAGE_MAX_MB 20 (encoded), 32 MP decoded, TOBU_IMAGE_MAX_PIXELS 1048576
after scaling (= ≤ 1,024 tokens/image), TOBU_IMAGE_MAX_COUNT 32 per request, TOBU_IMAGE_MAX_TOKENS 32768 per request
(agent clients resend all earlier images of the conversation every turn; older ones are mostly prefix hits).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import os
import re
from dataclasses import dataclass, field

PATCH, MERGE = 16, 2
FACTOR = PATCH * MERGE                    # 32 px per language-token side
MIN_PIXELS = 65536
MAX_PIXELS_LIMIT = 16777216
LOW_MAX_SIDE = 512
DECODED_MAX = 32 * 1024 * 1024            # like Gufo core::kMaxImagePixels
FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}


def limits() -> dict:
    return {"max_bytes": int(float(os.environ.get("TOBU_IMAGE_MAX_MB", "20")) * 2**20),
            "max_pixels": int(os.environ.get("TOBU_IMAGE_MAX_PIXELS", "1048576")),
            "max_count": int(os.environ.get("TOBU_IMAGE_MAX_COUNT", "32")),
            "max_tokens": int(os.environ.get("TOBU_IMAGE_MAX_TOKENS", "32768"))}


class MediaError(ValueError):
    pass


@dataclass
class Image:
    width: int
    height: int
    rgb: bytes = field(repr=False)
    detail: int = 0                       # 0 auto/high, 1 low
    max_pixels: int = 1048576
    sha: str = ""
    gh: int = 0
    gw: int = 0

    @property
    def tokens(self) -> int:
        return self.gh * self.gw


def _round_even(x: float) -> int:
    f = math.floor(x)
    part = x - f
    return int(f + (part > 0.5 or (part == 0.5 and (int(f) & 1))))


def grid(width: int, height: int, detail: int = 0, max_pixels: int = MAX_PIXELS_LIMIT) -> tuple[int, int]:
    """Dimensions after Gufo ResizeImage (prompt.cpp) -> (gh, gw) language tokens. Integer/float steps as there."""
    max_px = min(max(max_pixels, MIN_PIXELS), MAX_PIXELS_LIMIT)
    if width <= 0 or height <= 0 or width * height > DECODED_MAX or max(width, height) / min(width, height) > 200:
        raise MediaError("invalid image dimensions or aspect ratio")
    h = _round_even(height / FACTOR) * FACTOR
    w = _round_even(width / FACTOR) * FACTOR

    def raise_to_minimum():
        beta = math.sqrt(MIN_PIXELS / (width * height))
        return (math.ceil(height * beta / FACTOR) * FACTOR, math.ceil(width * beta / FACTOR) * FACTOR)

    if h * w > max_px:
        beta = math.sqrt(width * height / max_px)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < MIN_PIXELS:
        h, w = raise_to_minimum()
    if detail == 1 and (w > LOW_MAX_SIDE or h > LOW_MAX_SIDE):
        longest = max(width, height)

        def bound(side):
            return max(FACTOR, (side * LOW_MAX_SIDE // longest) // FACTOR * FACTOR)

        ah, aw = h, w
        h, w = bound(height), bound(width)
        if h * w < MIN_PIXELS:
            h, w = raise_to_minimum()
            if h * w > ah * aw:
                h, w = ah, aw
    return h // FACTOR, w // FACTOR


def decode(raw: bytes, detail: int = 0, lim: dict | None = None) -> Image:
    """Encoded bytes -> RGB8 in display orientation, limits checked, grid in advance."""
    from PIL import Image as PImage, ImageOps

    lim = lim or limits()
    if len(raw) > lim["max_bytes"]:
        raise MediaError(f"image too large: {len(raw)} bytes > {lim['max_bytes']}")
    PImage.MAX_IMAGE_PIXELS = DECODED_MAX           # decompression bomb -> error instead of memory
    try:
        im = PImage.open(io.BytesIO(raw))
        if im.format not in FORMATS:
            raise MediaError(f"image format {im.format!r} is not supported (PNG, JPEG, WebP, GIF)")
        if im.width * im.height > DECODED_MAX:
            raise MediaError("image has too many pixels")
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    except MediaError:
        raise
    except Exception as e:  # noqa: BLE001 – Pillow raises many things (UnidentifiedImageError, DecompressionBomb, OSError)
        raise MediaError(f"image could not be decoded: {type(e).__name__}") from e
    gh, gw = grid(im.width, im.height, detail, lim["max_pixels"])
    rgb = im.tobytes()
    sha = hashlib.sha256(f"{im.width}x{im.height}:{detail}:{lim['max_pixels']}:".encode() + rgb).hexdigest()
    return Image(im.width, im.height, rgb, detail, lim["max_pixels"], sha, gh, gw)


DATA_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+)?(;[^,]*)?,", re.S)


def from_url(url: str, detail: str | None = None, lim: dict | None = None) -> Image:
    d = 1 if detail == "low" else 0
    if not isinstance(url, str):
        raise MediaError("image_url.url must be a string")
    m = DATA_RE.match(url)
    if m:
        if ";base64" not in (m.group(2) or ""):
            raise MediaError("data: image URLs must be base64")
        return decode(_b64(url[m.end():], lim), d, lim)
    if url.startswith(("http://", "https://")):
        if os.environ.get("TOBU_IMAGE_FETCH") != "1":
            raise MediaError("http(s) image URLs are disabled on this server; send the image as a data: URL (base64)")
        return decode(fetch(url, lim), d, lim)
    raise MediaError("image URL must be a data: URL (base64)")


def from_anthropic(source: dict, lim: dict | None = None) -> Image:
    t = (source or {}).get("type")
    if t == "base64":
        mt = source.get("media_type") or ""
        if not mt.startswith("image/"):
            raise MediaError(f"image media_type {mt!r} is not supported")
        return decode(_b64(source.get("data") or "", lim), 0, lim)
    if t == "url":
        return from_url(source.get("url") or "", None, lim)
    raise MediaError(f"image source type {t!r} is not supported (base64, url)")


def _b64(s: str, lim: dict | None) -> bytes:
    lim = lim or limits()
    if len(s) > lim["max_bytes"] * 4 // 3 + 16:
        raise MediaError(f"image too large: > {lim['max_bytes']} bytes")
    try:
        return base64.b64decode(re.sub(r"\s+", "", s), validate=True)
    except (binascii.Error, ValueError) as e:
        raise MediaError("image data is not valid base64") from e


def check_request(images: list[Image], lim: dict | None = None):
    lim = lim or limits()
    if len(images) > lim["max_count"]:
        raise MediaError(f"too many images: {len(images)} > {lim['max_count']} per request (server setting "
                         f"TOBU_IMAGE_MAX_COUNT; agent clients resend all earlier images of the conversation every turn – "
                         f"start a new session or remove old screenshots)")
    tok = sum(i.tokens for i in images)
    if tok > lim["max_tokens"]:
        raise MediaError(f"images need {tok} tokens > {lim['max_tokens']} per request (server setting TOBU_IMAGE_MAX_TOKENS; "
                         f"use detail: low or smaller images)")


# ------------------------------------------------------------------------------------------- fetch (phase 4)
def public_ip(ip: str) -> bool:
    """Only globally reachable addresses (like Gufo IsPublicImageAddress): no loopback, private (RFC 1918), CGNAT
    100.64/10, link-local, ULA, multicast, reserved; v4-in-v6 (mapped, NAT64, 6to4, Teredo) by the inner v4."""
    import ipaddress

    a = ipaddress.ip_address(ip)
    if a.version == 6:
        inner = a.ipv4_mapped or a.sixtofour or (a.teredo[1] if a.teredo else None)
        if inner is None and a in ipaddress.ip_network("64:ff9b::/96"):
            inner = ipaddress.ip_address(int(a) & 0xFFFFFFFF)
        if inner is not None:
            return public_ip(str(inner))
    if a.version == 4 and a in ipaddress.ip_network("100.64.0.0/10"):
        return False
    return a.is_global and not (a.is_multicast or a.is_reserved or a.is_loopback or a.is_link_local or a.is_private)


def fetch(url: str, lim: dict | None = None, redirects: int = 3) -> bytes:
    """Fetch an http(s) image: DNS once, all addresses public, connection to the checked IP (no DNS rebinding),
    Host/SNI = original, redirects each checked anew, connect 3 s, total 10 s, abort via max_bytes,
    Content-Type image/*."""
    import http.client
    import socket
    import ssl
    import time
    import urllib.parse

    lim = lim or limits()
    t_end = time.monotonic() + 10.0
    for _ in range(redirects + 1):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password:
            raise MediaError("image URL must be http(s) without credentials")
        port = u.port or (443 if u.scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(u.hostname, port, type=socket.SOCK_STREAM)
        except OSError as e:
            raise MediaError(f"image host could not be resolved: {u.hostname}") from e
        ips = sorted({i[4][0] for i in infos})
        if not ips or not all(public_ip(ip) for ip in ips):
            raise MediaError("image URL points to a non-public address")
        ip = ips[0]
        left = max(0.5, t_end - time.monotonic())
        sock = socket.create_connection((ip, port), timeout=min(3.0, left))
        sock.settimeout(max(0.5, t_end - time.monotonic()))
        if u.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=u.hostname)
        conn = http.client.HTTPConnection(u.hostname, port, timeout=sock.gettimeout())
        conn.sock = sock                       # checked IP; the Host header stays the name
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        try:
            conn.request("GET", path, headers={"Host": u.netloc.split("@")[-1], "User-Agent": "tobu-image-fetch",
                                               "Accept": "image/*"})
            r = conn.getresponse()
            if r.status in (301, 302, 303, 307, 308) and r.getheader("Location"):
                url = urllib.parse.urljoin(url, r.getheader("Location"))
                continue
            if r.status != 200:
                raise MediaError(f"image URL returned HTTP {r.status}")
            ct = (r.getheader("Content-Type") or "").split(";")[0].strip().lower()
            if not ct.startswith("image/"):
                raise MediaError(f"image URL returned {ct or 'no content type'}, not an image")
            buf = bytearray()
            while True:
                if time.monotonic() > t_end:
                    raise MediaError("image download took longer than 10 s")
                chunk = r.read(65536)
                if not chunk:
                    return bytes(buf)
                buf += chunk
                if len(buf) > lim["max_bytes"]:
                    raise MediaError(f"image too large: > {lim['max_bytes']} bytes")
        except (OSError, http.client.HTTPException) as e:
            raise MediaError(f"image download failed: {type(e).__name__}") from e
        finally:
            conn.close()
    raise MediaError("too many redirects for the image URL")
