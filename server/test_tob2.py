"""TOB-2: acceptance test of the tobu server with the Messages API (Anthropic /v1/messages), no GPU, no model, no network.

    python test_tob2.py           (venv with numpy, tokenizers; no torch, no GPU, no model file)

Checks the server against the scope of TOB-2:

  A. /health ok, /v1/models in Anthropic form.
  B. Messages API over HTTP (test engine keyed: answer = fixed token sequence per marker in the prompt)
     B1  Non-stream: shape of the answer (id msg_..., type, role, model, content, stop_reason, stop_sequence, usage).
     B2  Stream: event sequence message_start ... message_stop, blocks == non-stream, same tokens.
     B3  thinking (enabled, adaptive): thinking block before the text.
     B4  Tools: tool_use with typed input and id toolu_..., stop_reason tool_use; follow-up request with tool_result.
     B5  max_tokens -> stop_reason max_tokens; stop_sequences -> stop_sequence.
     B6  tool_choice none/any/tool are accepted; count_tokens.
     B7  Errors in Anthropic format: max_tokens missing, prefill with thinking, server tools, unknown block,
         unknown tool_choice -> 400 invalid_request_error (prefill without thinking since TOB-12: 200); with --token:
         without/with wrong key 401, with x-api-key or Bearer 200; foreign host 403.
  C. Bit-identical (toy engine with real, prompt-dependent generation)
     C1  greedy (temperature 0): arm C with drafts == arm A without cache (plain greedy decode) == stream ==
         second round == default arm.
     C2  /v1/messages and /v1/chat/completions with the same history: the same prompt, the same tokens.
     C3  Sampling with seed: same request, same result.

The real Qwen tokenizer is not needed here: the test builds a byte tokenizer with
the same special tokens (<|im_end|>, <think>, <tool_call>, ...). Access keys in the test are placeholders.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILS: list[str] = []
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>"]
TOKEN = "test"                       # placeholder for --token (not a real key)
AN_TOOLS = [{"name": "read", "description": "Datei lesen", "input_schema": {
    "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "bash", "description": "Befehl", "input_schema": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
OA_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in AN_TOOLS]
ANSWERS = {
    "default": "Hallo zurück.<|im_end|>",
    "denk": "\nIch überlege kurz.\n</think>\n\nDas Ergebnis ist **42** – äöü 🙂.\n<|im_end|>",
    "tool": "Ich lese die Datei.\n\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/a.ts\n"
            "</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call><|im_end|>",
    "lang": "eins zwei drei vier fünf sechs sieben acht neun zehn " * 3 + "<|im_end|>",
    "stopp": "Anfang ### Ende, das nicht kommen darf.<|im_end|>",
}


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}{(' – ' + str(info)[:600]) if info and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------- helpers
def make_tokenizer(path: Path):
    """Byte tokenizer (BPE without merges, ByteLevel) with the special tokens of the Qwen templates. Additional ordinary
    tokens up to id ~6400, because the toy engine produces ids 1000...5999; each decodes to text."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    b2u = pre_tokenizers.ByteLevel.alphabet()
    vocab = {c: i for i, c in enumerate(sorted(b2u))}
    letters = "abcdefghijklmnopqrstuvwxyz"
    n = len(vocab)
    for a in letters:
        for b in letters:
            for c in letters:
                if n >= 6400:
                    break
                w = "Ġ" + a + b + c
                if w not in vocab:
                    vocab[w] = n
                    n += 1
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIAL)
    tok.save(str(path))
    return tok


def serve(engine: str, tokfile: Path, *extra: str):
    """Backend + HTTP handler like server.main(), but without loading real weights and without warm-up."""
    import server

    a = server.build_parser().parse_args(["--engine", engine, "--tokenizer", str(tokfile), "--port", "0",
                                          "--no-warmup", "--max-len", "32768", *extra])
    B = server.Backend(a)
    B.warm.set()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(B))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(url, body, headers=None, raw=False):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            b = resp.read()
            return resp.status, (b.decode() if raw else json.loads(b))
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b)
        except json.JSONDecodeError:
            return e.code, b.decode(errors="replace")


def get(url, headers=None):
    r = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def sse(raw: str) -> list[tuple[str, dict]]:
    out = []
    for part in raw.split("\n\n"):
        ev, data = None, None
        for line in part.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if ev and data:
            out.append((ev, json.loads(data)))
    return out


def an_collect(raw: str):
    """SSE -> (blocks, message from message_start, message_delta, event names)."""
    evs = sse(raw)
    blocks, msg, delta = [], None, None
    for ev, d in evs:
        if ev == "message_start":
            msg = d["message"]
        elif ev == "content_block_start":
            blocks.append(dict(d["content_block"]))
        elif ev == "content_block_delta":
            b, dl = blocks[d["index"]], d["delta"]
            if dl["type"] == "text_delta":
                b["text"] += dl["text"]
            elif dl["type"] == "thinking_delta":
                b["thinking"] += dl["thinking"]
            elif dl["type"] == "input_json_delta":
                b["_json"] = b.get("_json", "") + dl["partial_json"]
        elif ev == "message_delta":
            delta = d
    for b in blocks:
        if "_json" in b:
            b["input"] = json.loads(b.pop("_json"))
    return blocks, msg, delta, [e for e, _ in evs]


def norm(blocks):
    """Comparable shape of the blocks (without random tool_use ids)."""
    return [(b["type"], b.get("text") or b.get("thinking") or (b.get("name"), b.get("input"))) for b in blocks]


# ------------------------------------------------------------------------------------------------- B: API
def bodies(text, think=False, tools=True, **extra):
    an = {"model": "tobu", "system": "sys", "messages": [{"role": "user", "content": text}], "max_tokens": 200,
          "temperature": 0, **extra}
    if think:
        an["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    if tools:
        an["tools"] = AN_TOOLS
    return an


def both(base, an, headers=None):
    """Non-stream and stream -> (answer, stream blocks, message_start, message_delta, events) or None."""
    s1, j = post(base + "/v1/messages", an, headers)
    s2, raw = post(base + "/v1/messages", {**an, "stream": True}, headers, raw=True)
    if (s1, s2) != (200, 200):
        check("HTTP 200 (Stream und Nicht-Stream)", False, (s1, s2, j, raw[:300] if isinstance(raw, str) else raw))
        return None
    blocks, msg, delta, names = an_collect(raw)
    return j, blocks, msg, delta, names


def test_api(tokfile: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ANSWERS, f, ensure_ascii=False)
    httpd, base = serve("keyed:" + f.name, tokfile)
    try:
        s, h = get(base + "/health")
        check("A /health 200", s == 200 and h.get("status") == "ok", h)
        s, m = get(base + "/v1/models", {"anthropic-version": "2023-06-01"})
        check("A /v1/models in Anthropic-Form", s == 200 and m["data"][0]["type"] == "model" and "has_more" in m, m)

        r = both(base, bodies("Hallo", tools=False))
        if r:
            j, blocks, msg, delta, names = r
            check("B1 Form: id msg_…, type message, role assistant, model",
                  j["id"].startswith("msg_") and j["type"] == "message" and j["role"] == "assistant"
                  and j["model"] == "keyed", j)
            check("B1 Inhalt: ein text-Block, stop_reason end_turn, stop_sequence null",
                  j["content"] == [{"type": "text", "text": "Hallo zurück."}] and j["stop_reason"] == "end_turn"
                  and j["stop_sequence"] is None, j)
            u = j["usage"]
            check("B1 usage: input/output/cache_read/cache_creation",
                  {"input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"} <= set(u)
                  and u["input_tokens"] + u["cache_read_input_tokens"] > 0 and u["output_tokens"] > 0, u)
            check("B2 Stream: Ereignisfolge message_start, content_block_start/delta/stop, message_delta, message_stop",
                  names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"]
                  and names.count("content_block_start") == names.count("content_block_stop") == 1
                  and "content_block_delta" in names, names)
            check("B2 Stream == Nicht-Stream (Blöcke, stop_reason, Token-Hash)",
                  norm(blocks) == norm(j["content"]) and delta["delta"]["stop_reason"] == j["stop_reason"]
                  and delta["tobu"]["out_hash"] == j["tobu"]["out_hash"], (blocks, j["content"]))
            check("B2 message_start mit usage, message_delta mit output_tokens",
                  msg["usage"]["input_tokens"] > 0 and delta["usage"]["output_tokens"] == j["usage"]["output_tokens"])

        for kind in ("enabled", "adaptive"):
            an = bodies("Rechne [[denk]]", tools=False)
            an["thinking"] = {"type": kind} if kind == "adaptive" else {"type": kind, "budget_tokens": 1024}
            r = both(base, an)
            if r:
                j, blocks = r[0], r[1]
                check(f"B3 thinking {kind}: thinking-Block vor dem Text, Stream gleich",
                      [b["type"] for b in j["content"]] == ["thinking", "text"]
                      and j["content"][0]["thinking"] == "Ich überlege kurz."
                      and j["content"][1]["text"].startswith("Das Ergebnis ist **42**")
                      and norm(blocks) == norm(j["content"]), j["content"])
        s, j = post(base + "/v1/messages", bodies("Rechne [[denk]]", tools=False))
        check("B3 ohne thinking: kein thinking-Block (Standard aus)",
              s == 200 and all(b["type"] != "thinking" for b in j["content"]), j)

        r = both(base, bodies("Lies [[tool]]"))
        if r:
            j, blocks = r[0], r[1]
            tu = [b for b in j["content"] if b["type"] == "tool_use"]
            check("B4 tool_use: Text + Aufruf read mit typisierter Eingabe, id toolu_…",
                  [b["type"] for b in j["content"]] == ["text", "tool_use"] and len(tu) == 1
                  and tu[0]["id"].startswith("toolu_") and tu[0]["name"] == "read"
                  and tu[0]["input"] == {"path": "src/a.ts", "limit": 20}, j["content"])
            check("B4 stop_reason tool_use, Stream gleich",
                  j["stop_reason"] == "tool_use" and norm(blocks) == norm(j["content"]), (blocks, j["content"]))
            follow = bodies("Lies [[tool]]")
            follow["messages"] = [{"role": "user", "content": "Lies [[tool]]"},
                                  {"role": "assistant", "content": j["content"]},
                                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0]["id"],
                                                                "content": "export const a = 1;"},
                                                               {"type": "text", "text": "Danke [[default]]"}]}]
            s, j2 = post(base + "/v1/messages", follow)
            check("B4 Folgeanfrage mit tool_use/tool_result: 200, Antwort end_turn",
                  s == 200 and j2["stop_reason"] == "end_turn", j2)
            follow["messages"][2]["content"][0]["is_error"] = True
            s, j3 = post(base + "/v1/messages", follow)
            check("B4 tool_result mit is_error: 200, anderer Prompt",
                  s == 200 and j3["tobu"]["prompt_hash"] != j2["tobu"]["prompt_hash"], j3)

        s, j = post(base + "/v1/messages", bodies("Zähle [[lang]]", tools=False, max_tokens=9))
        check("B5 max_tokens 9 -> stop_reason max_tokens, output_tokens 9",
              s == 200 and j["stop_reason"] == "max_tokens" and j["usage"]["output_tokens"] == 9, j)
        r = both(base, bodies("Stopp [[stopp]]", tools=False, stop_sequences=["###"]))
        if r:
            j, blocks = r[0], r[1]
            check("B5 stop_sequences: Text endet davor, stop_reason/stop_sequence gemeldet, Stream gleich",
                  j["content"] == [{"type": "text", "text": "Anfang "}] and j["stop_reason"] == "stop_sequence"
                  and j["stop_sequence"] == "###" and norm(blocks) == norm(j["content"]), j)

        for tc in ({"type": "auto"}, {"type": "none"}, {"type": "any"}, {"type": "tool", "name": "read"},
                   {"type": "auto", "disable_parallel_tool_use": True}):
            s, j = post(base + "/v1/messages", bodies("Lies [[tool]]", tool_choice=tc))
            check(f"B6 tool_choice {tc}: angenommen (200)", s == 200, (s, j))
        s, j = post(base + "/v1/messages/count_tokens", {"model": "tobu", "system": "sys",
                                                         "messages": [{"role": "user", "content": "Hallo"}]})
        s2, j2 = post(base + "/v1/messages", bodies("Hallo", tools=False))
        check("B6 count_tokens == input_tokens + cache_read der echten Anfrage",
              s == 200 and s2 == 200 and j["input_tokens"] ==
              j2["usage"]["input_tokens"] + j2["usage"]["cache_read_input_tokens"], (j, j2.get("usage")))

        def err(name, body, status=400, kind="invalid_request_error", headers=None):
            s, j = post(base + "/v1/messages", body, headers)
            check(f"B7 {name} -> {status} {kind} (Anthropic-Form)",
                  s == status and isinstance(j, dict) and j.get("type") == "error" and j["error"]["type"] == kind
                  and j["error"].get("message"), (s, j))

        err("max_tokens fehlt", {"messages": [{"role": "user", "content": "x"}]})
        err("messages leer", {"max_tokens": 10, "messages": []})
        err("Prefill mit thinking (TOB-12: ohne thinking erlaubt)", {**bodies("x", think=True, tools=False), "messages": [
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]})
        s, j = post(base + "/v1/messages", {**bodies("x [[default]]", tools=False), "messages": [
            {"role": "user", "content": "x [[default]]"}, {"role": "assistant", "content": "Also:"}]})
        check("B7 Prefill ohne thinking (TOB-12) -> 200, Antwort nur die Fortsetzung",
              s == 200 and j["content"] == [{"type": "text", "text": "Hallo zurück."}], (s, j))
        err("Server-Tool (web_search)", {**bodies("x"), "tools": [{"type": "web_search_20250305", "name": "web_search"}]})
        err("unbekannter Block-Typ", {"max_tokens": 10, "messages": [
            {"role": "user", "content": [{"type": "container_upload", "file_id": "f"}]}]})
        err("unbekanntes tool_choice", {**bodies("x"), "tool_choice": {"type": "irgendwas"}})
        err("unbekannte Rolle", {"max_tokens": 10, "messages": [{"role": "robot", "content": "x"}]})
        err("output_format ohne json_schema", {**bodies("x", tools=False), "output_format": {"type": "text"}})
        s, j = post(base + "/v1/messages", bodies("x", tools=False), {"Host": "evil.example"})
        check("B7 fremder Host-Header -> 403 permission_error (Anthropic-Form)",
              s == 403 and j.get("type") == "error" and j["error"]["type"] == "permission_error", (s, j))
    finally:
        httpd.shutdown()

    # access with --token (placeholder)
    httpd, base = serve("keyed:" + f.name, tokfile, "--token", TOKEN)
    try:
        an = bodies("Hallo", tools=False)
        for name, hd, want in (("ohne Schlüssel", {}, 401), ("falscher x-api-key", {"x-api-key": "falsch"}, 401),
                               ("x-api-key", {"x-api-key": TOKEN}, 200),
                               ("Authorization: Bearer", {"Authorization": "Bearer " + TOKEN}, 200)):
            s, j = post(base + "/v1/messages", an, hd)
            ok = s == want and (want == 200 or (j.get("type") == "error"
                                               and j["error"]["type"] == "authentication_error"))
            check(f"B7 --token, {name} -> {want}", ok, (s, j))
        s, h = get(base + "/health")
        check("B7 --token: /health bleibt ohne Schlüssel erreichbar", s == 200, h)
    finally:
        httpd.shutdown()


# ------------------------------------------------------------------------------------------------- C: bit-identical
def test_bitexact(tokfile: Path):
    httpd, base = serve("toy", tokfile)
    try:
        txt = "Die Katze sitzt auf der Matte. Die Katze sitzt auf dem Dach. Die Katze sitzt"
        an = {"model": "tobu", "messages": [{"role": "user", "content": txt}], "max_tokens": 40, "temperature": 0}
        s0, ref = post(base + "/v1/messages", an, {"X-Tobu-Arm": "A", "X-Tobu-Cache": "off"})
        C = {"X-Tobu-Arm": "C"}                           # default arm: engine drafts
        s1, dflt = post(base + "/v1/messages", an, C)
        s2, raw = post(base + "/v1/messages", {**an, "stream": True}, C, raw=True)
        _, _, delta, _ = an_collect(raw) if s2 == 200 else (None, None, {"tobu": {}}, None)
        s3, again = post(base + "/v1/messages", an, C)    # second round (context cache of the drafts filled)
        s3b, plain = post(base + "/v1/messages", an)      # default arm of the parser (A)
        ok = (s0, s1, s2, s3, s3b) == (200,) * 5
        check("C1 greedy: alle Anfragen 200", ok, (s0, s1, s2, s3, s3b))
        if ok:
            h = {x["tobu"]["out_hash"] for x in (ref, dflt, delta, again, plain)}
            check("C1 Arm C (Entwürfe) == Arm A ohne Cache (reiner greedy Decode) == Stream == zweite Runde == Standard",
                  len(h) == 1 and ref["usage"]["output_tokens"] == dflt["usage"]["output_tokens"] > 0,
                  (ref["tobu"], dflt["tobu"]))
            check("C1 Arm wird gemeldet (Referenz A, Entwürfe C)",
                  ref["tobu"]["arm"] == "A" and dflt["tobu"]["arm"] == "C", (ref["tobu"]["arm"], dflt["tobu"]["arm"]))
        oa = {"model": "tobu", "messages": [{"role": "user", "content": txt}], "max_tokens": 40, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}}
        s4, oj = post(base + "/v1/chat/completions", oa)
        check("C2 Messages-API == Chat Completions: gleicher Prompt-Hash, gleiche Tokens",
              s4 == 200 and s1 == 200 and oj["tobu"]["prompt_hash"] == dflt["tobu"]["prompt_hash"]
              and oj["tobu"]["out_hash"] == dflt["tobu"]["out_hash"], (oj.get("tobu"), dflt.get("tobu")))
        sm = {**an, "temperature": 0.8, "seed": 7}
        s5, x = post(base + "/v1/messages", sm)
        s6, y = post(base + "/v1/messages", sm)
        check("C3 Sampling mit seed: gleiche Anfrage -> gleiche Ausgabe",
              (s5, s6) == (200, 200) and x["tobu"]["out_hash"] == y["tobu"]["out_hash"], (x.get("tobu"), y.get("tobu")))
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        make_tokenizer(tokfile)
        test_api(tokfile)
        test_bitexact(tokfile)
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
