"""TOB-12: close gaps in the Messages API left by TOB-2 (pure API compatibility), no GPU, no model, no network.

    python test_tob12.py              (venv with numpy, tokenizers; no torch, no GPU, no model file)
    python test_tob12.py --golden     rewrites fixtures/tob12_prompts.json (only on the state before TOB-12!)

Checks:
  P   Same prompt: every request of the CORPUS collection (Anthropic and OpenAI, templates qwen35 and bonsai) gives
      the same prompt hash as on main 07962aa (before TOB-12, fixtures/tob12_prompts.json).
      Intentional exception: defer_loading_reference - a tool loaded via tool_reference has since been
      in the prompt (as in the Anthropic API; before it was missing and the grammar redirected calls to Bash).
  E   Errors on /v1/messages* in Anthropic form: broken JSON, unknown path (POST/GET), /v1/messages/batches
      (404, not supported); error types outside the Anthropic list are mapped to api_error.
  R   Header request-id (req_...) on all responses, in the error JSON as request_id.
  F   Fields: mcp_servers and container -> 400; metadata, service_tier, cache_control, citations -> tobu.ignored.
  S   search_result as text (user and tool_result); container_upload stays 400.
  V   Parameter limits only for /v1/messages (temperature, top_p, top_k, max_tokens); /chat/completions unchanged.
  X   Prefill (last message assistant): prompt = prompt without prefill + prefill text, answer only the continuation;
      with thinking 400.
  T   Stream form: thinking block with signature_delta before content_block_stop.

No access keys occur here; placeholders as in test_tob2.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_tob2 as t2  # noqa: E402
from test_tob2 import AN_TOOLS, OA_TOOLS, an_collect, check, get, post  # noqa: E402

GOLDEN = HERE / "fixtures" / "tob12_prompts.json"
ANSWERS = {
    "default": "Hallo zurück.<|im_end|>",
    "denk": "\nIch überlege kurz.\n</think>\n\nDas Ergebnis ist 42.\n<|im_end|>",
    "weiter": " und weiter geht es.<|im_end|>",
}
TU = "toolu_0123456789abcdef01234567"

# ------------------------------------------------------------------------------------- collection (prompt equality)
CORPUS_AN = {
    "einfach": {"messages": [{"role": "user", "content": "Hallo"}]},
    "system_tools": {"system": "Du bist hilfreich.", "tools": AN_TOOLS,
                     "messages": [{"role": "user", "content": "Lies die Datei"}]},
    "system_liste_billing_cache": {
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cch=00000;"},
                   {"type": "text", "text": "Du bist Claude Code.", "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "Arbeite sorgfältig.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hallo",
                                                   "cache_control": {"type": "ephemeral"}}]}]},
    "thinking_enabled": {"thinking": {"type": "enabled", "budget_tokens": 2048},
                         "messages": [{"role": "user", "content": "Rechne"}]},
    "thinking_adaptive_tools": {"thinking": {"type": "adaptive"}, "tools": AN_TOOLS,
                                "messages": [{"role": "user", "content": "Rechne"}]},
    "tool_verlauf": {"tools": AN_TOOLS, "messages": [
        {"role": "user", "content": "Lies a.ts"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Ich lese.", "signature": ""},
            {"type": "text", "text": "Ich lese die Datei."},
            {"type": "tool_use", "id": TU, "name": "read", "input": {"path": "src/a.ts", "limit": 20}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": TU, "content": [{"type": "text", "text": "export const a = 1;"}]},
            {"type": "text", "text": "Danke"}]}]},
    "tool_fehler": {"tools": AN_TOOLS, "messages": [
        {"role": "user", "content": "Lies a.ts"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": TU, "name": "bash", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": TU, "content": "nicht gefunden",
                                      "is_error": True}]}]},
    "redacted_und_string": {"messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "xyz"},
                                          {"type": "text", "text": "b"}]},
        {"role": "user", "content": "c"}]},
    "system_rolle_mitte": {"system": "S", "messages": [
        {"role": "user", "content": "Frage"},
        {"role": "system", "content": "Umgebung: Linux"},
        {"role": "assistant", "content": "Antwort"},
        {"role": "system", "content": [{"type": "text", "text": "Erinnerung"}]},
        {"role": "user", "content": "Weiter"}]},
    "bild_platzhalter": {"tools": AN_TOOLS, "messages": [
        {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                   "data": "AAAA"}},
                                     {"type": "text", "text": "Was ist das?"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": TU, "name": "read", "input": {"path": "a.png"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": TU, "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "d"}}]}]}]},
    "felder_metadata_tier": {"metadata": {"user_id": "u-1"}, "service_tier": "auto",
                             "messages": [{"role": "user", "content": "Hallo"}]},
    "citations_am_text": {"messages": [
        {"role": "user", "content": "Frage"},
        {"role": "assistant", "content": [{"type": "text", "text": "Antwort mit Quelle",
                                           "citations": [{"type": "char_location", "cited_text": "x",
                                                          "document_index": 0, "start_char_index": 0,
                                                          "end_char_index": 1}]}]},
        {"role": "user", "content": "Danke"}]},
    "sampling_gueltig": {"temperature": 1.0, "top_p": 0.9, "top_k": 5, "stop_sequences": ["###"],
                         "messages": [{"role": "user", "content": "Hallo"}]},
    "tool_choice_any": {"tools": AN_TOOLS, "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
                        "messages": [{"role": "user", "content": "x"}]},
    "tool_choice_none": {"tools": AN_TOOLS, "tool_choice": {"type": "none"},
                         "messages": [{"role": "user", "content": "x"}]},
    "defer_loading_reference": {"tools": AN_TOOLS + [{**AN_TOOLS[1], "name": "spaet", "defer_loading": True}],
                                "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": TU, "name": "bash", "input": {"command": "s"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": TU, "content": [
            {"type": "tool_reference", "tool_name": "spaet"}]}]}]},
    "output_format": {"output_format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}},
        "messages": [{"role": "user", "content": "Titel"}]},
    "leere_assistant_vorgabe_mitte": {"messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "b"}]},
}
CORPUS_OA = {
    "oa_einfach": {"messages": [{"role": "user", "content": "Hallo"}],
                   "chat_template_kwargs": {"enable_thinking": False}},
    "oa_tools_temp_hoch": {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "x"}],
                           "tools": OA_TOOLS, "temperature": 1.5, "top_p": 1.0},
    "oa_prefill_assistant_zuletzt": {"messages": [{"role": "user", "content": "x"},
                                                  {"role": "assistant", "content": "y"}]},
}
STYLES = ("qwen35", "bonsai")


def _an(body: dict) -> dict:
    return {"model": "tobu", "max_tokens": 16, "temperature": 0, **body}


def corpus_hashes(tokfile: Path, answers: str) -> dict:
    """Prompt hash (server._h over the token ids, like tobu.prompt_hash) per request of the collection, via parse_* and
    Backend.prompt as in the server (without HTTP: the test engine cannot draw, only the prompt is counted)."""
    import api
    import server

    out = {}
    for style in STYLES:
        a = server.build_parser().parse_args(["--engine", "keyed:" + answers, "--tokenizer", str(tokfile),
                                              "--port", "0", "--no-warmup", "--max-len", "32768",
                                              "--chat", style])
        B = server.Backend(a)
        for name, body in CORPUS_AN.items():
            try:
                out[f"{style}/{name}"] = server._h(B.prompt(api.parse_anthropic(_an(body))))
            except api.ApiError as e:
                out[f"{style}/{name}"] = f"HTTP {e.status}: {e}"
        for name, body in CORPUS_OA.items():
            try:
                out[f"{style}/{name}"] = server._h(B.prompt(api.parse_openai({"model": "tobu", "max_tokens": 16,
                                                                               **body})))
            except api.ApiError as e:
                out[f"{style}/{name}"] = f"HTTP {e.status}: {e}"
    return out


def test_prompts(tokfile: Path, answers: str):
    if not GOLDEN.exists():
        check("P Golden-Datei fixtures/tob12_prompts.json vorhanden", False)
        return
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    have = corpus_hashes(tokfile, answers)
    for k, h in want.items():
        check(f"P gleicher Prompt wie vor TOB-12: {k}", have.get(k) == h, (have.get(k), h))
    check("P Sammlung vollständig geprüft", set(want) == set(have), set(want) ^ set(have))


# ------------------------------------------------------------------------------------------------- helpers
def raw(url: str, method: str = "POST", data: bytes | None = None, headers: dict | None = None):
    """-> (status, headers, body as dict or text)."""
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            s, h, b = resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as e:
        s, h, b = e.code, e.headers, e.read()
    try:
        return s, h, json.loads(b)
    except json.JSONDecodeError:
        return s, h, b.decode(errors="replace")


def an_err(s, j, status, kind="invalid_request_error") -> bool:
    return s == status and isinstance(j, dict) and j.get("type") == "error" and j["error"]["type"] == kind \
        and bool(j["error"].get("message")) and str(j.get("request_id", "")).startswith("req_")


# ------------------------------------------------------------------------------------------------- checks
def test_api(tokfile: Path, answers: str):
    import api

    httpd, base = t2.serve("keyed:" + answers, tokfile)
    try:
        M = base + "/v1/messages"
        # E: error form
        s, h, j = raw(M, data=b"{kaputt")
        check("E kaputtes JSON bei /v1/messages -> 400 invalid_request_error (Anthropic-Form)", an_err(s, j, 400), (s, j))
        s, h, j = raw(M + "/count_tokens", data=b"[1,")
        check("E kaputtes JSON bei count_tokens -> 400 (Anthropic-Form)", an_err(s, j, 400), (s, j))
        s, h, j = raw(base + "/v1/chat/completions", data=b"{kaputt")
        check("E kaputtes JSON bei /chat/completions bleibt OpenAI-Form",
              s == 400 and "type" not in j and j["error"]["type"] == "invalid_request_error", (s, j))
        for method in ("POST", "GET"):
            s, h, j = raw(M + "/batches", method, b"{}" if method == "POST" else None)
            check(f"E {method} /v1/messages/batches -> 404 not_found_error, Batches nicht unterstützt",
                  an_err(s, j, 404, "not_found_error") and "batch" in j["error"]["message"].lower(), (s, j))
            s, h, j = raw(M + "/gibtsnicht", method, b"{}" if method == "POST" else None)
            check(f"E {method} unbekannter Pfad unter /v1/messages -> 404 (Anthropic-Form)",
                  an_err(s, j, 404, "not_found_error"), (s, j))
        s, h, j = raw(M, "GET")
        check("E GET /v1/messages -> 404 (Anthropic-Form)", an_err(s, j, 404, "not_found_error"), (s, j))
        s, h, j = raw(base + "/v1/gibtsnicht", data=b"{}")
        check("E unbekannter Pfad außerhalb /v1/messages bleibt OpenAI-Form", s == 404 and "type" not in j, (s, j))
        b = api.AnthropicOut.error_body(api.ApiError("x", 502, "tool_call_error"))
        check("E Fehlertyp außerhalb der Anthropic-Liste -> api_error",
              b["error"]["type"] == "api_error" and api.OpenAIOut.error_body(
                  api.ApiError("x", 502, "tool_call_error"))["error"]["type"] == "tool_call_error", b)

        # R: request-id
        s, h, j = raw(M, data=json.dumps(_an({"messages": [{"role": "user", "content": "Hallo"}]})).encode())
        rid = h.get("request-id") or ""
        check("R request-id (req_…) bei erfolgreicher Antwort", s == 200 and rid.startswith("req_") and len(rid) > 10,
              dict(h))
        s, h2, j = raw(M, data=json.dumps({"messages": []}).encode())
        check("R request-id im Header und im Fehler-JSON gleich", an_err(s, j, 400)
              and h2.get("request-id") == j["request_id"] and h2.get("request-id") != rid, (dict(h2), j))
        s, h, j = raw(base + "/health", "GET")
        check("R request-id auch bei /health", (h.get("request-id") or "").startswith("req_"), dict(h))
        s, h, raw_s = raw(M, data=json.dumps(_an({"messages": [{"role": "user", "content": "Hallo"}],
                                                  "stream": True})).encode())
        check("R request-id beim Stream", s == 200 and (h.get("request-id") or "").startswith("req_"), dict(h))

        # F: fields
        for f, v in (("mcp_servers", [{"type": "url", "url": "http://127.0.0.1:1/x", "name": "m"}]),
                     ("container", "cont_1")):
            s, j = post(M, _an({f: v, "messages": [{"role": "user", "content": "x"}]}))
            check(f"F {f} -> 400 invalid_request_error", an_err(s, j, 400) and f in j["error"]["message"], (s, j))
        s, j = post(M, _an(CORPUS_AN["felder_metadata_tier"]))
        check("F metadata, service_tier: angenommen, in tobu.ignored",
              s == 200 and {"metadata", "service_tier"} <= set(j["tobu"].get("ignored", [])), j)
        s, j = post(M, _an(CORPUS_AN["system_liste_billing_cache"]))
        check("F cache_control: angenommen, in tobu.ignored", s == 200 and "cache_control" in j["tobu"]["ignored"], j)
        s, j = post(M, _an(CORPUS_AN["citations_am_text"]))
        check("F citations: angenommen, in tobu.ignored", s == 200 and "citations" in j["tobu"]["ignored"], j)
        s, j = post(M, _an({"messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "d"},
             "citations": {"enabled": True}}, {"type": "text", "text": "x"}]}]}))
        check("F citations an document: angenommen, vermerkt", s == 200 and "citations" in j["tobu"]["ignored"], j)
        s, j = post(M + "/count_tokens", {"model": "tobu", "mcp_servers": [{"name": "m"}],
                                          "messages": [{"role": "user", "content": "x"}]})
        check("F count_tokens: mcp_servers ebenfalls 400", an_err(s, j, 400), (s, j))

        # S: search_result
        sr = {"type": "search_result", "source": "https://example.org/a", "title": "Titel A",
              "content": [{"type": "text", "text": "Inhalt eins."}, {"type": "text", "text": "Inhalt zwei."}],
              "citations": {"enabled": True}}
        s, j = post(M, _an({"messages": [{"role": "user", "content": [sr, {"type": "text", "text": "Frage"}]}]}))
        s2, c = post(M + "/count_tokens", {"model": "tobu", "messages": [
            {"role": "user", "content": api.search_result_text(sr) + "Frage"}]})
        check("S search_result im user-Inhalt -> Text (gleicher Prompt wie der Text selbst)",
              s == 200 and s2 == 200 and j["usage"]["input_tokens"] + j["usage"]["cache_read_input_tokens"]
              == c["input_tokens"] and "Titel A" in api.search_result_text(sr), (s, j, c))
        s, j = post(M, _an({"tools": AN_TOOLS, "messages": [
            {"role": "user", "content": "Suche"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": TU, "name": "bash", "input": {"command": "s"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": TU, "content": [sr]}]}]}))
        check("S search_result im tool_result -> 200", s == 200, (s, j))
        s, j = post(M, _an({"messages": [{"role": "user", "content": [{"type": "container_upload", "file_id": "f"}]}]}))
        check("S container_upload bleibt 400", an_err(s, j, 400), (s, j))

        # V: parameter limits
        msg = [{"role": "user", "content": "Hallo"}]
        for name, extra in (("temperature 2.5", {"temperature": 2.5}), ("temperature -0.1", {"temperature": -0.1}),
                            ("temperature als Text", {"temperature": "0.5"}), ("top_p 1.5", {"top_p": 1.5}),
                            ("top_p -1", {"top_p": -1}), ("top_k -1", {"top_k": -1}), ("top_k 2.5", {"top_k": 2.5}),
                            ("max_tokens 0", {"max_tokens": 0}), ("max_tokens -5", {"max_tokens": -5}),
                            ("max_tokens 2.5", {"max_tokens": 2.5}), ("max_tokens als Text", {"max_tokens": "10"}),
                            ("max_tokens true", {"max_tokens": True})):
            s, j = post(M, {**_an({"messages": msg}), **extra})
            check(f"V {name} -> 400 invalid_request_error", an_err(s, j, 400), (s, j))
        for name, extra in (("top_p 1, top_k 0", {"top_p": 1, "top_k": 0}), ("top_p 0", {"top_p": 0}),
                            ("max_tokens 1", {"max_tokens": 1}), ("max_tokens 8.0", {"max_tokens": 8.0})):
            s, j = post(M, {**_an({"messages": msg}), **extra})
            check(f"V Grenzwert {name} angenommen (HTTP 200)", s == 200, (s, j))
        for t in (0, 0.0, 0.5, 1, 1.0):          # the test engine does not draw: limits checked directly at the parser
            r = api.parse_anthropic({"max_tokens": 4, "temperature": t, "messages": msg})
            check(f"V Grenzwert temperature {t!r} angenommen", (r.sampling is None) == (t == 0), r.sampling)
        s, j = post(M, _an({"messages": [{"role": "user", "content": "Hallo"}], "max_tokens": None}))
        check("V max_tokens null -> 400", an_err(s, j, 400), (s, j))
        s, j = post(base + "/v1/chat/completions", {"messages": msg, "max_tokens": 4, "temperature": 0,
                                                    "top_p": 1.5, "chat_template_kwargs": {"enable_thinking": False}})
        check("V /chat/completions unverändert (top_p 1.5 angenommen)", s == 200, (s, j))

        # X: Prefill
        base_msgs = [{"role": "user", "content": "Erzähl [[weiter]]"}]
        s0, h0 = post(M + "/count_tokens", {"model": "tobu", "messages": base_msgs})
        s1, j = post(M, _an({"messages": base_msgs + [{"role": "assistant", "content": "Es war einmal"}],
                             "max_tokens": 50}))
        s2, js = post(M, _an({"messages": base_msgs + [{"role": "assistant", "content": [
            {"type": "text", "text": "Es war "}, {"type": "text", "text": "einmal"}]}], "max_tokens": 50, "stream": True}),
            raw=True)
        check("X Prefill: 200, Antwort nur die Fortsetzung (ohne Prefill, führendes Leerzeichen bleibt)",
              s1 == 200 and j["content"] == [{"type": "text", "text": " und weiter geht es."}]
              and j["stop_reason"] == "end_turn", (s1, j))
        if s2 == 200:
            blocks = an_collect(js)[0]
            check("X Prefill im Stream gleich (Blöcke als Liste)", t2.norm(blocks) == t2.norm(j["content"]), blocks)
        n_pre = len("Es war einmal".encode())        # byte tokenizer without merges: one token per byte
        check("X Prefill: Prompt = Prompt ohne Prefill + Prefill-Text (Länge)",
              s0 == 200 and s1 == 200 and j["usage"]["input_tokens"] + j["usage"]["cache_read_input_tokens"]
              == h0["input_tokens"] + n_pre, (h0, j.get("usage")))
        s, c = post(M + "/count_tokens", {"model": "tobu", "messages": base_msgs + [
            {"role": "assistant", "content": "Es war einmal"}]})
        check("X count_tokens zählt den Prefill mit", s == 200 and c["input_tokens"] == h0["input_tokens"] + n_pre, c)
        s, j2 = post(M, _an({"messages": base_msgs + [{"role": "assistant", "content": ""}]}))
        s3, j3 = post(M, _an({"messages": base_msgs}))
        check("X leerer Prefill = kein Prefill (gleicher Prompt)",
              s == s3 == 200 and j2["tobu"]["prompt_hash"] == j3["tobu"]["prompt_hash"], (j2, j3))
        s, j = post(M, _an({"thinking": {"type": "enabled", "budget_tokens": 1024},
                            "messages": base_msgs + [{"role": "assistant", "content": "Es"}]}))
        check("X Prefill mit thinking -> 400", an_err(s, j, 400), (s, j))
        s, j = post(M, _an({"messages": base_msgs + [{"role": "assistant", "content": "Es war "}]}))
        check("X Prefill mit Leerzeichen am Ende -> 400 (wie Anthropic)", an_err(s, j, 400), (s, j))
        s, j = post(M, _an({"tools": AN_TOOLS, "messages": base_msgs + [{"role": "assistant", "content": [
            {"type": "tool_use", "id": TU, "name": "bash", "input": {"command": "ls"}}]}]}))
        check("X Prefill mit tool_use -> 400", an_err(s, j, 400), (s, j))
        s, j = post(M, _an({**CORPUS_AN["output_format"],
                            "messages": CORPUS_AN["output_format"]["messages"] + [{"role": "assistant", "content": "{"}]}))
        check("X Prefill mit output_format -> 400", an_err(s, j, 400), (s, j))
        s, j = post(base + "/v1/chat/completions", {"model": "tobu", "max_tokens": 16, "temperature": 0,
                                                    **CORPUS_OA["oa_prefill_assistant_zuletzt"]})
        check("X /chat/completions: assistant zuletzt unverändert (kein Prefill)", s == 200, (s, j))

        # T: stream form thinking
        s, js = post(M, _an({"thinking": {"type": "enabled", "budget_tokens": 1024},
                             "messages": [{"role": "user", "content": "Rechne [[denk]]"}], "stream": True}), raw=True)
        evs = t2.sse(js) if s == 200 else []
        seq = [(e, d.get("delta", {}).get("type") or d.get("content_block", {}).get("type")) for e, d in evs]
        i_stop = next((i for i, (e, _) in enumerate(seq) if e == "content_block_stop"), -1)
        check("T thinking-Block im Stream: signature_delta direkt vor content_block_stop",
              i_stop > 0 and seq[i_stop - 1] == ("content_block_delta", "signature_delta")
              and seq[1] == ("content_block_start", "thinking"), seq)
        blocks = an_collect(js)[0] if s == 200 else []
        check("T thinking-Block: signature leer (Stream und Nicht-Stream gleich)",
              bool(blocks) and blocks[0].get("signature") == "", blocks)
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        t2.make_tokenizer(tokfile)
        ans = Path(td) / "answers.json"
        ans.write_text(json.dumps(ANSWERS, ensure_ascii=False), encoding="utf-8")
        if "--golden" in sys.argv:
            h = corpus_hashes(tokfile, str(ans))
            bad = {k: v for k, v in h.items() if v.startswith("HTTP")}
            assert not bad, bad
            GOLDEN.write_text(json.dumps(h, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(f"{len(h)} Prompt-Hashes nach {GOLDEN.name} geschrieben "
                  f"(sha256 der Datei {hashlib.sha256(GOLDEN.read_bytes()).hexdigest()[:12]})")
            sys.exit(0)
        test_prompts(tokfile, str(ans))
        test_api(tokfile, str(ans))
    print("FEHLER:", ", ".join(t2.FAILS) if t2.FAILS else "keine")
    sys.exit(1 if t2.FAILS else 0)
