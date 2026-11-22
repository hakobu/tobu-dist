"""Edge cases of the tobu server with two interfaces, without GPU:  python test_api.py

1. Bonsai template in Python == template from the GGUF (jinja2), thinking on/off, effort, preserve_thinking
2. Splitter (stream splitting) == chat.parse, with random chunking; detok; stop strings
3. Anthropic and OpenAI request with the same history -> the same prompt; unsupported -> ApiError
4. End to end over HTTP (test engines): OpenAI/Anthropic × stream/non-stream -> the same tokens, same texts and
   tool calls, usage with cache fields, errors in the format of the respective interface
With --url http://127.0.0.1:PORT, part 4b (the same request over all four paths, tokens equal) runs against a
running server with a real engine, see check_server(). Tokenizer: TOBU_TEST_TOKENIZER (tokenizer.json), otherwise
the byte tokenizer from test_tob2 (part 1 needs the Bonsai template under templates/, otherwise skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

from hftok import HFTok  # noqa: E402

import api  # noqa: E402
import chat  # noqa: E402

TOKFILE = Path(os.environ.get("TOBU_TEST_TOKENIZER") or tempfile.mkdtemp(prefix="tobu-tok-")) / (
    "" if os.environ.get("TOBU_TEST_TOKENIZER") else "tokenizer.json")
FAILS: list[str] = []
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Datei lesen", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Befehl", "parameters": {
             "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
AN_TOOLS = [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in TOOLS]


def check(name, cond, info=""):
    print(("ok   " if cond else "FEHLER ") + name + (f"  [{info}]" if info and not cond else ""))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------ 1 template
def tc(name, args, id_="c1"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


CONVS = [
    [{"role": "system", "content": "  Du bist hilfreich.  "}, {"role": "user", "content": "Hallo"}],
    [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix bug"},
     {"role": "assistant", "content": "Ich lese.", "reasoning_content": " denk \n",
      "tool_calls": [tc("read", {"path": "a.ts", "limit": 20, "flag": True, "x": None, "l": [1, 2]})]},
     {"role": "tool", "tool_call_id": "c1", "content": "inhalt\nzeile"}, {"role": "tool", "tool_call_id": "c1", "content": "2"},
     {"role": "assistant", "content": "", "tool_calls": [tc("bash", {"command": "ls"}), tc("read", {"path": "b"})]},
     {"role": "tool", "tool_call_id": "c1", "content": "ok"}],
    [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
     {"role": "user", "content": [{"type": "text", "text": "q2"}]}],
    [{"role": "developer", "content": "dev"}, {"role": "user", "content": "x"}, {"role": "assistant", "content": "</think>y"},
     {"role": "user", "content": "z"}],
    [{"role": "system", "content": "   "}, {"role": "user", "content": "x"}],
]


def test_template():
    if not chat.TEMPLATE_BONSAI.exists():
        print("übersprungen: Bonsai-Vorlage (templates/) fehlt")
        return
    n = bad = 0
    for msgs in CONVS:
        for tools in (None, TOOLS):
            for th in (True, False):
                for eff in (None, "low", "medium", "xhigh"):
                    for pres in (None, False):
                        kw = {}
                        if eff:
                            kw["reasoning_effort"] = eff
                        if pres is not None:
                            kw["preserve_thinking"] = pres
                        a = chat.render_jinja(msgs, tools, enable_thinking=th, template=chat.TEMPLATE_BONSAI, **kw)
                        b = chat.render_bonsai_py(msgs, tools, enable_thinking=th, **kw)
                        n += 1
                        bad += a != b
    check(f"Bonsai-Vorlage Python == GGUF-Vorlage ({n} Fälle)", bad == 0, f"{bad} abweichend")
    try:
        chat.render_bonsai_py(CONVS[0], None, reasoning_effort="max")
        check("Bonsai-Vorlage: unbekannter Aufwand -> Fehler", False)
    except ValueError:
        check("Bonsai-Vorlage: unbekannter Aufwand -> Fehler", True)


# ------------------------------------------------------------------------------------------------ 2 splitting
OUTS = [
    "\nIch denke nach.\n\n</think>\n\nDie Antwort ist 42.\n",
    "kurz</think>Text ohne Umbruch",
    "\n\n</think>\n\n",
    "<think>\nmit Tag</think>\nA\n\nB\n\n",
    "denk\n</think>\n\nIch lese die Datei.\n\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/a.ts\n</parameter>\n"
    "<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\n"
    "ls -la\n</parameter>\n</function>\n</tool_call>",
    "x</think><tool_call>\n<function=read>\n<parameter=path>\na\nb\n</parameter>\n</function>\n</tool_call>",
    "endet im Denken ohne Abschluss </thi",
    "Umlaute äöü und ein Emoji 🙂 </think>\nÜberschrift\n\n- Punkt <tool_c",
]


def agg(events):
    r, c, calls = "", "", []
    for k, x in events:
        if k == "think":
            r += x
        elif k == "text":
            c += x
        elif k == "tool":
            calls.append((x["name"], x["input"]))
    return r, c, calls


def test_splitter():
    rnd = random.Random(7)
    bad = 0
    for text in OUTS:
        for thinking in (True, False):
            # thinking off: everything is answer (the generation prompt already closes <think>) – chat.parse would still
            # read a "</think>" in the output as end of thinking; streamed that is not possible, so deliberately different.
            want = chat.parse(text if thinking else text.replace("</think>", "\x00"), TOOLS, thinking)
            if not thinking:
                want = {k: (v.replace("\x00", "</think>") if isinstance(v, str) else v) for k, v in want.items()}
            wcalls = [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in want["tool_calls"]]
            for trial in range(40):
                sp = api.Splitter(TOOLS, thinking)
                ev, i = [], 0
                while i < len(text):
                    j = i + rnd.randint(1, 9) if trial else i + 1
                    ev += sp.feed(text[i:j])
                    i = j
                ev += sp.finish()
                got = agg(ev)
                if got != (want["reasoning_content"], want["content"], wcalls):
                    bad += 1
                    if bad <= 2:
                        print("   ", repr(text[:40]), thinking, got, (want["reasoning_content"], want["content"], wcalls))
    check(f"Splitter == chat.parse ({len(OUTS)} Ausgaben × Denken an/aus × 40 Stückelungen)", bad == 0, f"{bad} abweichend")
    sp = api.Splitter(None, False)
    ev = sp.feed("a <tool_call> b") + sp.finish()
    check("Splitter ohne Tools: <tool_call> bleibt Text", agg(ev) == ("", "a <tool_call> b", []))
    sp = api.Splitter(TOOLS, False)
    sp.feed("<tool_call>\n<function=read>\n<parameter=path>\nx")
    sp.finish()
    check("Splitter: unvollständiger Tool-Aufruf wird erkannt", sp.malformed is not None)
    sp = api.Splitter(TOOLS, False)
    sp.feed("<tool_call>\n<function=rm>\n</function>\n</tool_call>")
    sp.finish()
    check("Splitter: unbekanntes Tool wird erkannt", "unknown tool" in (sp.malformed or ""))


def test_detok_stop(tok):
    rnd = random.Random(3)
    text = "Grüße aus Köln – 🙂🙃 漢字 und <|im_end|>? " * 4
    ids = tok.encode(text)
    bad = 0
    for _ in range(30):
        d, out, i = api.Detok(tok, set()), "", 0
        while i < len(ids):
            j = i + rnd.randint(1, 4)
            out += d.feed(ids[i:j])
            i = j
        out += d.flush()
        bad += out != tok.decode(ids)
    check("Detok: Stücke zusammen == decode(alle), auch mitten in UTF-8-Zeichen", bad == 0, f"{bad}/30")
    eot = tok.t.token_to_id(chat.EOT)
    d = api.Detok(tok, {eot})
    check("Detok: Stopp-Token erscheint nicht im Text", d.feed(tok.encode("ok") + [eot]) + d.flush() == "ok")
    for stops, chunks, want in ((["STOP"], ["ab", "cST", "OPxy"], "abc"), (["\n\n"], ["a\n", "\nb"], "a"),
                                (["xyz"], ["ab", "xy"], "abxy"), (["b", "ab"], ["xab"], "x")):
        sf = api.StopFilter(stops)
        out = "".join(sf.feed(c) for c in chunks) + sf.flush()
        check(f"StopFilter {stops} {chunks} -> {want!r}", out == want, repr(out))


# ------------------------------------------------------------------------------------------------ 3 mapping
def test_mapping():
    oa = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "lies a.ts"},
                       {"role": "assistant", "content": "Mache ich.", "reasoning_content": "erst lesen",
                        "tool_calls": [tc("read", {"path": "a.ts"}, "toolu_1")]},
                       {"role": "tool", "tool_call_id": "toolu_1", "content": "export const x = 1;"},
                       {"role": "user", "content": "und jetzt?"}],
          "tools": TOOLS, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}
    an = {"system": [{"type": "text", "text": "sys"}], "max_tokens": 64, "tools": AN_TOOLS,
          "messages": [{"role": "user", "content": "lies a.ts"},
                       {"role": "assistant", "content": [{"type": "thinking", "thinking": "erst lesen", "signature": "x"},
                                                         {"type": "text", "text": "Mache ich."},
                                                         {"type": "tool_use", "id": "toolu_1", "name": "read",
                                                          "input": {"path": "a.ts"}}]},
                       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                                     "content": [{"type": "text", "text": "export const x = 1;"}]},
                                                    {"type": "text", "text": "und jetzt?"}]}]}
    ro, ra = api.parse_openai(oa), api.parse_anthropic(an)
    po = chat.render_bonsai_py(ro.messages, ro.tools, enable_thinking=ro.thinking)
    pa = chat.render_bonsai_py(ra.messages, ra.tools, enable_thinking=ra.thinking)
    check("Abbildung: Anthropic- und OpenAI-Verlauf -> derselbe Prompt", po == pa)
    check("Abbildung: Anthropic ohne thinking -> Denken aus, OpenAI-Standard an",
          not ra.thinking and api.parse_openai({"messages": oa["messages"]}).thinking)
    ra2 = api.parse_anthropic({**an, "thinking": {"type": "enabled", "budget_tokens": 2000}, "top_p": 0.5})
    check("Abbildung: thinking.enabled -> an, Standard-Sampling Denken an + eigener top_p",
          ra2.thinking and ra2.sampling["temperature"] == 1.0 and ra2.sampling["top_p"] == 0.5
          and "thinking.budget_tokens" not in ra2.ignored and ra2.thinking_budget == 2000, str(ra2.sampling))
    check("Abbildung: temperature 0 -> greedy", api.parse_openai({**oa, "temperature": 0}).sampling is None
          and api.parse_anthropic({**an, "temperature": 0}).sampling is None)
    bad = [("OpenAI Bild", api.parse_openai, {"messages": [{"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": "data:"}}]}]}),
           ("OpenAI tool_choice required ohne Tools", api.parse_openai, {"messages": oa["messages"],
                                                                         "tool_choice": "required"}),
           ("OpenAI JSON-Schema, das auch der Fork ablehnt (pattern mit offener Klammer)", api.parse_openai, {
               "messages": oa["messages"], "response_format": {"type": "json_schema", "json_schema": {
                   "schema": {"type": "object", "properties": {"a": {"type": "string", "pattern": "^(x$"}}}}}}),
           ("OpenAI freie Grammatik mit Tools", api.parse_openai, {"messages": oa["messages"], "tools": TOOLS,
                                                                   "grammar": 'root ::= "x"'}),
           ("OpenAI n=2", api.parse_openai, {"messages": oa["messages"], "n": 2}),
           ("Anthropic ohne max_tokens", api.parse_anthropic, {"messages": an["messages"]}),
           ("Anthropic unbekannter Block", api.parse_anthropic, {"max_tokens": 5, "messages": [{"role": "user",
               "content": [{"type": "container_upload", "file_id": "f"}]}]}),   # image: placeholder since L35,
                                                                                 # search_result: text since TOB-12
           ("Anthropic tool_choice any ohne Tools", api.parse_anthropic, {**an, "tools": None,
                                                                          "tool_choice": {"type": "any"}}),
           ("Anthropic Server-Tool", api.parse_anthropic, {**an, "tools": [{"type": "web_search_20250305",
                                                                           "name": "web_search"}]}),
           ("Anthropic Antwort-Vorgabe mit Denken", api.parse_anthropic, {"max_tokens": 5,  # without thinking: TOB-12
               "thinking": {"type": "enabled"}, "messages": [
               {"role": "user", "content": "x"}, {"role": "assistant", "content": "Vorgabe"}]})]
    for name, fn, body in bad:
        try:
            fn(body)
            check(f"nicht unterstützt -> Fehler: {name}", False)
        except api.ApiError as e:
            check(f"nicht unterstützt -> Fehler: {name}", e.status == 400, str(e))


# ------------------------------------------------------------------------------------------------ Sampling
def test_sampler():
    import numpy as np

    import sampling

    lg = np.log(np.array([0.5, 0.3, 0.2] + [1e-12] * 7, dtype=np.float32))
    base = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
    s = sampling.Sampler(dict(base, seed=1), [])
    c = np.bincount([s(lg) for _ in range(20000)], minlength=10) / 20000
    check("Sampler: Häufigkeiten = Softmax (0,5/0,3/0,2 ± 0,015)", np.abs(c[:3] - [0.5, 0.3, 0.2]).max() < 0.015, str(c[:3]))
    s = sampling.Sampler(dict(base, top_p=0.75, seed=2), [])
    check("Sampler: top_p 0,75 -> nur die ersten zwei", set(s(lg) for _ in range(2000)) == {0, 1})
    s = sampling.Sampler(dict(base, min_p=0.5, seed=3), [])
    check("Sampler: min_p 0,5 -> nur p >= 0,25", set(s(lg) for _ in range(2000)) == {0, 1})
    s = sampling.Sampler(dict(base, top_k=1, seed=4), [])
    check("Sampler: top_k 1 -> Argmax", set(s(lg) for _ in range(500)) == {0})
    a = [sampling.Sampler(dict(base, seed=9), [])(lg) for _ in range(1)] + [sampling.Sampler(dict(base, seed=9), [])(lg)]
    check("Sampler: gleicher Seed -> gleiches Token", a[0] == a[1])
    s = sampling.Sampler(dict(base, presence_penalty=50.0, seed=5), [])
    seq = [s(lg) for _ in range(3)]
    check("Sampler: presence_penalty 50 -> keine Wiederholung unter den ersten drei", len(set(seq)) == 3, str(seq))
    check("Standardwerte je Modus (Modellkarte)", sampling.resolve({}, True)["temperature"] == 1.0 and
          sampling.resolve({}, False)["presence_penalty"] == 1.5 and sampling.resolve({"temperature": 0}, True) is None)


# ------------------------------------------------------------------------------------------------ 4 HTTP
def post(url, body, headers=None, raw=False):
    data = json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=3600) as resp:
            out = resp.read()
            return resp.status, (out if raw else json.loads(out))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def sse(raw: bytes):
    """-> list of (event, data) from an SSE stream (comments skipped)."""
    out, ev = [], None
    for line in raw.decode().split("\n"):
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            d = line[6:]
            out.append((ev, d if d == "[DONE]" else json.loads(d)))
            ev = None
    return out


def oa_collect(raw):
    r, c, calls, last = "", "", {}, None
    for _, d in sse(raw):
        if d == "[DONE]" or not d.get("choices"):
            continue
        dl = d["choices"][0]["delta"]
        r += dl.get("reasoning_content") or ""
        c += dl.get("content") or ""
        for t in dl.get("tool_calls") or []:   # like OpenAI clients: name from the first chunk, join the arguments
            f = t.get("function") or {}
            name, args = calls.get(t["index"], (None, ""))
            calls[t["index"]] = (f.get("name") or name, args + (f.get("arguments") or ""))
        if d["choices"][0]["finish_reason"]:
            last = d
    return r, c, [(n, json.loads(a)) for n, a in (calls[i] for i in sorted(calls))], last


def an_collect(raw):
    blocks, msg, delta = {}, None, None
    for ev, d in sse(raw):
        if ev == "message_start":
            msg = d["message"]
        elif ev == "content_block_start":
            blocks[d["index"]] = dict(d["content_block"], _json="")
        elif ev == "content_block_delta":
            b, dl = blocks[d["index"]], d["delta"]
            if dl["type"] == "text_delta":
                b["text"] += dl["text"]
            elif dl["type"] == "thinking_delta":
                b["thinking"] += dl["thinking"]
            elif dl["type"] == "input_json_delta":
                b["_json"] += dl["partial_json"]
        elif ev == "message_delta":
            delta = d
    out = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "tool_use":
            b["input"] = json.loads(b["_json"])
        b.pop("_json")
        out.append(b)
    return out, msg, delta


def oa_parts(msg):
    m = msg["choices"][0]["message"]
    return (m.get("reasoning_content") or "", m.get("content") or "",
            [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in m.get("tool_calls") or []])


def an_parts(blocks):
    r = "".join(b["thinking"] for b in blocks if b["type"] == "thinking")
    c = "".join(b["text"] for b in blocks if b["type"] == "text")
    return r, c, [(b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"]


def four_ways(base, oa_body, an_body, label, headers=None):
    """The same request over OpenAI/Anthropic × stream/non-stream -> (parts, hashes, answers)."""
    s1, j1 = post(base + "/v1/chat/completions", oa_body, headers)
    s2, r2 = post(base + "/v1/chat/completions", {**oa_body, "stream": True}, headers, raw=True)
    s3, j3 = post(base + "/v1/messages", an_body, headers)
    s4, r4 = post(base + "/v1/messages", {**an_body, "stream": True}, headers, raw=True)
    ok = (s1, s2, s3, s4) == (200, 200, 200, 200)
    check(f"{label}: alle vier Wege HTTP 200", ok, str((s1, s2, s3, s4, j1 if s1 != 200 else "", j3 if s3 != 200 else "")))
    if not ok:
        return None
    r2c, c2c, k2, last2 = oa_collect(r2)
    blocks4, msg4, delta4 = an_collect(r4)
    parts = [oa_parts(j1), (r2c, c2c, k2), an_parts(j3["content"]), an_parts(blocks4)]
    hashes = [j1["tobu"]["out_hash"], last2["tobu"]["out_hash"], j3["tobu"]["out_hash"], delta4["tobu"]["out_hash"]]
    check(f"{label}: dieselben Tokens über alle vier Wege", len(set(hashes)) == 1, str(hashes))
    check(f"{label}: Stream == Nicht-Stream (OpenAI)", parts[0] == parts[1], f"{parts[0]} | {parts[1]}")
    check(f"{label}: Stream == Nicht-Stream (Anthropic)", parts[2] == parts[3], f"{parts[2]} | {parts[3]}")
    check(f"{label}: OpenAI == Anthropic (Denken, Text, Tool-Calls)", parts[0] == parts[2], f"{parts[0]} | {parts[2]}")
    return {"parts": parts, "j1": j1, "last2": last2, "j3": j3, "msg4": msg4, "delta4": delta4, "hash": hashes[0]}


def serve(engine: str, extra=()):
    import server

    a = server.build_parser().parse_args(["--engine", engine, "--tokenizer", str(TOKFILE), "--arm", "C", "--kmax", "8",
                                          "--max-len", "32768", "--port", "0", "--no-warmup"])
    for k, v in extra:
        setattr(a, k, v)
    B = server.Backend(a)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(B))
    httpd.backend = B
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_http():
    answers = {
        "default": "Hallo zurück.<|im_end|>",
        "denk": "\nIch überlege kurz.\n</think>\n\nDas Ergebnis ist **42** – mit Umlauten: äöü 🙂.\n<|im_end|>",
        "tool": "Ich lese die Datei.\n\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/a.ts\n"
                "</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "kaputt": "<tool_call>\n<function=read>\n<parameter=path>\nx<|im_end|>",
        "lang": "eins zwei drei vier fünf sechs sieben acht neun zehn " * 3 + "<|im_end|>",
        "stopp": "Anfang ### Ende, das nicht kommen darf.<|im_end|>",
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False)
    httpd, base = serve("keyed:" + f.name)
    try:
        def bodies(text, think, extra_oa=None, extra_an=None, tools=True):
            oa = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": text}], "max_tokens": 200,
                  "chat_template_kwargs": {"enable_thinking": think}, "temperature": 0, **(extra_oa or {})}
            an = {"system": "sys", "messages": [{"role": "user", "content": text}], "max_tokens": 200, "temperature": 0,
                  **(extra_an or {})}
            if think:
                an["thinking"] = {"type": "enabled", "budget_tokens": 1024}
            if tools:
                oa["tools"], an["tools"] = TOOLS, AN_TOOLS
            return oa, an

        r = four_ways(base, *bodies("Rechne [[denk]]", True), "Denken an")
        if r:
            check("Denken an: Denktext und Antwort getrennt", r["parts"][0][0] == "Ich überlege kurz." and
                  r["parts"][0][1].startswith("Das Ergebnis ist **42**"), str(r["parts"][0]))
            th = [b for b in r["j3"]["content"] if b["type"] == "thinking"]
            check("Anthropic: thinking-Block vor dem Text", r["j3"]["content"][0]["type"] == "thinking" and len(th) == 1)
            check("Anthropic: stop_reason end_turn, OpenAI finish stop",
                  r["j3"]["stop_reason"] == "end_turn" and r["j1"]["choices"][0]["finish_reason"] == "stop")
        r = four_ways(base, *bodies("Lies [[tool]]", False), "Tool-Call")
        if r:
            check("Tool-Call: Text + ein Aufruf mit typisierten Argumenten",
                  r["parts"][0][1:] == ("Ich lese die Datei.", [("read", {"path": "src/a.ts", "limit": 20})]),
                  str(r["parts"][0]))
            check("Tool-Call: finish_reason tool_calls / stop_reason tool_use",
                  r["j1"]["choices"][0]["finish_reason"] == "tool_calls" and r["j3"]["stop_reason"] == "tool_use")
            tu = [b for b in r["j3"]["content"] if b["type"] == "tool_use"][0]
            check("Tool-Call: Anthropic tool_use mit id toolu_…", tu["id"].startswith("toolu_") and tu["input"]["limit"] == 20)
        r = four_ways(base, *bodies("Zähle [[lang]]", False, {"max_tokens": 9}, {"max_tokens": 9}), "Längenlimit")
        if r:
            check("Längenlimit: 9 Tokens, finish length / max_tokens",
                  r["j1"]["usage"]["completion_tokens"] == 9 and r["j1"]["choices"][0]["finish_reason"] == "length"
                  and r["j3"]["stop_reason"] == "max_tokens" and r["j3"]["usage"]["output_tokens"] == 9)
        r = four_ways(base, *bodies("Stopp [[stopp]]", False, {"stop": ["###"]}, {"stop_sequences": ["###"]}), "Stopp-Text")
        if r:
            check("Stopp-Text: Text endet davor, stop_sequence gemeldet", r["parts"][0][1] == "Anfang " and
                  r["j3"]["stop_reason"] == "stop_sequence" and r["j3"]["stop_sequence"] == "###"
                  and r["j1"]["choices"][0]["finish_reason"] == "stop", str(r["parts"][0]))
        r = four_ways(base, *bodies("Hallo", False, tools=False), "ohne Tools")
        if r:
            u1, u3 = r["j1"]["usage"], r["j3"]["usage"]
            check("Usage: OpenAI prompt_tokens_details.cached_tokens, Anthropic cache_read/creation, input = Rest",
                  "cached_tokens" in u1["prompt_tokens_details"] and {"cache_read_input_tokens",
                                                                      "cache_creation_input_tokens"} <= set(u3)
                  and u3["input_tokens"] + u3["cache_read_input_tokens"] == u1["prompt_tokens"])
            check("Anthropic-Stream: message_start mit Usage, message_delta mit output_tokens",
                  r["msg4"]["usage"]["input_tokens"] > 0 and r["delta4"]["usage"]["output_tokens"] ==
                  u1["completion_tokens"])
        # errors in the respective format
        oa, an = bodies("Lies [[kaputt]]", False)
        s, j = post(base + "/v1/chat/completions", oa)
        check("Kaputter Tool-Aufruf -> Fehler (OpenAI-Form, 502)", s == 502 and j["error"]["type"] == "tool_call_error",
              str((s, j)))
        s, j = post(base + "/v1/messages", an)
        check("Kaputter Tool-Aufruf -> Fehler (Anthropic-Form, 502)", s == 502 and j["type"] == "error", str((s, j)))
        s, raw = post(base + "/v1/messages", {**an, "stream": True}, raw=True)
        evs = sse(raw)
        check("Kaputter Tool-Aufruf im Stream -> error-Ereignis am Ende", evs and evs[-1][0] == "error", str(evs[-1:]))
        s, j = post(base + "/v1/messages", {"messages": [{"role": "user", "content": "x"}]})
        check("Anthropic ohne max_tokens -> 400 invalid_request_error",
              s == 400 and j["error"]["type"] == "invalid_request_error")
        s, j = post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "x"}}]}]})
        check("OpenAI Bild -> 400 im OpenAI-Format", s == 400 and "error" in j and "message" in j["error"])
        s, j = post(base + "/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "Hallo"}]})
        check("count_tokens", s == 200 and j["input_tokens"] > 3, str(j))
        with urllib.request.urlopen(base + "/v1/models") as resp:
            mo = json.loads(resp.read())
        r = urllib.request.Request(base + "/v1/models", headers={"anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(r) as resp:
            ma = json.loads(resp.read())
        check("/v1/models in beiden Formaten", mo["object"] == "list" and ma["data"][0]["type"] == "model")
    finally:
        httpd.shutdown()
    # toy engine: real generation (context copy), prompt-dependent
    httpd, base = serve("toy")
    try:
        txt = ("Die Katze sitzt auf der Matte. Die Katze sitzt auf dem Dach. Die Katze sitzt")
        oa = {"messages": [{"role": "user", "content": txt}], "max_tokens": 40, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}}
        an = {"messages": [{"role": "user", "content": txt}], "max_tokens": 40, "temperature": 0}
        four_ways(base, oa, an, "Spielzeug-Engine, Arm C")
        four_ways(base, oa, an, "Spielzeug-Engine, Arm A", {"X-Tobu-Arm": "A"})
        # sampling: same seed -> same tokens over all four paths, speculation off
        so = {**oa, "temperature": 0.7, "seed": 11}
        sa = {**an, "temperature": 0.7, "seed": 11}
        r = four_ways(base, so, sa, "Sampling (Seed 11)")
        if r:
            check("Sampling: Spekulation aus, Parameter gemeldet", r["j1"]["tobu"]["spec"] == "off (sampling)" and
                  r["j1"]["tobu"]["sampling"]["temperature"] == 0.7 and r["j1"]["timings"]["tokens_per_round"] <= 1.0,
                  str(r["j1"]["tobu"]))
            s2, j2 = post(base + "/v1/chat/completions", {**so, "seed": 12})
            check("Sampling: anderer Seed -> andere Tokens", j2["tobu"]["out_hash"] != r["hash"])
        # --greedy-harness: a harness without temperature runs greedy; other harnesses and X-Tobu-Sampling: model not
        nt = {k: v for k, v in oa.items() if k != "temperature"}
        _, jref = post(base + "/v1/chat/completions", oa)
        httpd.backend.greedy_harness = {"claude-cli"}
        try:
            _, jg = post(base + "/v1/chat/completions", nt, {"User-Agent": "claude-cli/2.1 (external, cli)"})
            _, jm = post(base + "/v1/chat/completions", nt, {"User-Agent": "claude-cli/2.1 (external, cli)",
                                                              "X-Tobu-Sampling": "model"})
            _, jo = post(base + "/v1/chat/completions", nt, {"User-Agent": "opencode/1.0"})
            _, jt = post(base + "/v1/chat/completions", {**nt, "temperature": 0.7, "seed": 11},
                         {"User-Agent": "claude-cli/2.1 (external, cli)"})
        finally:
            httpd.backend.greedy_harness = set()
        check("greedy-harness: Harness ohne temperature greedy, gleiche Tokens wie temperature 0",
              jg["tobu"]["sampling"] == "greedy" and jg["tobu"]["out_hash"] == jref["tobu"]["out_hash"], str(jg["tobu"]))
        check("greedy-harness: X-Tobu-Sampling: model hebt auf", isinstance(jm["tobu"]["sampling"], dict), str(jm["tobu"]))
        check("greedy-harness: anderes Harness unverändert", isinstance(jo["tobu"]["sampling"], dict), str(jo["tobu"]))
        check("greedy-harness: ausdrückliche temperature gilt", isinstance(jt["tobu"]["sampling"], dict) and
              jt["tobu"]["sampling"]["temperature"] == 0.7, str(jt["tobu"]))
        s3, j3 = post(base + "/v1/chat/completions", {k: v for k, v in oa.items() if k != "temperature"})
        check("ohne temperature: Standard der Modellkarte (Denken aus: 0,7 / top_p 0,8 / presence 1,5)",
              j3["tobu"]["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                         "presence_penalty": 1.5, "repetition_penalty": 1.0}, str(j3["tobu"]))
    finally:
        httpd.shutdown()


# ------------------------------------------------------------------------------------------------ 4b real server
RULES = "\n".join(f"Regel {i}: Wenn im Text das Stichwort Nummer {i} vorkommt, notiere es mit kurzer Begründung "
                   f"und ordne es einer der Kategorien A, B oder C zu (Beispiel {i * 7 % 13})." for i in range(1, 46))
LONG_SYS = "Du bist ein sorgfältiger, knapper Assistent und antwortest auf Deutsch.\n\n" + RULES


def timed_stream(url, body, headers, parse):
    """Stream with timing at the client: TTFT until the first content, total time."""
    t0 = time.perf_counter()
    r = urllib.request.Request(url, data=json.dumps({**body, "stream": True}).encode(),
                               headers={"Content-Type": "application/json", **(headers or {})})
    chunks, t_first = [], None
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            if t_first is None and (b'"content"' in line and b'"content": ""' not in line or b"_delta" in line):
                t_first = time.perf_counter() - t0
            chunks.append(line)
    return b"".join(chunks), t_first, time.perf_counter() - t0


def check_server(base: str, out: str | None):
    """Against a running server with a real engine . Reference per prompt: arm A + X-Tobu-Cache: off
    (cold start, pure greedy decode). Everything else (cache, arm C, both interfaces, stream) must deliver the same
    tokens."""
    res = {"cases": {}}
    REF = {"X-Tobu-Arm": "A", "X-Tobu-Cache": "off"}
    oa_url, an_url = base + "/v1/chat/completions", base + "/v1/messages"

    def oa(msgs, mt=64, think=False, tools=None, **kw):
        b = {"messages": msgs, "max_tokens": mt, "chat_template_kwargs": {"enable_thinking": think}, "temperature": 0, **kw}
        if tools:
            b["tools"] = tools
        return b

    def an_of(msgs, mt=64, think=False, tools=None):
        sys_ = [m["content"] for m in msgs if m["role"] == "system"]
        b = {"messages": [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"],
             "max_tokens": mt, "temperature": 0}
        if sys_:
            b["system"] = sys_[0]
        if think:
            b["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        if tools:
            b["tools"] = AN_TOOLS
        return b

    def ref(body, label):
        t0 = time.perf_counter()
        st, j = post(oa_url, body, REF)
        check(f"{label}: Referenz (Arm A, Kaltstart) HTTP 200", st == 200, str(j)[:300])
        res["cases"].setdefault(label, {})["ref"] = {"hash": j["tobu"]["out_hash"], "timings": j["timings"],
                                                     "wall_s": round(time.perf_counter() - t0, 2),
                                                     "parts": oa_parts(j)}
        return j

    q1 = [{"role": "system", "content": LONG_SYS}, {"role": "user", "content": "Welche Regel nennt Beispiel 5? Kurz."}]
    q2 = [{"role": "system", "content": LONG_SYS}, {"role": "user", "content": "Fasse Regel 3 in einem Satz zusammen."}]
    r1, r2 = ref(oa(q1), "Q1"), ref(oa(q2), "Q2")
    # 1) cache + arm C: Q1 cold (after Q2 in the state: shared system prompt), then Q2 over all four paths
    st, j = post(oa_url, oa(q1))
    check("Q1 mit Cache und Arm C == Referenz", j["tobu"]["out_hash"] == r1["tobu"]["out_hash"])
    check("Q1: Cache-Treffer im gemeinsamen Systemprompt (cached_tokens >= 512)",
          j["usage"]["prompt_tokens_details"]["cached_tokens"] >= 512, str(j["usage"]))
    res["cases"]["Q1"]["cache"] = {"hash": j["tobu"]["out_hash"], "timings": j["timings"], "usage": j["usage"]}
    r = four_ways(base, oa(q2), an_of(q2), "Q2 (Cache, Arm C)")
    if r:
        check("Q2 über alle vier Wege == Referenz", r["hash"] == r2["tobu"]["out_hash"])
        check("Q2: Anthropic cache_read_input_tokens >= 512", r["j3"]["usage"]["cache_read_input_tokens"] >= 512,
              str(r["j3"]["usage"]))
        res["cases"]["Q2"]["cache"] = {"hash": r["hash"], "timings": r["j1"]["timings"], "usage_oa": r["j1"]["usage"],
                                       "usage_an": r["j3"]["usage"]}
    # 2) multiple turns: append answer, follow-up -> cache up to the last grid boundary, tokens == cold start
    a1 = r1["choices"][0]["message"]["content"] or ""
    q3 = q1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Und warum? Ein Satz."}]
    r3 = ref(oa(q3), "Q3 Folgeturn")
    post(oa_url, oa(q1))                                 # state as in the conversation: Q1 + answer
    st, j = post(oa_url, oa(q3))
    check("Folgeturn mit Cache == Kaltstart", j["tobu"]["out_hash"] == r3["tobu"]["out_hash"])
    res["cases"]["Q3 Folgeturn"]["cache"] = {"hash": j["tobu"]["out_hash"], "timings": j["timings"], "usage": j["usage"]}
    # 3) thinking on: OpenAI (enable_thinking) and Anthropic (thinking.enabled) -> the same template, the same tokens
    qd = [{"role": "user", "content": "Was ist 17*23?"}]
    rd = ref(oa(qd, mt=160, think=True), "Denken an")
    r = four_ways(base, oa(qd, mt=160, think=True), an_of(qd, mt=160, think=True), "Denken an")
    if r:
        check("Denken an: == Referenz, Denktext vorhanden", r["hash"] == rd["tobu"]["out_hash"] and bool(r["parts"][0][0]),
              str(r["parts"][0])[:200])
        res["cases"]["Denken an"]["parts"] = r["parts"][0]
    # 4) tool call
    qt = [{"role": "system", "content": "Du hast Werkzeuge. Nutze sie, statt zu raten."},
          {"role": "user", "content": "Zeig mir den Inhalt der Datei src/index.ts."}]
    rt = ref(oa(qt, mt=128, tools=TOOLS), "Tool-Call")
    r = four_ways(base, oa(qt, mt=128, tools=TOOLS), an_of(qt, mt=128, tools=True), "Tool-Call")
    if r:
        check("Tool-Call: == Referenz, Aufruf read erkannt", r["hash"] == rt["tobu"]["out_hash"] and
              any(c[0] == "read" for c in r["parts"][0][2]), str(r["parts"][0])[:300])
        res["cases"]["Tool-Call"]["parts"] = r["parts"][0]
        res["cases"]["Tool-Call"]["finish"] = (r["j1"]["choices"][0]["finish_reason"], r["j3"]["stop_reason"])
    # 5) time at the client vs. server metrics (stream, cache on)
    for label, url, body in (("OpenAI-Stream Q2", oa_url, oa(q2, mt=128)), ("Anthropic-Stream Q2", an_url, an_of(q2, mt=128))):
        raw, ttft, wall = timed_stream(url, body, None, None)
        evs = sse(raw)
        tm = next((d.get("timings") or d.get("tobu") for _, d in reversed(evs) if isinstance(d, dict)
                   and (d.get("timings") or d.get("tobu"))), {})
        res.setdefault("client", {})[label] = {"ttft_client_s": round(ttft or -1, 3), "wall_s": round(wall, 2),
                                               "server": tm}
    # 6) sampling (model card defaults, thinking off): same seed -> same tokens, text readable, speculation off
    sb = {"messages": [{"role": "user", "content": "Nenne drei Farben, kommagetrennt."}], "max_tokens": 32,
          "chat_template_kwargs": {"enable_thinking": False}, "seed": 7}
    s1, j1 = post(oa_url, sb)
    s2, j2 = post(oa_url, sb)
    check("Sampling: gleicher Seed -> gleiche Tokens, Spekulation aus", s1 == s2 == 200 and
          j1["tobu"]["out_hash"] == j2["tobu"]["out_hash"] and j1["tobu"]["spec"] == "off (sampling)", str(j1)[:300])
    res["sampling"] = {"text": oa_parts(j1)[1], "tobu": j1["tobu"], "timings": j1["timings"]}
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="laufender Server (Teil 4b)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.url:
        check_server(a.url.rstrip("/"), a.out)
    else:
        if not os.environ.get("TOBU_TEST_TOKENIZER"):
            from test_tob2 import make_tokenizer
            make_tokenizer(TOKFILE)
        tok = HFTok.from_file(TOKFILE)
        test_template()
        test_splitter()
        test_detok_stop(tok)
        test_mapping()
        test_sampler()
        test_http()
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
