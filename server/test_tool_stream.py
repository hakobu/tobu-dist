"""Tool arguments in the stream (api._ArgStream): the streamed chunks give exactly json.dumps(input) of the
finished call, for every split into chunks; the finished calls are the same as without stream; OpenAI/Anthropic
output assembles into valid JSON. No GPU, no network.

python3 -I test_tool_stream.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api  # noqa: E402

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}" + (f" – {str(detail)[:300]}" if not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]}}},
         {"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
             "filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"},
             "replaceAll": {"type": "boolean"}, "n": {"type": "integer"}, "opts": {"type": "object"}}}}}]


def call(name, params, close_param=True, nl=True):
    s = f"<tool_call>\n<function={name}>\n"
    for k, v in params:
        s += f"<parameter={k}>" + ("\n" if nl else "") + v + (("\n" if nl else "") + "</parameter>\n" if close_param else "")
    return s + "</function>\n</tool_call>"


CASES = {
    "einfach": "Ich schreibe die Datei.\n\n" + call("write", [("filePath", "/w/a.py"), ("content", "print('hi')\n")]),
    "groß + escapes": call("write", [("filePath", "/w/b.py"), ("content", "\n".join(
        f'zeile {i}: "zitat" \\ tab\t ü€😀 </param' for i in range(300)))]),
    "nicht-string": call("edit", [("filePath", "x"), ("oldString", "a"), ("newString", "b"), ("replaceAll", "true"),
                                  ("n", "42"), ("opts", '{"a": [1, 2]}')]),
    "kaputtes json": call("edit", [("filePath", "x"), ("n", "zwölf"), ("opts", "{kaputt")]),
    "leer + nur \\n": call("edit", [("filePath", ""), ("oldString", "\n"), ("newString", "\n\n")]),
    "einzeilig": call("write", [("filePath", "a.py"), ("content", "x = 1")], nl=False),
    "zwei aufrufe": call("write", [("filePath", "a"), ("content", "1\n2")]) + "\n" +
                    call("edit", [("filePath", "a"), ("oldString", "1"), ("newString", "3")]),
    "marker im wert": call("write", [("filePath", "a"), ("content", "x <parameter=y> </function> z </tool_call")]),
    "fehlendes </parameter>": "<tool_call>\n<function=write>\n<parameter=filePath>\na\n</parameter>\n"
                              "<parameter=content>\nletzter wert\nohne ende\n</function>\n</tool_call>",
    "ohne </function>": "<tool_call>\n<function=write>\n<parameter=filePath>\na\n</parameter>\n</tool_call>",
    "doppelter parameter": call("write", [("filePath", "a"), ("filePath", "b"), ("content", "c")]),
    "text zwischen parametern": "<tool_call>\n<function=write>\n<parameter=filePath>\na\n</parameter>\nblabla\n"
                                "<parameter=content>\nc\n</parameter>\n</function>\n</tool_call>",
    "crlf": call("write", [("filePath", "a"), ("content", "a\r\nb\r\n")]),
    "unbekannter parameter": call("write", [("filePath", "a"), ("extra", "{\"x\": 1}"), ("content", "c")]),
}


def run(text: str, stream: bool, rng: random.Random | None, thinking=False):
    sp = api.Splitter(TOOLS, thinking, stream_args=stream)
    ev = []
    i = 0
    while i < len(text):
        n = rng.randint(1, 9) if rng else len(text)
        ev += sp.feed(text[i:i + n])
        i += n
    ev += sp.finish()
    return ev, sp


def pieces_ok(ev) -> tuple[bool, str]:
    sent: dict[int, str] = {}
    for k, x in ev:
        if k == "tool_open":
            sent[x["i"]] = ""
        elif k == "tool_args":
            sent[x["i"]] += x["s"]
        elif k == "tool" and "streamed" in x:
            full = json.dumps(x["input"], ensure_ascii=False)
            got = sent.get(x["streamed"])
            if got is None or not full.startswith(got):
                return False, f"{x['name']}: gesendet {got!r:.120} kein Präfix von {full!r:.120}"
    return True, ""


def main() -> int:
    rng = random.Random(7)
    for name, text in CASES.items():
        base, _ = run(text, False, None)
        calls0 = [x for k, x in base if k == "tool"]
        ok_all, why, opened, progressed = True, "", 0, 0
        for rep in range(60):
            ev, sp = run(text, True, rng)
            calls = [x for k, x in ev if k == "tool"]
            same = [(c["name"], c["input"]) for c in calls] == [(c["name"], c["input"]) for c in calls0]
            if name == "doppelter parameter":   # in the stream the first one counts (client has it), without stream the last one
                same = [c["input"] for c in calls] == [{"filePath": "a", "content": "c"}] and \
                    [c["input"] for c in calls0] == [{"filePath": "b", "content": "c"}]
            ok, why2 = pieces_ok(ev)
            opened = max(opened, sum(1 for k, _ in ev if k == "tool_open"))
            progressed = max(progressed, sum(1 for k, _ in ev if k == "tool_args"))
            if not (same and ok):
                ok_all, why = False, why2 or f"Aufrufe anders: {calls} vs {calls0}"
                break
        check(f"{name}: Stücke = Präfix des fertigen JSON, Aufrufe wie ohne Strom (60 Zerlegungen)", ok_all, why)
    # large write call: content arrives while decoding, not only at the end
    text = CASES["groß + escapes"]
    sp = api.Splitter(TOOLS, False, stream_args=True)
    first_args_at, seen = None, 0
    for i in range(0, len(text), 5):
        ev = sp.feed(text[i:i + 5])
        seen += sum(len(x["s"]) for k, x in ev if k == "tool_args")
        if seen > 1000 and first_args_at is None:
            first_args_at = i
    check("großer Aufruf: > 1000 Zeichen Argumente vor 15 % des Textes", first_args_at is not None and
          first_args_at < 0.15 * len(text), first_args_at)
    # output: OpenAI deltas assembled = arguments; Anthropic partial_json assembled = input
    req = api.Req.__new__(api.Req)
    for k, v in {"tools": TOOLS, "stream": True, "include_usage": False, "thinking": False}.items():
        setattr(req, k, v)
    for name in ("zwei aufrufe", "nicht-string", "fehlendes </parameter>"):
        ev, _ = run(CASES[name], True, rng)
        oa = api.OpenAIOut("m", req)
        args: dict[int, str] = {}
        for k, x in ev:
            for line in oa.event(k, x).decode().splitlines():
                if line.startswith("data:"):
                    for tc in json.loads(line[5:])["choices"][0]["delta"].get("tool_calls") or []:
                        args[tc["index"]] = args.get(tc["index"], "") + tc["function"].get("arguments", "")
        want = [x["input"] for k, x in ev if k == "tool"]
        check(f"OpenAI {name}: Deltas -> gültiges JSON = Argumente", [json.loads(args[i]) for i in sorted(args)] == want,
              (args, want))
        an = api.AnthropicOut("m", req)
        parts: dict[int, str] = {}
        for k, x in ev:
            for line in an.event(k, x).decode().splitlines():
                if line.startswith("data:"):
                    d = json.loads(line[5:])
                    if d.get("type") == "content_block_delta" and d["delta"]["type"] == "input_json_delta":
                        parts[d["index"]] = parts.get(d["index"], "") + d["delta"]["partial_json"]
        tu = [b for b in an.blocks if b["type"] == "tool_use"]
        check(f"Anthropic {name}: partial_json -> input, Blöcke mit input",
              [json.loads(parts[i]) for i in sorted(parts)] == want and [b["input"] for b in tu] == want, (parts, tu))
    check("kein Präfix-Fehler gezählt", api.STREAM_MISMATCH[0] == 0, api.STREAM_MISMATCH)
    # thinking -> tool, phase
    ev, sp = run("<think>überlege</think>\n" + CASES["einfach"], True, rng, thinking=True)
    check("Denken dann Aufruf: think-Text, Aufruf gestreamt, Phase tool/write",
          any(k == "think" for k, _ in ev) and any(k == "tool_open" for k, _ in ev) and sp.phase == ("tool", "write"),
          (ev[:3], sp.phase))
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
