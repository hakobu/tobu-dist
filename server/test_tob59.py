"""TOB-59: escape route of the tool grammar for a name in no list (no silent redirect to a listed
tool). The grammar also allows an unknown name after "<function=" and
then ends the output after the head; the server reports "unknown tool" instead of a call to Bash.
The native grammar (C++) is checked by engine/tobu/test_tool_escape.cpp.

    python3 test_tob59.py        (no GPU, no model)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api  # noqa: E402
import tool_format as tf  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail=None):
    print(("ok   " if ok else "FEHLER ") + name + ("" if ok or detail is None else f" – {detail}"))
    if not ok:
        FAILS.append(name)


TOOLS = [{"type": "function", "function": {"name": "Bash", "parameters": {
             "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "mcp__task-board__plan_steps", "parameters": {
             "type": "object", "properties": {"steps": {"type": "string"}}, "required": ["steps"]}}}]
HEAD = "<tool_call>\n<function="
CALL = HEAD + "Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"


def run(m: tf.Matcher, text: str):
    """(not rejected, complete) after the text."""
    st = m.start()
    for c in text:
        st = m.step(st, ord(c))
        if not st:
            return False, False
    return True, () in st


def main() -> int:
    esc = tf.Matcher(tf.Grammar(tf.fork_gbnf(TOOLS, escape=True)))
    old = tf.Matcher(tf.Grammar(tf.fork_gbnf(TOOLS)))
    check("unbekannter Name: Kopf erlaubt, danach Ende", run(esc, HEAD + "mcp__other__thing>\n") == (True, True))
    check("unbekannter Name: keine Argumente danach", run(esc, HEAD + "mcp__other__thing>\n<parameter=")[0] is False)
    check("Gegenprobe ohne Ausweg: unbekannter Name abgelehnt (Umbiegen)", run(old, HEAD + "mcp__other__thing>")[0]
          is False)
    check("gelisteter Name: nach dem Kopf nicht vollständig", run(esc, HEAD + "Bash>\n") == (True, False))
    check("gelisteter Aufruf ganz: vollständig", run(esc, CALL) == (True, True))
    check("gelisteter Name: unbekannter Parameter abgelehnt", run(esc, HEAD + "Bash>\n<parameter=nope>")[0] is False)
    check("Präfix/Verlängerung eines gelisteten Namens gilt als unbekannt",
          run(esc, HEAD + "Bas>") == (True, True) and run(esc, HEAD + "Bash2>") == (True, True))
    check("leerer Name abgelehnt", run(esc, HEAD + ">")[0] is False)
    check("nach gültigem Aufruf: unbekannter beendet ebenfalls", run(esc, CALL + "\n" + HEAD + "nope>") == (True, True))
    f = tf.ToolFormat(TOOLS, None, set(), set())
    check("ToolFormat (auto): mit Ausweg", "tool-esc-0" in f.text)
    f = tf.ToolFormat(TOOLS, None, set(), set(), force=["Bash"])
    check("ToolFormat (benanntes tool_choice): ohne Ausweg", "tool-esc-0" not in f.text)

    # server: output after the escape -> clear error message, in the stream no tool_open for the unknown name
    sp = api.Splitter(TOOLS, False, stream_args=True)
    evs = sp.feed("Ich plane.\n" + HEAD + "mcp__other__thing>\n") + sp.finish()
    check("Splitter: Fehler nennt das unbekannte Werkzeug",
          sp.malformed == "model called an unknown tool: mcp__other__thing (not in tools)", sp.malformed)
    check("Splitter: kein tool_open/tool für den unbekannten Namen", not [e for e in evs if e[0].startswith("tool")], evs)
    sp = api.Splitter(TOOLS, False, stream_args=True)
    evs = sp.feed(CALL) + sp.finish()
    check("Splitter: gelisteter Aufruf unverändert gültig",
          sp.malformed is None and evs[0][0] == "tool_open" and evs[-1][0] == "tool" and evs[-1][1]["name"] == "Bash",
          (sp.malformed, [e[0] for e in evs]))
    print("FEHLER: " + (", ".join(FAILS) if FAILS else "keine"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
