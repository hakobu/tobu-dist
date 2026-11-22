"""Tool search (Claude Code, defer_loading/tool_reference): deferred tools not in the prompt, but in the
grammar and in call detection; after tool_reference like listed ones. Reason: the model wanted to call
mcp__task-board__plan_steps, the grammar only knew the listed tools and redirected every call to Bash.

    python3 test_deferred_tools.py        (without GPU, without model)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail=None):
    print(("ok   " if ok else "FEHLER ") + name + ("" if ok or detail is None else f" – {detail}"))
    if not ok:
        FAILS.append(name)


SCHEMA = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
PLAN = {"type": "object", "properties": {"steps": {"type": "string"}}, "required": ["steps"]}
TOOLS = [{"name": "Bash", "description": "Run a shell command", "input_schema": SCHEMA},
         {"name": "mcp__task-board__plan_steps", "description": "Plan steps", "input_schema": PLAN,
          "defer_loading": True}]


def names(ts) -> list[str]:
    return [t["function"]["name"] for t in ts or []]


def main() -> int:
    body = {"model": "x", "max_tokens": 64, "tools": TOOLS, "messages": [{"role": "user", "content": "plane das"}]}
    r = api.parse_anthropic(body)
    check("zurückgestellt: nicht im Prompt (tools)", names(r.tools) == ["Bash"], names(r.tools))
    check("zurückgestellt: in der Grammatik (all_tools)", names(r.all_tools()) == ["Bash", "mcp__task-board__plan_steps"],
          names(r.all_tools()))
    check("zurückgestellt: in tobu.ignored vermerkt", "tools.defer_loading" in r.ignored, r.ignored)

    tu = "toolu_01"
    body2 = dict(body, messages=[
        {"role": "user", "content": "plane das"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": tu, "name": "ToolSearch", "input": {"query": "plan"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu, "content": [
            {"type": "tool_reference", "tool_name": "mcp__task-board__plan_steps"}]}]}])
    body2["tools"] = TOOLS + [{"name": "ToolSearch", "input_schema": {"type": "object", "properties": {}}}]
    r2 = api.parse_anthropic(body2)
    check("nach tool_reference: geladen, im Prompt (wie Anthropic-API)",
          "mcp__task-board__plan_steps" in names(r2.tools) and not r2.deferred_tools, (names(r2.tools), r2.deferred_tools))

    # call detection: a call of the deferred tool is a valid call (not broken, not Bash)
    out = ('<tool_call>\n<function=mcp__task-board__plan_steps>\n<parameter=steps>\na, b\n</parameter>\n'
           '</function>\n</tool_call>')
    for label, tools in (("nur gelistete (alt)", r.tools), ("alle (neu)", r.all_tools())):
        sp = api.Splitter(tools, False)
        evs = sp.feed(out) + sp.finish()
        calls = [e for e in evs if e[0] == "tool"]
        got = [c[1].get("name") if isinstance(c[1], dict) else c[1] for c in calls]
        if label.startswith("alle"):
            check("Aufruf des zurückgestellten Werkzeugs erkannt, Name unverändert",
                  got == ["mcp__task-board__plan_steps"] and not sp.malformed, (got, sp.malformed))
        else:
            check("Gegenprobe: mit nur gelisteten Werkzeugen gilt der Aufruf als kaputt (ursprünglicher Fehler)",
                  bool(sp.malformed) or got != ["mcp__task-board__plan_steps"], (got, sp.malformed))
    print("FEHLER: " + (", ".join(FAILS) if FAILS else "keine"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
