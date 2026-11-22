"""Chat template and answer parser like llama-server --jinja, for Qwen3.5 (and Bonsai 2, same vocabulary).

render(): OpenAI messages + tools -> prompt text like the official template from ggml-org/llama.cpp
(`templates/Qwen3.5-4B.jinja`, models/templates). render = render_py (pure Python, without jinja2);
render_jinja renders via the real template and serves as reference (test_jf.py: all E2 turns byte-identical). Tool-call arguments arrive as a JSON string and are, as in
llama.cpp, parsed before rendering (the template iterates over `arguments|items`); role "developer" -> "system".
parse(): model output (from the generation prompt "<think>\\n") -> reasoning_content, content, tool_calls.
Parameter values in the Qwen XML format are raw; type from the tool schema: string stays text, else JSON (fallback text).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "templates" / "Qwen3.5-4B.jinja"
EOT = "<|im_end|>"


def _raise(msg):
    import jinja2

    raise jinja2.exceptions.TemplateError(msg)


def _env():
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)
    # like transformers/minja: json.dumps with ", " and ": ", without ASCII escapes
    env.filters["tojson"] = lambda x, indent=None: json.dumps(x, ensure_ascii=False, indent=indent)
    env.globals["raise_exception"] = _raise
    return env


_TPL: dict = {}


class TemplateText(str):
    """Template text instead of a path (L45: chat template from the GGUF, tokenizer.chat_template)."""


def _template(template):
    if isinstance(template, TemplateText):
        key = "text:" + hashlib.sha256(template.encode("utf-8")).hexdigest()
        if key not in _TPL:
            _TPL[key] = _env().from_string(str(template))
        return _TPL[key]
    path = str(template or TEMPLATE)
    if path not in _TPL:
        _TPL[path] = _env().from_string(Path(path).read_text(encoding="utf-8"))
    return _TPL[path]


def render_jinja(messages: list[dict], tools: list[dict] | None, enable_thinking: bool = True,
                 add_generation_prompt: bool = True, template: str | Path | None = None, **kwargs) -> str:
    """Via the real template (jinja2): reference for render_py (Qwen3.5), or with `template` another template,
    e.g. the one from the GGUF. kwargs = further chat_template_kwargs (preserve_thinking, ...)."""
    tpl = _template(template)
    msgs = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":     # pi/OpenAI role for the system prompt; llama-server treats it as system
            m["role"] = "system"
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                tcs.append(tc)
            m["tool_calls"] = tcs
        msgs.append(m)
    return tpl.render(messages=msgs, tools=tools or None, add_generation_prompt=add_generation_prompt,
                      enable_thinking=enable_thinking, **kwargs)


def _param_types(tools: list[dict] | None) -> dict:
    out = {}
    for t in tools or []:
        fn = t.get("function", t)
        props = (fn.get("parameters") or {}).get("properties") or {}
        out[fn["name"]] = {k: (v or {}).get("type", "string") for k, v in props.items()}
    return out


def _required(tools: list[dict] | None) -> dict:
    """Required parameters per tool (error message to the client if one is missing)."""
    out = {}
    for t in tools or []:
        fn = t.get("function", t)
        out[fn["name"]] = list((fn.get("parameters") or {}).get("required") or [])
    return out


# </function> may be missing (L33: then the call ends at </tool_call>, the last parameter at its end)
TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\n?(.*?)(?:</function>\s*)?</tool_call>", re.S)
# line break after <parameter=…> and before </parameter> as in the template, but optional (L33: the model writes
# short values sometimes on one line, e.g. <parameter=path>a.py</parameter>; previously the parameter was silently dropped)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
OPEN_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?\Z", re.S)


def params(body: str) -> list[tuple[str, str]]:
    """Parameters of a <function> block. If the closing </parameter> is missing for the last parameter (L33 cross-check:
    the model sometimes writes </function> right away), the rest up to the end of the function counts as its value."""
    out, end = [], 0
    for m in PARAM_RE.finditer(body):
        out.append((m.group(1).strip(), m.group(2)))
        end = m.end()
    m = OPEN_RE.search(body, end)
    if m and not body[end: m.start()].strip():
        out.append((m.group(1).strip(), m.group(2)))
    return out


def parse(out: str, tools: list[dict] | None, thinking: bool = True) -> dict:
    """-> {"reasoning_content", "content", "tool_calls"} (OpenAI form, arguments as JSON string)."""
    text = out[: -len(EOT)] if out.endswith(EOT) else out
    reasoning = ""
    if thinking or "</think>" in text:
        i, j = text.find("</think>"), text.find("<tool_call>") if tools else -1
        if j >= 0 and (i < 0 or j < i):    # like the fork: <tool_call> ends the thinking even without </think>
            reasoning, text = text[:j], text[j:]
        elif i >= 0:
            reasoning, text = text[:i], text[i + len("</think>"):]
        else:   # output ends inside the thinking (length limit)
            reasoning, text = text, ""
        reasoning = reasoning.removeprefix("<think>").strip("\n")
    types = _param_types(tools)
    calls = []
    first = text.find("<tool_call>")
    content = (text if first < 0 else text[:first]).strip("\n")
    for m in TOOL_RE.finditer(text):
        name = m.group(1).strip()
        args = {}
        for k, v in params(m.group(2)):
            if types.get(name, {}).get(k, "string") != "string":
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
            args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return {"reasoning_content": reasoning, "content": content, "tool_calls": calls}


# ---------------------------------------------------------------------------------------------- without jinja2
TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_TAIL = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")


def _content(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for it in c:
            if not isinstance(it, dict) or "text" not in it:
                raise ValueError("only text content is supported")
            out.append(it["text"])
        return "".join(out)
    raise ValueError("Unexpected content type.")


def _arg(v) -> str:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)          # like jinja2 `|string` (Python representation)


def render_py(messages: list[dict], tools: list[dict] | None, enable_thinking: bool = True,
              add_generation_prompt: bool = True) -> str:
    """The same output as the template templates/Qwen3.5-4B.jinja (text contents only), without jinja2.
    Equality with render() is checked by test_jf.py over all E2 turns."""
    msgs = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        msgs.append(m)
    if not msgs:
        raise ValueError("No messages provided.")
    o = []
    if tools:
        o.append("<|im_start|>system\n" + TOOLS_HEAD)
        for t in tools:
            o.append("\n" + json.dumps(t, ensure_ascii=False))
        o.append(TOOLS_TAIL)
        if msgs[0].get("role") == "system":
            c = _content(msgs[0].get("content")).strip()
            if c:
                o.append("\n\n" + c)
        o.append("<|im_end|>\n")
    elif msgs[0].get("role") == "system":
        o.append("<|im_start|>system\n" + _content(msgs[0].get("content")).strip() + "<|im_end|>\n")
    last_q = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            c = _content(msgs[i].get("content")).strip()
            if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                last_q = i
                break
    if last_q is None:
        raise ValueError("No user query found in messages.")
    n = len(msgs)
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = _content(m.get("content")).strip()
        if role == "system":
            if i:
                raise ValueError("System message must be at the beginning.")
        elif role == "user":
            o.append("<|im_start|>user\n" + content + "<|im_end|>\n")
        elif role == "assistant":
            rc = m.get("reasoning_content")
            if not isinstance(rc, str):
                rc = ""
                if "</think>" in content:
                    rc = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = content.split("</think>")[-1].lstrip("\n")
            rc = rc.strip()
            if i > last_q:
                o.append("<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + content)
            else:
                o.append("<|im_start|>assistant\n" + content)
            tcs = m.get("tool_calls")
            if tcs and isinstance(tcs, list):
                for j, tc in enumerate(tcs):
                    fn = tc.get("function", tc)
                    head = "<tool_call>\n<function=" + fn["name"] + ">\n"
                    o.append(("\n\n" + head if content.strip() else head) if j == 0 else "\n" + head)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args or "{}")
                        except json.JSONDecodeError:
                            args = {}
                    for k, v in (args or {}).items():
                        o.append("<parameter=" + k + ">\n" + _arg(v) + "\n</parameter>\n")
                    o.append("</function>\n</tool_call>")
            o.append("<|im_end|>\n")
        elif role == "tool":
            if i > 0 and msgs[i - 1].get("role") != "tool":
                o.append("<|im_start|>user")
            o.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if i < n - 1 and msgs[i + 1].get("role") != "tool":
                o.append("<|im_end|>\n")
            elif i == n - 1:
                o.append("<|im_end|>\n")
        else:
            raise ValueError("Unexpected message role.")
    if add_generation_prompt:
        o.append("<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"))
    return "".join(o)


render = render_py


# ---------------------------------------------------------------------------------------------- Bonsai 2 / Qwen3.8
EFFORT = {"xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                   "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
          "medium": "",
          "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
                 "without unnecessary elaboration."}
TEMPLATE_BONSAI = HERE / "templates" / "Bonsai-2-27B.gguf.jinja"


def render_bonsai_py(messages: list[dict], tools: list[dict] | None, enable_thinking: bool = True,
                     add_generation_prompt: bool = True, reasoning_effort: str | None = None,
                     preserve_thinking: bool = True) -> str:
    """The same output as the template from the Bonsai GGUF (templates/Bonsai-2-27B.gguf.jinja, Qwen3.8 family),
    without jinja2. Differences to Qwen3.5: thinking instruction per reasoning_effort (default xhigh) in the
    system prompt as soon as thinking is on; earlier answers keep their <think> (preserve_thinking, default on);
    parameter values: text raw, everything else as JSON. Equality with the template: test_api.py."""
    msgs = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        msgs.append(m)
    if not msgs:
        raise ValueError("No messages provided.")
    ri = ""
    if enable_thinking is not False:
        eff = reasoning_effort or "xhigh"
        if eff not in EFFORT:
            raise ValueError(f"Unexpected reasoning effort {eff}. Supported types are xhigh (default), medium, and low.")
        ri = EFFORT[eff]
    o = []
    if tools:
        o.append("<|im_start|>system\n" + (ri + "\n\n" if ri else "") + TOOLS_HEAD)
        for t in tools:
            o.append("\n" + json.dumps(t, ensure_ascii=False))
        o.append(TOOLS_TAIL)
        if msgs[0].get("role") == "system":
            c = _content(msgs[0].get("content")).strip()
            if c:
                o.append("\n\n" + c)
        o.append("<|im_end|>\n")
    elif msgs[0].get("role") == "system":
        c = _content(msgs[0].get("content")).strip()
        if c:
            o.append("<|im_start|>system\n" + (ri + "\n\n" if ri else "") + c + "<|im_end|>\n")
        elif ri:
            o.append("<|im_start|>system\n" + ri + "<|im_end|>\n")
    elif ri:
        o.append("<|im_start|>system\n" + ri + "<|im_end|>\n")
    last_q = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            c = _content(msgs[i].get("content")).strip()
            if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                last_q = i
                break
    if last_q is None:
        raise ValueError("No user query found in messages.")
    n = len(msgs)
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = _content(m.get("content")).strip()
        if role == "system":
            if i:
                raise ValueError("System message must be at the beginning.")
        elif role == "user":
            o.append("<|im_start|>user\n" + content + "<|im_end|>\n")
        elif role == "assistant":
            rc = m.get("reasoning_content")
            rc = rc.strip() if isinstance(rc, str) else ""
            if preserve_thinking is not False or i > last_q:
                o.append("<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + content)
            else:
                o.append("<|im_start|>assistant\n" + content)
            tcs = m.get("tool_calls")
            if tcs and isinstance(tcs, list):
                for j, tc in enumerate(tcs):
                    fn = tc.get("function", tc)
                    head = "<tool_call>\n<function=" + fn["name"] + ">\n"
                    o.append(("\n\n" + head if content.strip() else head) if j == 0 else "\n" + head)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args != "" else None
                        except json.JSONDecodeError:
                            args = {}
                    for k, v in (args or {}).items():
                        o.append("<parameter=" + k + ">\n" + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                                 + "\n</parameter>\n")
                    o.append("</function>\n</tool_call>")
            o.append("<|im_end|>\n")
        elif role == "tool":
            if i > 0 and msgs[i - 1].get("role") != "tool":
                o.append("<|im_start|>user")
            o.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if i < n - 1 and msgs[i + 1].get("role") != "tool":
                o.append("<|im_end|>\n")
            elif i == n - 1:
                o.append("<|im_end|>\n")
        else:
            raise ValueError("Unexpected message role.")
    if add_generation_prompt:
        o.append("<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"))
    return "".join(o)


RENDER = {"qwen35": render_py, "bonsai": render_bonsai_py}


# ---------------------------------------------------------------------------------------------- L45: template from the GGUF
KNOWN = {"qwen35": TEMPLATE, "bonsai": TEMPLATE_BONSAI}     # own Python versions (render_py, render_bonsai_py)


def template_info(text: str | None) -> dict:
    """Which template does the model bring (tokenizer.chat_template), and what follows from it?
      kind          qwen35 | bonsai: text-identical with a template for which there is a checked Python version
                    (starting point for known models, test_jf); gguf: other template -> jinja2 with the GGUF text;
                    none: no template in the GGUF -> Qwen3.5 template as assumption (with a note)
      tool_format   qwen3-xml (<tool_call><function=…><parameter=…>: format grammar like the fork) | json | None
      default_thinking   generation prompt without enable_thinking: True (<think>\n) | False (<think></think>) | None
    No code execution: jinja2 only renders text (sandbox-free environment like llama-server --jinja)."""
    if not text:
        return {"kind": "none", "sha": None, "tool_format": "qwen3-xml", "default_thinking": True,
                "note": "no chat template in the GGUF - assuming the Qwen3.5 template"}
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    kind = next((k for k, p in KNOWN.items() if p.exists() and p.read_text(encoding="utf-8") == text), "gguf")
    if "<tool_call>" in text and "<function=" in text and "<parameter=" in text:
        tf = "qwen3-xml"
    elif "<tool_call>" in text and ("\"name\"" in text or "tojson" in text):
        tf = "json"
    else:
        tf = None
    dt = None
    try:
        out = _template(TemplateText(text)).render(messages=[{"role": "user", "content": "x"}], tools=None,
                                                   add_generation_prompt=True)
        tail = out[out.rfind("<|im_start|>assistant"):] if "<|im_start|>assistant" in out else out[-40:]
        dt = False if "</think>" in tail else True if "<think>" in tail else None
    except Exception as e:                        # noqa: BLE001  (jinja2 missing or template does not render)
        return {"kind": kind, "sha": sha, "tool_format": tf, "default_thinking": None,
                "note": f"template does not render: {type(e).__name__}: {str(e)[:120]}"}
    return {"kind": kind, "sha": sha, "tool_format": tf, "default_thinking": dt, "note": ""}
