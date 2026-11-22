"""Two interfaces, one engine: OpenAI Chat Completions and Anthropic Messages.

parse_openai / parse_anthropic: request -> Req (messages in OpenAI form, as chat.render* expects them; tools in
OpenAI form). What tobu cannot do is reported as ApiError (HTTP 400) instead of silently done differently: images/documents,
forced tool calls (tool_choice required/any/tool), JSON mode/schema, logprobs, n > 1, server tools, Anthropic
mcp_servers/container. TOB-12 (only /v1/messages): prefill (last message assistant, text only, no thinking) is appended to the
generation prompt, search_result becomes text, parameter limits as with Anthropic (check_params), metadata,
service_tier, cache_control and citations only appear in tobu.ignored. Sampling (sampling.py): without values, the recommended values of the
model card per mode (thinking on/off), temperature = 0 -> greedy; logit_bias is not honoured (tobu.ignored).
Exceptions for Claude Code (L35, Anthropic format, noted in tobu.ignored): images/documents become a
placeholder, role "system" in the history is appended, tools with defer_loading and the billing marker in the
system prompt are dropped; output_config.format counts like output_format (structured output).
Script ban (tobu extension, both formats): ban_scripts: ["cjk"] hard-bans CJK tokens in the engine (flashnext,
also in MTP rounds); [] switches off the server default (--ban-scripts); unknown script -> 400. The default only applies
if the request says nothing and its messages themselves contain no CJK (ban_scripts_for).

Detok + StopFilter + Splitter: tokens -> text -> events ("think", s) / ("text", s) / ("tool", call), incremental
and with the same split as chat.parse (leading/trailing newlines removed, content up to the first
<tool_call>, tool calls per TOOL_RE). The non-stream response is the sum of the same events, so always equal
to the stream.

OpenAIOut / AnthropicOut: events -> JSON response or SSE in the original format (chat.completion.chunk;
message_start / content_block_* / message_delta / message_stop). Cache hits in the usage: OpenAI
prompt_tokens_details.cached_tokens, Anthropic cache_read_input_tokens (input_tokens = rest, as with Anthropic).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field

import chat
import sampling

MAX_NEW = 4096          # default without max_tokens (thinking off); the server caps it to the room left up to max_len
MAX_NEW_THINK = 16384   # default without max_tokens with thinking (L28: 8k was not enough for a third of the bundles)
UNSUPPORTED = ("logit_bias",)
DEFAULT_THINKING = True  # without enable_thinking; L45: the server sets it from the model's template (Qwen3.5-2B: off)
DEFAULT_THINKING_BUDGET = 0   # TOB-32: thinking budget when the request gives none (--default-thinking-budget), 0 = off
# reasoning_effort -> thinking budget when the request names no thinking_budget/max_thinking_tokens (20.11.: Open WebUI
# and other chat clients send no budget, Flash-Next otherwise thinks very long). 0 = no budget (not the server default either).
EFFORT_BUDGET = {"low": 4096, "medium": 16384, "high": 16384, "xhigh": 24576}
# reasoning_effort for the template: Bonsai knows xhigh/medium/low (high -> xhigh); Flash-Next (HF template) knows
# xhigh/medium/low/none, the template default would be xhigh – for Flash-Next the server sets EFFORT_TEMPLATE
# {"high": "medium"} and EFFORT_DEFAULT "medium" (Unsloth 20.11.).
EFFORT_TEMPLATE = {"high": "xhigh"}
EFFORT_DEFAULT: str | None = None
THINKING_ANSWER_RESERVE = 1024   # TOB-33: reserve for the answer (--thinking-answer-reserve), at least 25 % of max_tokens
SCRIPTS = ("cjk",)               # script ban: known scripts (tobu-fnx CONFIG ban_scripts)
DEFAULT_BAN_SCRIPTS: list[str] = []   # server default without ban_scripts in the request (--ban-scripts), [] = none


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.kind = status, kind


@dataclass
class Req:
    api: str                                  # "openai" | "anthropic"
    messages: list[dict]                      # OpenAI form
    tools: list[dict] | None                  # OpenAI form
    thinking: bool
    effort: str | None = None                 # reasoning_effort of the Bonsai template (xhigh | medium | low)
    kwargs: dict = field(default_factory=dict)   # further chat_template_kwargs (preserve_thinking, ...)
    max_tokens: int = MAX_NEW
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    include_usage: bool = False
    ignored: list[str] = field(default_factory=list)
    sampling: dict | None = None              # None = greedy
    parallel: bool = True                     # several tool calls per answer (parallel_tool_calls)
    force: list | None = None                 # tool_choice required/any ([]) or one tool ([name])
    structured: dict | None = None            # structured output: {"schema": …} or {"grammar": GBNF}
    gen_prompt: str = ""                      # generation prompt of the template (grammar prefill as in the fork)
    prefill: str = ""                         # TOB-12: Anthropic prefill, appended after the generation prompt
    thinking_budget: int | None = None        # TOB-32: at most this many thinking tokens (None = server default, 0 = off)
    penalties: dict | None = None             # penalties with greedy, only when given explicitly (presence_penalty …)
    ban_scripts: list[str] | None = None      # script ban of the request (None = not given -> server default)
    images: list | None = None                # TOB-37: [(media.Image, offset of the first image token)] after encode()
    deferred_tools: list | None = None        # tool search (Claude Code): tools not loaded yet – not in the
                                              # prompt, but in the grammar (otherwise it bent a call onto a
                                              # listed tool, e.g. Bash; incident 20.11.)

    def all_tools(self) -> list | None:
        """Tools for grammar and call detection: listed + deferred."""
        t = (self.tools or []) + (self.deferred_tools or [])
        return t or None


def _budget(v, where: str) -> int | None:
    """TOB-32: check thinking_budget (integer >= 0, None = not given)."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ApiError(f"{where} must be a non-negative integer")
    return v


# ------------------------------------------------------------------------------------------- script ban
# The same blocks as engine/tobu/scripts.hpp (kCjk): Hangul Jamo, CJK radicals,
# Kangxi, description characters, CJK punctuation, Kana, Bopomofo, Hangul compatibility Jamo, Kanbun, CJK strokes,
# enclosed/compatibility characters, Ext A, Han, Hangul Ext A, syllables, Ext B, compatibility ideographs, vertical
# and compatibility forms, full-/half-width, Kana supplements, planes 2/3.
CJK_RE = re.compile("[\u1100-\u11ff\u2e80-\u2fdf\u2ff0-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7ff\uf900-\ufaff"
                    "\ufe10-\ufe1f\ufe30-\ufe4f\uff00-\uffef\U0001aff0-\U0001b16f\U00020000-\U0003ffff]")


def _ban_scripts(body: dict) -> list[str] | None:
    """Check the request's ban_scripts: list of known scripts (None = not given, [] = explicitly none)."""
    v = body.get("ban_scripts")
    if v is None:
        return None
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ApiError("ban_scripts must be a list of strings")
    bad = [x for x in v if x not in SCRIPTS]
    if bad:
        raise ApiError(f"ban_scripts: unknown script {bad[0]!r} (known: {', '.join(SCRIPTS)})")
    return sorted(set(v))


def _texts(m: dict):
    c = m.get("content")
    if isinstance(c, str):
        yield c
    elif isinstance(c, list):
        yield from (p.get("text", "") for p in c if isinstance(p, dict))


def ban_scripts_for(req: Req) -> list[str]:
    """Effective script ban: the request's ban_scripts, else the server default – but not if the messages
    of the request themselves contain CJK (system, user, tool results: whoever writes or reads CJK also gets it
    back, e.g. for edits to files with CJK text). Assistant messages don't count: that is the model's own
    output, and Chinese drifting in there in particular must not switch the ban off."""
    if req.ban_scripts is not None:
        return list(req.ban_scripts)
    if not DEFAULT_BAN_SCRIPTS:
        return []
    for m in req.messages:
        if m.get("role") != "assistant" and any(CJK_RE.search(t or "") for t in _texts(m)):
            return []
    return list(DEFAULT_BAN_SCRIPTS)


# ------------------------------------------------------------------------------------------------------ images
VISION = False      # TOB-37: the server sets True when the engine has a vision tower (tobu-fnx --mmproj)


def _image_part(img) -> dict:
    """Image as content part: the template renders {"type": "image"} as <|vision_start|><|image_pad|><|vision_end|>."""
    return {"type": "image", "image": img}


def _media(fn, *a):
    import media

    try:
        return fn(*a)
    except media.MediaError as e:
        raise ApiError(str(e)) from e


def images_of(messages: list[dict]) -> list:
    """Images in render order (messages, content parts within them) – the order of their <|image_pad|> in the prompt."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            out += [p["image"] for p in c if isinstance(p, dict) and p.get("type") == "image" and "image" in p]
    return out


# ------------------------------------------------------------------------------------------------------ OpenAI
def _oa_content(c, role: str):
    if c is None or isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for it in c:
            if VISION and role == "user" and isinstance(it, dict) and it.get("type") == "image_url":
                import media

                iu = it.get("image_url")
                iu = {"url": iu} if isinstance(iu, str) else (iu or {})
                out.append(_image_part(_media(media.from_url, iu.get("url"), iu.get("detail"))))
                continue
            if not isinstance(it, dict) or it.get("type", "text") != "text" or "text" not in it:
                raise ApiError(f"message ({role}): only text content is supported, not {it.get('type') if isinstance(it, dict) else it!r}")
            out.append({"type": "text", "text": it["text"]})
        return out
    raise ApiError(f"message ({role}): content must be a string or a list of text parts")


def parse_openai(body: dict) -> Req:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ApiError("messages is missing or empty")
    if (body.get("n") or 1) != 1:
        raise ApiError("n > 1 is not supported")
    if body.get("logprobs") or body.get("top_logprobs"):
        raise ApiError("logprobs are not supported")
    tools = body.get("tools") or None
    structured = _structured_openai(body, tools)
    for t in tools or []:
        if t.get("type", "function") != "function" or "function" not in t:
            raise ApiError(f"tool type {t.get('type')} is not supported (only function)")
    tc = body.get("tool_choice")
    force = None                                  # forced tool call: [] = any tool, [name] = this one
    if tc in (None, "auto"):
        pass
    elif tc == "none":
        tools = None
    elif tc == "required" and tools:
        force = []
    elif isinstance(tc, dict) and tc.get("type") == "function" and tools:
        force = [(tc.get("function") or {}).get("name")]
    else:
        raise ApiError(f"tool_choice {tc!r} is not supported")
    out = []
    for m in msgs:
        role = m.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ApiError(f"role {role!r} is not supported")
        m = dict(m)
        m["content"] = _oa_content(m.get("content"), role)
        out.append(m)
    kw = dict(body.get("chat_template_kwargs") or {})
    thinking = kw.pop("enable_thinking", DEFAULT_THINKING)
    effort = kw.pop("reasoning_effort", None)
    tb = kw.pop("thinking_budget", None)          # TOB-32: in chat_template_kwargs or at the top level of the request
    tb = tb if tb is not None else body.get("thinking_budget")
    if tb is None and body.get("max_thinking_tokens") is not None:
        # max_thinking_tokens (some proxies rewrite thinking_budget to it): same meaning
        tb = body.get("max_thinking_tokens")
    tb = _budget(tb, "thinking_budget")
    re_ = body.get("reasoning_effort")
    raw_effort = re_ if re_ is not None else effort
    if re_ is not None:
        if re_ in ("none", "minimal"):
            thinking = False
        else:
            effort = re_
    if effort is None and thinking:
        effort = EFFORT_DEFAULT
    if isinstance(effort, str):
        effort = EFFORT_TEMPLATE.get(effort, effort)
    if tb is None and isinstance(raw_effort, str) and raw_effort in EFFORT_BUDGET:
        tb = EFFORT_BUDGET[raw_effort]
    stop = body.get("stop") or []
    stop = [stop] if isinstance(stop, str) else list(stop)
    mt = body.get("max_completion_tokens") or body.get("max_tokens") or (MAX_NEW_THINK if thinking else MAX_NEW)
    so = body.get("stream_options") or {}
    return Req("openai", out, tools, bool(thinking), effort, kw, int(mt), stop, bool(body.get("stream")),
               bool(so.get("include_usage")), [k for k in UNSUPPORTED if body.get(k)],
               sampling.resolve(body, bool(thinking)),
               parallel=body.get("parallel_tool_calls") is not False, force=force,
               structured=structured, thinking_budget=tb, penalties=sampling.greedy_penalties(body),
               ban_scripts=_ban_scripts(body))


# --------------------------------------------------------------------------------------------------- Anthropic
BILLING = "x-anthropic-billing-header:"     # Claude Code: billing marker as the first system block (L35)
OMIT = {"image": "[image omitted: text-only model]", "document": "[document omitted: text-only model]"}
REJECT = ("mcp_servers", "container")       # TOB-12: features tobu doesn't have -> 400 instead of silently dropping them
NOTE = ("metadata", "service_tier")         # TOB-12: no effect on the answer -> only in tobu.ignored
AN_ERRORS = {"invalid_request_error", "authentication_error", "permission_error", "not_found_error",
             "request_too_large", "rate_limit_error", "api_error", "overloaded_error", "billing_error",
             "timeout_error"}
AN_STATUS = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
             404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error", 529: "overloaded_error"}


def search_result_text(b: dict) -> str:
    """TOB-12: search_result block (Anthropic, RAG) as text: title, source, content of the text blocks. citations on the block
    have no effect (tobu does not cite)."""
    body = "\n".join(x.get("text", "") for x in (b.get("content") or []) if isinstance(x, dict) and x.get("type") == "text")
    return f"Search result: {b.get('title', '')}\nSource: {b.get('source', '')}\n{body}\n\n"


def _has(obj, key: str) -> bool:
    """Does key (with a value) occur anywhere in obj (blocks, lists)?"""
    if isinstance(obj, dict):
        return obj.get(key) not in (None, False, {}, []) or any(_has(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has(v, key) for v in obj)
    return False


def _num(body: dict, k: str, lo: float, hi: float | None, integer: bool = False):
    v = body.get(k)
    if v is None:
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and isinstance(v, float) and not v.is_integer()):
        raise ApiError(f"{k}: Input should be a valid {'integer' if integer else 'number'}")
    if v < lo or (hi is not None and v > hi):
        raise ApiError(f"{k}: must be {'>= ' + str(lo) if hi is None else f'between {lo} and {hi}'}, got {v}")


def check_params(body: dict):
    """TOB-12: limits like the Anthropic API, only for /v1/messages (Chat Completions unchanged)."""
    if body.get("max_tokens") is None:
        raise ApiError("max_tokens is required")
    _num(body, "max_tokens", 1, None, integer=True)
    _num(body, "temperature", 0.0, 1.0)
    _num(body, "top_p", 0.0, 1.0)
    _num(body, "top_k", 0, None, integer=True)


def _an_text(blocks, where: str, ign: list | None = None) -> str:
    """Text blocks -> text. With ign (L35, Claude Code): images/documents become a short placeholder (instead of 400,
    which would break the agent loop, e.g. Read on a PNG), tool_reference (tool search) a text line;
    noted in tobu.ignored."""
    if isinstance(blocks, str):
        return blocks
    out = []
    for b in blocks or []:
        t = b.get("type")
        if t == "text":
            out.append(b.get("text", ""))
        elif ign is not None and t in OMIT:
            out.append(OMIT[t])
            ign.append(f"{where}.{t}")
        elif ign is not None and t == "tool_reference":
            out.append(f"Tool {b.get('tool_name', '')} loaded.")
        elif ign is not None and t == "search_result":
            out.append(search_result_text(b))
        else:
            raise ApiError(f"{where}: block type {t!r} is not supported (only text)")
    return "".join(out)


def _an_system(system, ign: list) -> str:
    """System blocks -> text. Claude Code's billing marker (changes per session with the first prompt)
    does not belong in the prompt: drop it, otherwise prompts of different sessions differ from token ~5 on
    (prefix cache)."""
    if isinstance(system, list):
        keep = [b for b in system if not (b.get("type") == "text" and b.get("text", "").startswith(BILLING))]
        if len(keep) < len(system):
            ign.append("system.billing_header")
        system = keep
    return _an_text(system, "system")


def _prefill(last: dict, body: dict) -> str:
    """TOB-12: last message assistant = prefill. Text only; with thinking or structured output 400 (Anthropic: prefill
    works with neither). If it ends in whitespace: 400 as with Anthropic (otherwise the tokenization falls apart)."""
    c = last.get("content")
    if isinstance(c, list):
        if any(not isinstance(b, dict) or b.get("type") != "text" for b in c):
            raise ApiError("assistant prefill: only text blocks are supported in the final assistant message")
        c = "".join(b.get("text", "") for b in c)
    if not isinstance(c, str):
        raise ApiError("assistant prefill: content must be a string or a list of text blocks")
    if not c:
        return ""
    th = body.get("thinking") or {}
    if th.get("type") in ("enabled", "adaptive"):
        raise ApiError("assistant prefill is not supported with thinking enabled")
    if body.get("output_format") or (body.get("output_config") or {}).get("format"):
        raise ApiError("assistant prefill is not supported with structured outputs (output_format)")
    if c != c.rstrip():
        raise ApiError("final assistant content cannot end with trailing whitespace")
    return c


def parse_anthropic(body: dict, count: bool = False) -> Req:
    """count = True: /v1/messages/count_tokens (without max_tokens and without checking the sampling parameters)."""
    if not count:
        check_params(body)
    for k in REJECT:
        if body.get(k):
            raise ApiError(f"{k}: not supported by tobu")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ApiError("messages is missing or empty")
    prefill = ""
    if isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant":
        prefill = _prefill(msgs[-1], body)
        msgs = msgs[:-1]
    out: list[dict] = []
    ign: list[str] = [k for k in NOTE if body.get(k) is not None]
    if _has([body.get("system"), body.get("messages"), body.get("tools"), body.get("cache_control")], "cache_control"):
        ign.append("cache_control")
    if _has(body.get("messages"), "citations"):
        ign.append("citations")
    if body.get("system"):
        s = _an_system(body["system"], ign)
        if s:
            out.append({"role": "system", "content": s})
    for m in msgs:
        role, c = m.get("role"), m.get("content")
        if role == "system":
            # Claude Code (L35): environment, skills etc. as role "system" in the middle of the history (after the first
            # user message). The Bonsai template allows system only at the start -> append to the user message before it
            # (the position stays the same across rounds, prefix cache); before any message into the
            # system prompt, after assistant/tool as its own user message.
            s = _an_text(c, "system", ign)
            if not s:
                continue
            if all(x["role"] == "system" for x in out):
                if out:
                    out[0] = {"role": "system", "content": out[0]["content"] + "\n\n" + s}
                else:
                    out.append({"role": "system", "content": s})
            elif out[-1]["role"] == "user" and isinstance(out[-1]["content"], str):
                out[-1] = {"role": "user", "content": out[-1]["content"] + "\n\n" + s}
            else:
                out.append({"role": "user", "content": s})
            continue
        if role == "user":
            if isinstance(c, str):
                out.append({"role": "user", "content": c})
                continue
            text = []
            for b in c or []:
                t = b.get("type")
                if t == "tool_result":
                    bc = b.get("content") or ""
                    if VISION and isinstance(bc, list) and any(isinstance(x, dict) and x.get("type") == "image"
                                                               for x in bc):
                        # tool result with image (Claude Code: Read on a PNG, screenshot): parts in order
                        parts = [{"type": "text", "text": "Error: "}] if b.get("is_error") else []
                        for x in bc:
                            if x.get("type") == "image":
                                import media

                                parts.append(_image_part(_media(media.from_anthropic, x.get("source") or {})))
                            else:
                                parts.append({"type": "text", "text": _an_text([x], "tool_result", ign)})
                        out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": parts})
                        continue
                    rc = _an_text(bc, "tool_result", ign)
                    out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                "content": ("Error: " + rc) if b.get("is_error") else rc})
                elif t == "text":
                    text.append(b.get("text", ""))
                elif t == "image" and VISION:
                    import media

                    text.append(_image_part(_media(media.from_anthropic, b.get("source") or {})))
                elif t in OMIT:
                    text.append(OMIT[t])
                    ign.append(f"user.{t}")
                elif t == "search_result":
                    text.append(search_result_text(b))
                else:
                    raise ApiError(f"user: block type {t!r} is not supported (only text, tool_result, search_result)")
            if text and any(isinstance(x, dict) for x in text):      # with images: content parts in order
                out.append({"role": "user", "content": [x if isinstance(x, dict) else {"type": "text", "text": x}
                                                        for x in text]})
            elif text:
                out.append({"role": "user", "content": "".join(text)})
        elif role == "assistant":
            if isinstance(c, str):
                out.append({"role": "assistant", "content": c})
                continue
            text, think, calls = [], [], []
            for b in c or []:
                t = b.get("type")
                if t == "text":
                    text.append(b.get("text", ""))
                elif t == "thinking":
                    think.append(b.get("thinking", ""))
                elif t == "redacted_thinking":
                    continue
                elif t == "tool_use":
                    calls.append({"id": b.get("id"), "type": "function",
                                  "function": {"name": b["name"],
                                               "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
                else:
                    raise ApiError(f"assistant: block type {t!r} is not supported")
            msg = {"role": "assistant", "content": "".join(text)}
            if think:
                msg["reasoning_content"] = "".join(think)
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            raise ApiError(f"role {role!r} is not supported (user, assistant, system)")
    tools, deferred = None, []
    if body.get("tools"):
        tools = []
        loaded = _referenced_tools(body.get("messages"))
        for t in body["tools"]:
            if t.get("type") not in (None, "custom") or "input_schema" not in t:
                raise ApiError(f"tool {t.get('name')!r}: server/built-in tools are not supported")
            fn = {"name": t["name"]}
            if t.get("description"):
                fn["description"] = t["description"]
            fn["parameters"] = t["input_schema"]
            if t.get("defer_loading") and t["name"] not in loaded:
                # tool search: visible in the prompt only after tool_reference; until then only in the grammar
                ign.append("tools.defer_loading")
                deferred.append({"type": "function", "function": fn})
                continue
            tools.append({"type": "function", "function": fn})
        tools = tools or None
    tcb = body.get("tool_choice") or {}
    tc = tcb.get("type", "auto")
    force = None
    if tc == "none":
        tools = None
    elif tc == "any" and tools:
        force = []
    elif tc == "tool" and tools:
        force = [tcb.get("name")]
    elif tc != "auto":
        raise ApiError(f"tool_choice {tc} is not supported")
    th = body.get("thinking") or {}
    thinking = th.get("type") in ("enabled", "adaptive")
    tb = _budget(th.get("budget_tokens") if thinking and th.get("type") == "enabled" else None,
                 "thinking.budget_tokens")       # TOB-32: acts as thinking budget (previously ignored)
    if tb is None:
        tb = _budget(body.get("thinking_budget"), "thinking_budget")
    ign = sorted(set(ign))
    return Req("anthropic", out, tools, thinking, None, {}, int(body.get("max_tokens") or 1),
               list(body.get("stop_sequences") or []), bool(body.get("stream")), True, ign,
               sampling.resolve(body, thinking),
               parallel=not (body.get("tool_choice") or {}).get("disable_parallel_tool_use"),
               force=force, structured=_structured_anthropic(body, tools),
               prefill=prefill, thinking_budget=tb, penalties=sampling.greedy_penalties(body),
               ban_scripts=_ban_scripts(body), deferred_tools=deferred or None)


def _referenced_tools(obj) -> set:
    """Names from tool_reference blocks (result of the tool search) anywhere in the history: these deferred
    tools are loaded and belong in the tool list, as in the Anthropic API."""
    out, stack = set(), [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            if o.get("type") == "tool_reference" and isinstance(o.get("tool_name"), str):
                out.add(o["tool_name"])
            stack.extend(v for v in o.values() if isinstance(v, (dict, list)))
        elif isinstance(o, list):
            stack.extend(x for x in o if isinstance(x, (dict, list)))
    return out


def _schema_ok(schema) -> dict:
    import tool_format
    if not isinstance(schema, dict):
        raise ApiError("json_schema: the schema must be an object")
    try:
        text = tool_format.format_gbnf(schema)     # the fork's schema converter; errors as there -> 400
    except tool_format.SchemaError as e:
        raise ApiError("JSON schema conversion failed: " + str(e)[:300])
    _gbnf_ok(text)
    return {"schema": schema}


def _gbnf_ok(text: str):
    """If the fork can't load the grammar (syntax, left recursion), it reports 400 "failed to parse grammar"."""
    import tool_format
    try:
        tool_format.Grammar(text)
    except (ValueError, IndexError, AttributeError, RecursionError) as e:
        raise ApiError("Failed to initialize samplers: failed to parse grammar (" + str(e)[:200] + ")")


def _structured_openai(body: dict, tools) -> dict | None:
    """OpenAI response_format (json_object / json_schema incl. strict) and the llama.cpp server's fields
    `json_schema` and `grammar`. As in the fork: json_object without schema yields no grammar (empty schema), with
    tools a custom grammar is not allowed."""
    rf = body.get("response_format") or {}
    t = rf.get("type", "text") if isinstance(rf, dict) else "text"
    if t not in ("text", "json_object", "json_schema"):
        raise ApiError(f"response_format {t} is not supported")
    schema = None
    if t == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema")
    elif t == "json_object":
        schema = rf.get("schema") or None
    if body.get("json_schema") is not None:
        schema = body["json_schema"]
    if body.get("grammar"):
        if tools:
            raise ApiError("Cannot use custom grammar constraints with tools.")
        if not isinstance(body["grammar"], str):
            raise ApiError("grammar must be GBNF text")
        _gbnf_ok(body["grammar"])
        return {"grammar": body["grammar"]}
    return _schema_ok(schema) if schema else None


def _structured_anthropic(body: dict, tools) -> dict | None:
    """Anthropic structured outputs: output_format {"type": "json_schema", "schema": …} (analogous to OpenAI); newer
    form output_config.format (Claude Code 2.1.304: session title {title}, L35)."""
    of = body.get("output_format") or (body.get("output_config") or {}).get("format") or {}
    if not of:
        return None
    if of.get("type") != "json_schema" or not of.get("schema"):
        raise ApiError(f"output_format {of.get('type')} is not supported (only json_schema)")
    return _schema_ok(of["schema"])


# ------------------------------------------------------------------------------------------- tokens -> events
class Detok:
    """Incremental decoding (byte-level BPE): output chunks only once they don't end in the middle of a UTF-8
    character. decode(a + b) == decode(a) + decode(b) as soon as a ends on a character boundary."""

    def __init__(self, tok, skip: set[int]):
        self.tok, self.skip = tok, skip
        self.pend: list[int] = []

    def feed(self, ids: list[int]) -> str:
        self.pend += [t for t in ids if t not in self.skip]
        s = self.tok.decode(self.pend) if self.pend else ""
        if s.endswith("�"):
            return ""
        self.pend = []
        return s

    def flush(self) -> str:
        s = self.tok.decode(self.pend) if self.pend else ""
        self.pend = []
        return s


def _hold(buf: str, tags) -> int:
    """Length of the longest end of buf that is the start of one of the tags (hold back until it is clear whether it becomes a tag)."""
    best = 0
    for tag in tags:
        for n in range(min(len(tag) - 1, len(buf)), best, -1):
            if buf.endswith(tag[:n]):
                best = n
                break
    return best


class StopFilter:
    """Stop strings (OpenAI stop, Anthropic stop_sequences) in the generated text: text up to before it, then end."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.buf = ""
        self.hit: str | None = None

    def feed(self, s: str) -> str:
        if not self.stops or self.hit is not None:
            return s if self.hit is None else ""
        self.buf += s
        first = None
        for st in self.stops:
            i = self.buf.find(st)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, st)
        if first:
            self.hit = first[1]
            out, self.buf = self.buf[: first[0]], ""
            return out
        k = _hold(self.buf, self.stops)
        out, self.buf = self.buf[: len(self.buf) - k], self.buf[len(self.buf) - k:]
        return out

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out if self.hit is None else ""


_CALL_OPEN = re.compile(r"<tool_call>\s*<function=([^>\n]+)>")
_PARAM_OPEN = re.compile(r"<parameter=([^>\n]+)>")
_ENDS = ("</parameter>", "</tool_call>", "</function>")


def _jstr(s: str) -> str:
    """JSON escape of a string chunk (like json.dumps, ensure_ascii=False, without quotes). The escape
    works per character, so escape(a + b) == escape(a) + escape(b) holds – the basis of the chunked stream."""
    return json.dumps(s, ensure_ascii=False)[1:-1]


class _ArgStream:
    """Tool arguments already while decoding (20.11.: opencode stalled on large write/edit).

    Reads the growing tool part (Splitter.tbuf) in the Qwen format <tool_call><function=N><parameter=K>…</parameter>
    and yields ("tool_open", {i, id, name}) and ("tool_args", {i, s}) – chunks of exactly the JSON that at the end
    json.dumps(input) yields (same order, same escape). Rules as chat.TOOL_RE/PARAM_RE/OPEN_RE:
    value without one leading \n, end at the first </parameter> (without one \n before it), if missing, at the end of the function body.
    Only what belongs to the value under every reading is streamed; non-strings only at the parameter end
    (json.loads like Splitter._calls). Anything unusual (duplicate parameter, text between parameters, newline in the
    name) freezes the stream of this call – the output part sends the rest at the end from the finished call.
    Only switched on with tool grammar (there order and form are guaranteed)."""

    def __init__(self, types: dict):
        self.types = types
        self.pos = 0                 # processed in tbuf up to here
        self.calls: list[dict] = []  # per opened call: id, name, keys, frozen
        self.cur: dict | None = None
        self.phase = "seek"          # seek | call | value
        self.val: dict | None = None

    def _open_call(self, tbuf: str, ev: list) -> bool:
        m = _CALL_OPEN.search(tbuf, self.pos)
        if not m:
            return False
        c = {"i": len(self.calls), "id": uuid.uuid4().hex[:24], "name": m.group(1).strip(), "keys": [],
             "frozen": False, "body": m.end()}
        unknown = c["name"] not in self.types       # TOB-59: name in no list -> error at the end, stream nothing
        if tbuf[c["body"]: c["body"] + 1] == "\n":
            c["body"] += 1
        self.calls.append(c)
        self.cur, self.pos, self.phase = c, m.end(), "call"
        self.gap = c["body"]     # end of the last parameter (or body start) for the OPEN_RE condition
        if unknown:
            c["frozen"] = True
        else:
            ev.append(("tool_open", {"i": c["i"], "id": c["id"], "name": c["name"]}))
        return True

    def _send(self, s: str, ev: list):
        if s and not self.cur["frozen"]:
            ev.append(("tool_args", {"i": self.cur["i"], "s": s}))

    def _key(self, k: str, ev: list):
        self._send(("{" if not self.cur["keys"] else ", ") + json.dumps(k, ensure_ascii=False) + ": ", ev)
        self.cur["keys"].append(k)

    def feed(self, tbuf: str) -> list:
        ev: list = []
        while True:
            if self.phase == "seek":
                if not self._open_call(tbuf, ev):
                    return ev
                continue
            if self.phase == "call":
                ip, ie = tbuf.find("<parameter=", self.pos), tbuf.find("</tool_call>", self.pos)
                if ie >= 0 and (ip < 0 or ie < ip):          # call finished: continue with the next one
                    self.pos, self.phase, self.cur = ie + len("</tool_call>"), "seek", None
                    continue
                if ip < 0:
                    return ev
                m = _PARAM_OPEN.match(tbuf, ip)
                if not m:
                    if "\n" in tbuf[ip:] or tbuf.find(">", ip) >= 0:   # no valid name: no longer safe
                        self.cur["frozen"] = True
                        self.pos = ip + 1
                        continue
                    return ev                                  # name still incomplete
                k = m.group(1).strip()
                if tbuf[self.gap: ip].strip() or k in self.cur["keys"]:
                    self.cur["frozen"] = True              # text between parameters / duplicate: form not guaranteed
                vs = m.end()
                if vs >= len(tbuf):
                    return ev                              # whether a \n follows is still open
                if tbuf[vs] == "\n":
                    vs += 1
                string = self.types.get(self.cur["name"], {}).get(k, "string") == "string"
                self.val = {"k": k, "start": vs, "sent": vs, "scan": vs, "string": string}
                if string:
                    self._key(k, ev)
                    self._send('"', ev)
                self.pos, self.phase = vs, "value"
                continue
            # value: find the first end (</parameter>; else body end via </function>\s*</tool_call> or </tool_call>)
            v = self.val
            end = None
            j = v["scan"]
            while True:
                hits = [(tbuf.find(t, j), t) for t in _ENDS]
                hits = [(i, t) for i, t in hits if i >= 0]
                if not hits:
                    break
                i, t = min(hits)
                if t == "</parameter>":
                    end, after = i, i + len(t)
                    break
                if t == "</tool_call>":
                    end, after = i, i                      # body end; the call ends there (phase call finds it)
                    break
                rest = tbuf[i + len(t):]                    # </function>: only body end if \s*</tool_call> follows
                st = rest.lstrip()
                if st.startswith("</tool_call>"):
                    end, after = i, i
                    break
                if not st or "</tool_call>".startswith(st):
                    v["scan"] = max(v["scan"], i - 16)
                    return self._partial(tbuf, i, ev)      # still open: safe up to before </function>
                j = i + 1                                   # part of the value
            if end is None:
                v["scan"] = max(v["scan"], len(tbuf) - 16)
                hold = 0
                for t in _ENDS:                             # end of the buffer could start an end marker
                    for n in range(min(len(t) - 1, len(tbuf)), 0, -1):
                        if t.startswith(tbuf[-n:]):
                            hold = max(hold, n)
                            break
                return self._partial(tbuf, len(tbuf) - hold, ev)
            stop = end - 1 if end > v["start"] and tbuf[end - 1] == "\n" else end
            value = tbuf[v["start"]: stop]
            if v["string"]:
                self._send(_jstr(tbuf[v["sent"]: stop]), ev)
                self._send('"', ev)
            else:
                x = value
                try:
                    x = json.loads(value)
                except json.JSONDecodeError:
                    pass
                self._key(v["k"], ev)
                self._send(json.dumps(x, ensure_ascii=False), ev)
            self.val, self.pos, self.phase, self.gap = None, after, "call", after

    def _partial(self, tbuf: str, upto: int, ev: list) -> list:
        """Send a string value up to `upto`, hold back a \n at the end (could be the one omitted before the end)."""
        v = self.val
        if v["string"]:
            upto = max(v["sent"], upto)
            if upto > v["sent"] and tbuf[upto - 1] == "\n":
                upto -= 1
            if upto > v["sent"]:
                self._send(_jstr(tbuf[v["sent"]: upto]), ev)
                v["sent"] = upto
        return ev


class Splitter:
    """Model output (text from the generation prompt on) -> events, same split as chat.parse."""

    def __init__(self, tools: list[dict] | None, thinking: bool, stream_args: bool = False):
        self.tools = tools
        self.state = "think" if thinking else "text"
        self.buf = ""
        self.started = False        # text already output in the current section (leading \n removed)
        self.pend_nl = ""           # held-back \n at the section end (strip("\n"))
        self.at_start = True        # at the very start of the output: "<think>" is removed (removeprefix)
        self.tbuf, self.tdone = "", 0
        self.types = chat._param_types(tools)
        self.names = set(self.types)
        self.required = chat._required(tools)
        self.missing: list[str] = []          # "tool.parameter" of missing required fields
        self.args = _ArgStream(self.types) if (stream_args and tools) else None   # arguments in the stream

    def _emit(self, kind: str, s: str, ev: list):
        if not s:
            return
        if not self.started:
            s = s.lstrip("\n")
            if not s:
                return
            self.started = True
        body = s.rstrip("\n")
        if body:
            ev.append((kind, self.pend_nl + body))
            self.pend_nl = s[len(body):]
        else:
            self.pend_nl += s

    def _section(self, state: str):
        self.state, self.started, self.pend_nl = state, False, ""

    def _calls(self, final: bool, ev: list):
        ms = list(chat.TOOL_RE.finditer(self.tbuf))
        for m in ms[self.tdone:]:
            name, args = m.group(1).strip(), {}
            for k, v in chat.params(m.group(2)):
                if self.args is not None and k in args:
                    # duplicate parameter (the grammar allows optional ones more than once): in the stream the first one counts – the
                    # client already has it; without stream the last one, as before
                    continue
                if self.types.get(name, {}).get(k, "string") != "string":
                    try:
                        v = json.loads(v)
                    except json.JSONDecodeError:
                        pass
                args[k] = v
            self.missing += [f"{name}.{k}" for k in self.required.get(name, []) if k not in args]
            k = ms.index(m)                        # n-th call = n-th opened one in the stream (same rule)
            op = self.args.calls[k] if (self.args is not None and k < len(self.args.calls)) else None
            ev.append(("tool", {"id": op["id"] if op else uuid.uuid4().hex[:24], "name": name, "input": args,
                                **({"streamed": op["i"]} if op else {})}))
        self.tdone = len(ms)

    def feed(self, s: str) -> list:
        ev: list = []
        self.buf += s
        while True:
            if self.state == "think":
                if self.at_start:
                    if len(self.buf) < 7 and "<think>".startswith(self.buf):
                        return ev
                    self.buf = self.buf.removeprefix("<think>")
                    self.at_start = False
                tag = "</think>"
            elif self.state == "text":
                self.at_start = False
                if not self.tools:
                    self._emit("text", self.buf, ev)
                    self.buf = ""
                    return ev
                tag = "<tool_call>"
            else:
                self.tbuf += self.buf
                self.buf = ""
                if self.args is not None:
                    ev += self.args.feed(self.tbuf)
                self._calls(False, ev)
                return ev
            i = self.buf.find(tag)
            if self.state == "think" and self.tools:         # like the fork: <tool_call> also ends the thinking
                j = self.buf.find("<tool_call>")
                if j >= 0 and (i < 0 or j < i):
                    self._emit("think", self.buf[:j], ev)
                    self.buf = self.buf[j:]
                    self._section("tools")
                    continue
            if i >= 0:
                self._emit(self.state, self.buf[:i], ev)
                if self.state == "think":
                    self.buf = self.buf[i + len(tag):]
                    self._section("text")
                else:
                    self.buf = self.buf[i:]
                    self._section("tools")
                continue
            k = _hold(self.buf, [tag, "<tool_call>"] if (self.state == "think" and self.tools) else [tag])
            self._emit(self.state, self.buf[: len(self.buf) - k], ev)
            self.buf = self.buf[len(self.buf) - k:]
            return ev

    def finish(self) -> list:
        ev: list = []
        if self.state == "think" and self.at_start:
            self.buf = self.buf.removeprefix("<think>")
        if self.state in ("think", "text"):
            self._emit(self.state, self.buf, ev)
        else:
            self.tbuf += self.buf
            if self.args is not None:
                ev += self.args.feed(self.tbuf)
            self._calls(True, ev)
        self.buf = ""
        return ev

    @property
    def phase(self) -> tuple[str, str | None]:
        """Current phase for /tobu/metrics: think | text | tool (+ name of the tool once known)."""
        if self.state != "tools":
            return self.state, None
        cur = self.args.cur if self.args is not None else None
        if cur is None:                          # without argument stream: last <function=…> head (per round, not per token)
            i = self.tbuf.rfind("<function=")
            j = self.tbuf.find(">", i) if i >= 0 else -1
            return "tool", (self.tbuf[i + 10: j].strip() if j > i >= 0 else None)
        return "tool", cur["name"]

    @property
    def malformed(self) -> str | None:
        """Tool part that is not a complete call (or an unknown tool) -> error message."""
        if self.state != "tools":
            return None
        ms = list(chat.TOOL_RE.finditer(self.tbuf))
        # TOB-59: also call heads without a complete call – the tool grammar's escape hatch ends the output
        # after the head of a name in no list; that is an unknown tool, not a "broken" call
        heads = [m.group(1).strip() for m in _CALL_OPEN.finditer(self.tbuf)]
        unknown = [n for n in heads if n not in self.names]
        if unknown:
            return f"model called an unknown tool: {unknown[0]} (not in tools)"
        if not ms:
            return "model output contains <tool_call> but no complete call"
        if self.missing:
            return "tool call without required parameter: " + ", ".join(self.missing)
        return None


# -------------------------------------------------------------------------------------------------- outputs
STREAM_MISMATCH = [0]     # counter: streamed arguments were not a prefix of the finished call (must never happen)


def _rest(sent: str, call: dict) -> str:
    """Rest of the arguments after the already streamed part. By construction the stream yields a prefix of json.dumps(input);
    if it did not, a client could no longer assemble the chunks into valid JSON – then counted and
    reported (tests check that this never happens)."""
    full = json.dumps(call["input"], ensure_ascii=False)
    if full.startswith(sent):
        return full[len(sent):]
    STREAM_MISMATCH[0] += 1
    print(f"tool stream: arguments diverge ({len(sent)} sent, call {call.get('name')})", flush=True)
    return ""


def _sse(obj: dict, event: str | None = None) -> bytes:
    head = f"event: {event}\n".encode() if event else b""
    return head + b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


@dataclass
class Done:
    finish: str                  # stop | length | tool_calls | stop_sequence
    stop_sequence: str | None
    prompt_n: int
    cached: int
    completion_n: int
    timings: dict
    tobu: dict
    reasoning_n: int | None = None    # TOB-32: tokens in the thinking block (None = thinking off)


class OpenAIOut:
    keepalive = b": keepalive\n\n"

    def __init__(self, model: str, req: Req):
        self.model, self.req = model, req
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.reasoning, self.content, self.calls = [], [], []

    def _chunk(self, delta: dict, finish=None, extra=None) -> bytes:
        d = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if extra:
            d.update(extra)
        return _sse(d)

    @staticmethod
    def usage(d: Done) -> dict:
        u = {"prompt_tokens": d.prompt_n, "completion_tokens": d.completion_n,
             "total_tokens": d.prompt_n + d.completion_n, "prompt_tokens_details": {"cached_tokens": d.cached}}
        if d.reasoning_n is not None:            # TOB-32: as OpenAI (completion_tokens_details.reasoning_tokens)
            u["completion_tokens_details"] = {"reasoning_tokens": d.reasoning_n}
        if (d.tobu or {}).get("usage_tobu"):     # per-request metrics, extra field
            u["tobu"] = d.tobu["usage_tobu"]
        return u

    @staticmethod
    def finish(d: Done) -> str:
        return "stop" if d.finish == "stop_sequence" else d.finish

    def call(self, c: dict) -> dict:
        return {"id": "call_" + c["id"], "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["input"], ensure_ascii=False)}}

    # Stream
    def start(self, info: dict) -> bytes:
        return self._chunk({"role": "assistant", "content": ""})

    def event(self, kind: str, x) -> bytes:
        if kind == "think":
            self.reasoning.append(x)
            return self._chunk({"reasoning_content": x})
        if kind == "text":
            self.content.append(x)
            return self._chunk({"content": x})
        if kind == "tool_open":                  # arguments in the stream: announce the call, then chunks
            self.sent = getattr(self, "sent", {})
            self.sent[x["i"]] = ""
            return self._chunk({"tool_calls": [{"index": x["i"], "id": "call_" + x["id"], "type": "function",
                                                "function": {"name": x["name"], "arguments": ""}}]})
        if kind == "tool_args":
            self.sent[x["i"]] += x["s"]
            return self._chunk({"tool_calls": [{"index": x["i"], "function": {"arguments": x["s"]}}]})
        i = len(self.calls)
        self.calls.append(x)
        if "streamed" in x and x["streamed"] in getattr(self, "sent", {}):
            rest = _rest(self.sent.pop(x["streamed"]), x)
            return self._chunk({"tool_calls": [{"index": x["streamed"], "function": {"arguments": rest}}]}) if rest \
                else b""
        return self._chunk({"tool_calls": [{"index": i, **self.call(x)}]})

    def end(self, d: Done) -> bytes:
        out = self._chunk({}, self.finish(d), {"usage": self.usage(d), "timings": d.timings, "tobu": d.tobu})
        if self.req.include_usage:
            out += _sse({"id": self.id, "object": "chat.completion.chunk", "created": self.created,
                         "model": self.model, "choices": [], "usage": self.usage(d)})
        return out + b"data: [DONE]\n\n"

    def error(self, e: ApiError) -> bytes:
        return _sse({"error": {"message": str(e), "type": e.kind}}) + b"data: [DONE]\n\n"

    # non-stream: the same events, assembled
    def body(self, d: Done) -> dict:
        msg = {"role": "assistant", "content": "".join(self.content) or None,
               "reasoning_content": "".join(self.reasoning) or None}
        if self.calls:
            msg["tool_calls"] = [self.call(c) for c in self.calls]
        return {"id": self.id, "object": "chat.completion", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": self.finish(d)}],
                "usage": self.usage(d), "timings": d.timings, "tobu": d.tobu}

    @staticmethod
    def error_body(e: ApiError) -> dict:
        return {"error": {"message": str(e), "type": e.kind}}


class AnthropicOut:
    keepalive = _sse({"type": "ping"}, "ping")

    def __init__(self, model: str, req: Req):
        self.model, self.req = model, req
        self.id = "msg_" + uuid.uuid4().hex[:24]
        self.blocks: list[dict] = []
        self.open: str | None = None          # type of the open block in the stream
        self.request_id: str | None = None    # TOB-12: also in the stream's error event

    @staticmethod
    def usage(d: Done | None, info: dict | None = None) -> dict:
        p, c = (d.prompt_n, d.cached) if d else (info["prompt_n"], info["cached"])
        u = {"input_tokens": p - c, "cache_creation_input_tokens": 0, "cache_read_input_tokens": c,
             "output_tokens": d.completion_n if d else 1}
        if d is not None and (d.tobu or {}).get("usage_tobu"):     # per-request metrics (extra field)
            u["tobu"] = d.tobu["usage_tobu"]
        return u

    @staticmethod
    def stop_reason(d: Done) -> str:
        return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                "stop_sequence": "stop_sequence"}[d.finish]

    def _msg(self) -> dict:
        return {"id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": [],
                "stop_reason": None, "stop_sequence": None}

    def _close(self) -> bytes:
        if self.open is None:
            return b""
        i = len(self.blocks) - 1
        out = b""
        if self.open == "thinking":           # like Anthropic: signature_delta before the end (tobu does not sign: "")
            out = _sse({"type": "content_block_delta", "index": i,
                        "delta": {"type": "signature_delta", "signature": self.blocks[i]["signature"]}},
                       "content_block_delta")
        self.open = None
        return out + _sse({"type": "content_block_stop", "index": i}, "content_block_stop")

    # Stream
    def start(self, info: dict) -> bytes:
        return _sse({"type": "message_start", "message": {**self._msg(), "usage": self.usage(None, info)}},
                    "message_start")

    def event(self, kind: str, x) -> bytes:
        out = b""
        if kind in ("think", "text"):
            t = "thinking" if kind == "think" else "text"
            if self.open != t:
                out += self._close()
                self.blocks.append({"type": t, t: ""} if t == "text" else {"type": t, t: "", "signature": ""})
                self.open = t
                out += _sse({"type": "content_block_start", "index": len(self.blocks) - 1,
                             "content_block": dict(self.blocks[-1])}, "content_block_start")
            self.blocks[-1][t] += x
            out += _sse({"type": "content_block_delta", "index": len(self.blocks) - 1,
                         "delta": {"type": t + "_delta", t: x}}, "content_block_delta")
            return out
        if kind == "tool_open":                  # arguments in the stream: open tool_use block, then input_json_delta
            out += self._close()
            self.blocks.append({"type": "tool_use", "id": "toolu_" + x["id"], "name": x["name"], "input": {}})
            self.open = "tool_use"
            self.sent = getattr(self, "sent", {})
            self.sent[x["i"]] = (len(self.blocks) - 1, "")
            return out + _sse({"type": "content_block_start", "index": len(self.blocks) - 1,
                               "content_block": {"type": "tool_use", "id": "toolu_" + x["id"], "name": x["name"],
                                                 "input": {}}}, "content_block_start")
        if kind == "tool_args":
            bi, sent = self.sent[x["i"]]
            self.sent[x["i"]] = (bi, sent + x["s"])
            return _sse({"type": "content_block_delta", "index": bi,
                         "delta": {"type": "input_json_delta", "partial_json": x["s"]}}, "content_block_delta")
        if "streamed" in x and x["streamed"] in getattr(self, "sent", {}):
            bi, sent = self.sent.pop(x["streamed"])
            self.blocks[bi]["input"] = x["input"]
            rest = _rest(sent, x)
            if rest:
                out += _sse({"type": "content_block_delta", "index": bi,
                             "delta": {"type": "input_json_delta", "partial_json": rest}}, "content_block_delta")
            return out + self._close()
        out += self._close()
        blk = {"type": "tool_use", "id": "toolu_" + x["id"], "name": x["name"], "input": x["input"]}
        self.blocks.append(blk)
        i = len(self.blocks) - 1
        out += _sse({"type": "content_block_start", "index": i,
                     "content_block": {"type": "tool_use", "id": blk["id"], "name": blk["name"], "input": {}}},
                    "content_block_start")
        out += _sse({"type": "content_block_delta", "index": i,
                     "delta": {"type": "input_json_delta", "partial_json": json.dumps(x["input"], ensure_ascii=False)}},
                    "content_block_delta")
        out += _sse({"type": "content_block_stop", "index": i}, "content_block_stop")
        return out

    def end(self, d: Done) -> bytes:
        out = self._close()
        out += _sse({"type": "message_delta", "delta": {"stop_reason": self.stop_reason(d), "stop_sequence": d.stop_sequence},
                     "usage": self.usage(d), "tobu": d.tobu}, "message_delta")
        return out + _sse({"type": "message_stop"}, "message_stop")

    def error(self, e: ApiError) -> bytes:
        return _sse(self.error_body(e, self.request_id), "error")

    def body(self, d: Done) -> dict:
        return {**self._msg(), "content": self.blocks, "stop_reason": self.stop_reason(d),
                "stop_sequence": d.stop_sequence, "usage": self.usage(d), "tobu": d.tobu}

    @staticmethod
    def error_body(e: ApiError, request_id: str | None = None) -> dict:
        """Error type from the Anthropic list (own types like tool_call_error -> by status, else api_error)."""
        kind = e.kind if e.kind in AN_ERRORS else AN_STATUS.get(e.status, "api_error")
        d = {"type": "error", "error": {"type": kind, "message": str(e)}}
        if request_id:
            d["request_id"] = request_id
        return d
