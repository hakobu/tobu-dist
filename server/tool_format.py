"""Format grammar for tool calls (L33, decision 20.10.2026): enforces the tool-call scaffold like the fork.

Modelled on llama.cpp (PrismML fork prism-b10777, `common_chat_params_init_qwen3_coder` + json-schema-to-grammar):
a *lazy* GBNF grammar that only takes effect once the model begins a tool call (token `<tool_call>` or the
text `<function=NAME>`), and then enforces tags, function names, parameter names and required fields. String parameters
are free up to the first `\\n</parameter>\\n`; parameters of other types follow their JSON schema (as in the fork).
Normal answers stay pure greedy.

Selection as in the fork at temperature 0 (common_sampler_sample): first the unconstrained argmax; if it doesn't fit the
grammar, the argmax among the allowed tokens. So the output is token-identical to "greedy + the same grammar"
without speculation: drafts are cut to the allowed beginning before verifying, accepted draft tokens are
unconstrained argmaxes *and* allowed, only the bonus token is checked and if needed replaced from the logits of this row
(`FormatVerifier`).

Building blocks:
  fork_gbnf(tools, parallel)  GBNF text as the fork generates it (test: same language as the text reported by the fork)
  Grammar(text)               GBNF parser (subset as in llama.cpp: literals, character classes, groups, * + ? {m,n})
  Matcher                     stack automaton like llama-grammar.cpp (per code point, incomplete UTF-8 at the token end)
  ToolFormat                  lazy trigger (token ID or word), allowed/accept/best per token
  FormatVerifier              engine wrapper for decoder.generate (start/verify with grammar)
"""

from __future__ import annotations

import bisect
import copy
import json
import re
from functools import lru_cache
from itertools import permutations

import numpy as np

import json_schema_to_grammar as j2g

# ------------------------------------------------------------------------------------------------ GBNF like the fork
PARAM_END = "\n</parameter>\n"


def lit(s: str) -> str:
    """GBNF literal (format_literal in the fork)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n") + '"'


def _type(s: dict):
    return s.get("type")


def resolves_to_string(s, refs: dict | None = None, seen: frozenset = frozenset()) -> bool:
    """common_schema_info::resolves_to_string (refs: resolved $ref targets from resolve_refs)."""
    if not isinstance(s, dict):
        return False
    ref = s.get("$ref")
    if ref is not None:
        return ref not in seen and refs is not None and resolves_to_string(refs.get(ref), refs, seen | {ref})
    t = s.get("type")
    if t == "string" or (isinstance(t, list) and "string" in t):
        return True
    for k in ("oneOf", "anyOf"):
        if any(resolves_to_string(a, refs, seen) for a in s.get(k) or []):
            return True
    if s.get("allOf") and all(resolves_to_string(a, refs, seen) for a in s["allOf"]):
        return True
    if isinstance(s.get("const"), str):
        return True
    return any(isinstance(v, str) for v in s.get("enum") or [])


class _Unsupported(Exception):
    pass


def _esc_len(p: str, pos: int) -> int:
    """gbnf_escape_length (C++ in the fork)."""
    if pos + 1 >= len(p) or p[pos] != "\\":
        return 0
    c = p[pos + 1]
    if c in 'trn\\"[]':
        return 2
    n = {"x": 2, "u": 4, "U": 8}.get(c)
    if n is None or pos + 2 + n > len(p):
        return 0
    return 2 + n if all(h in "0123456789abcdefABCDEF" for h in p[pos + 2:pos + 2 + n]) else 0


def _stoi(x: str) -> int:
    m = re.match(r"\s*[+-]?\d+", x)
    if not m:
        raise _Unsupported("invalid number in curly brackets")
    return int(m.group())


def _rep(item: str, lo: int, hi: int | None) -> str:
    """build_repetition (C++) without separator; hi None = unbounded."""
    if hi == 0:
        return ""
    if lo == 0 and hi == 1:
        return item + "?"
    if lo == 1 and hi is None:
        return item + "+"
    if lo == 0 and hi is None:
        return item + "*"
    return item + "{" + str(lo) + "," + ("" if hi is None else str(hi)) + "}"


class _ForkConverter(j2g.SchemaConverter):
    """The C++ version in the fork (common/json-schema-to-grammar.cpp) differs from the Python reference; here as in the fork:
    - patterns (`pattern`): _pattern_to_rule rebuilt from the C++ version (MIT, The ggml authors), byte-wise as there.
      Patterns without an equivalent in GBNF (not anchored with ^…$, \\d, (?=…), inner anchors, …) become an
      arbitrary string; only invalid patterns (unbalanced brackets, open character class, quantifier without target,
      backslash at the end) are an error.
    - remote $ref (https://…) are not loaded; the fork substitutes an arbitrary JSON object for them."""

    def _visit_pattern(self, pattern, name):
        snap = dict(self._rules)
        try:
            return self._pattern_to_rule(pattern.encode("utf-8").decode("latin-1"), name)
        except _Unsupported:
            self._rules.clear()
            self._rules.update(snap)
            return self._add_rule(name, self._add_primitive("string", j2g.PRIMITIVE_RULES["string"]))

    def _pattern_to_rule(self, pattern: str, name: str) -> str:
        if len(pattern) < 2 or pattern[0] != "^" or pattern[-1] != "$":
            raise _Unsupported("not anchored")
        sub = pattern[1:-1]
        n = len(sub)
        st = {"i": 0, "depth": 0}
        sub_ids: dict[str, str | None] = {}
        nonlit = set("|.()[]{}*+?^$")

        def to_rule(x):
            return '"' + x[0] + '"' if x[1] else x[0]

        def transform():
            seq = []

            def join_seq():
                ret, lit_ = [], ""
                for item in seq:
                    if item[1]:
                        lit_ += item[0]
                    else:
                        if lit_:
                            ret.append((lit_, True))
                            lit_ = ""
                        ret.append(item)
                if lit_:
                    ret.append((lit_, True))
                return (" ".join(to_rule(x) for x in ret), False)

            while st["i"] < n:
                i = st["i"]
                c = sub[i]
                if c == ".":
                    seq.append((self._add_rule("dot", j2g.DOTALL if self._dotall else j2g.DOT), False))
                    st["i"] += 1
                elif c == "(":
                    st["i"] += 1
                    i = st["i"]
                    if i < n and sub[i] == "?":
                        if i + 1 < n and sub[i + 1] == ":":
                            st["i"] += 2
                        else:
                            raise _Unsupported("unsupported group syntax")
                    st["depth"] += 1
                    if st["depth"] > 100:
                        raise _Unsupported("pattern nesting too deep")
                    seq.append(("(" + to_rule(transform()) + ")", False))
                elif c == ")":
                    st["i"] += 1
                    if st["depth"] == 0:
                        raise ValueError(f"Invalid pattern {pattern}: unbalanced parentheses")
                    st["depth"] -= 1
                    return join_seq()
                elif c in "^$":
                    raise _Unsupported("anchor inside the pattern")
                elif c == "[":
                    sq = c
                    i += 1
                    while i < n and sub[i] != "]":
                        if sub[i] == "\\":
                            k = _esc_len(sub, i)
                            if k == 0:
                                raise _Unsupported("unsupported escape in character class")
                            sq += sub[i:i + k]
                            i += k
                        else:
                            sq += sub[i]
                            i += 1
                    if i >= n:
                        raise ValueError(f"Invalid pattern {pattern}: unterminated character class")
                    st["i"] = i + 1
                    seq.append((sq + "]", False))
                elif c == "|":
                    seq.append(("|", False))
                    st["i"] += 1
                elif c in "*+?":
                    if not seq:
                        raise ValueError(f"Invalid pattern {pattern}: nothing to repeat")
                    seq[-1] = (to_rule(seq[-1]) + c, False)
                    st["i"] += 1
                elif c == "{":
                    cb = c
                    i += 1
                    while i < n and sub[i] != "}":
                        cb += sub[i]
                        i += 1
                    if i >= n:
                        raise _Unsupported("unterminated curly brackets")
                    st["i"] = i + 1
                    nums = cb[1:].split(",")
                    if len(nums) not in (1, 2):
                        raise _Unsupported("wrong number of values in curly brackets")
                    if len(nums) == 1:
                        lo = hi = _stoi(nums[0])
                    else:
                        lo = _stoi(nums[0]) if nums[0] else 0
                        hi = _stoi(nums[1]) if nums[1] else None
                    if not seq:
                        raise ValueError(f"Invalid pattern {pattern}: nothing to repeat")
                    s_, is_lit = seq[-1]
                    if not is_lit:
                        if sub_ids.get(s_) is None:
                            sub_ids[s_] = None
                            sub_ids[s_] = self._add_rule(f"{name}-{len(sub_ids)}", s_)
                        s_ = sub_ids[s_]
                    seq[-1] = (_rep('"' + s_ + '"' if is_lit else s_, lo, hi), False)
                else:
                    lit_ = ""
                    while i < n:
                        if sub[i] == "\\":
                            if i == n - 1:
                                raise ValueError(f"Invalid pattern {pattern}: trailing backslash")
                            nx = sub[i + 1]
                            if nx in "^$.[]()|{}*+?":
                                lit_ += nx
                                i += 2
                            else:
                                k = _esc_len(sub, i)
                                if k == 0:
                                    raise _Unsupported("unsupported escape")
                                lit_ += sub[i:i + k]
                                i += k
                        elif sub[i] == '"':
                            lit_ += '\\"'
                            i += 1
                        elif sub[i] not in nonlit and (i == n - 1 or not lit_ or sub[i + 1] == "."
                                                       or sub[i + 1] not in nonlit):
                            lit_ += sub[i]
                            i += 1
                        else:
                            break
                    st["i"] = i
                    if not lit_:
                        raise _Unsupported("unsupported character")
                    seq.append((lit_, True))
            return join_seq()

        rule = to_rule(transform())
        if st["depth"] != 0:
            raise ValueError(f"Invalid pattern {pattern}: unbalanced parentheses")
        body = '"\\"" (' + rule + ') "\\""'
        try:                             # bytes as in the C++ text; no valid UTF-8 -> the fork can't load the GBNF
            body = body.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"failed to parse grammar (pattern {pattern.encode('latin-1').decode('utf-8', 'replace')})")
        return self._add_rule(name, body)

    def resolve_refs(self, schema, url):
        def remote(n):
            if isinstance(n, list):
                for x in n:
                    remote(x)
            elif isinstance(n, dict):
                ref = n.get("$ref")
                if isinstance(ref, str) and ref.startswith("https://"):
                    self._refs.setdefault(ref, {})
                elif ref is None:
                    for v in n.values():
                        remote(v)
        remote(schema)
        return super().resolve_refs(schema, url)


class _Rules:
    """Rule collection like the fork's GBNF builder. JSON schemas are converted by the fork's schema converter
    (json_schema_to_grammar.py, taken over verbatim); own rules (tool scaffold, thinking block) are added."""

    def __init__(self):
        self.conv = _ForkConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        self.rules: dict[str, str] = self.conv._rules
        self.n = 0

    def add(self, name: str, body: str) -> str:
        return self.conv._add_rule(name, body)

    def text(self) -> str:
        return "\n".join(f"{k} ::= {v}" for k, v in self.rules.items()) + "\n"


def resolve_refs(R: _Rules, sch, url: str):
    """Resolve $ref like builder.resolve_refs in the fork (only local '#/…', no fetches over the network). The Python reference
    does not descend further below a $ref; the fork's C++ version does (e.g. root {"$ref": "#/$defs/node",
    "$defs": …} with recursion). Hence resolve everything except the root $ref first, then the root."""
    if not isinstance(sch, dict):
        return {}
    sch = copy.deepcopy(sch)
    if "$ref" in sch:
        R.conv.resolve_refs({k: v for k, v in sch.items() if k != "$ref"}, url)   # the same sub-objects
    return R.conv.resolve_refs(sch, url)


def schema_rule(R: _Rules, sch, name: str) -> str:
    """builder.add_schema in the fork: rule for an (already resolved) JSON schema."""
    return R.conv.visit(sch if isinstance(sch, dict) else {}, name)


def _including(R: _Rules, name: str, delim: str) -> str:
    """Everything up to and including the first occurrence of delim (gbnf_including_grammar, KMP instead of Aho-Corasick)."""
    L = len(delim)
    fail = [0] * (L + 1)
    for i in range(1, L):
        k = fail[i]
        while k and delim[i] != delim[k]:
            k = fail[k]
        fail[i + 1] = k + 1 if delim[i] == delim[k] else 0

    def nxt(q, c):
        while q and delim[q] != c:
            q = fail[q]
        return q + 1 if delim[q] == c else 0

    alphabet = sorted(set(delim))
    st = lambda q: name if q == 0 else f"{name}-{q:02d}"   # noqa: E731
    esc = lambda c: {"\n": "\\n", "\t": "\\t", "\r": "\\r", "-": "\\-", "]": "\\]", "[": "\\[", "\\": "\\\\"}.get(c, c)  # noqa: E731
    for q in range(L):
        parts, specific = [], []
        for c in alphabet:
            d = nxt(q, c)
            if d == L:
                parts.append(f"[{esc(c)}]")
                specific.append(c)
            elif d:
                parts.append(f"[{esc(c)}] {st(d)}")
                specific.append(c)
        parts.append("[^" + "".join(esc(c) for c in specific) + "] " + st(0))
        R.rules[st(q)] = " | ".join(parts)
    return name


def _excluding(R: _Rules, name: str, delims: list[str]) -> str:
    """Text that contains none of the delimiters (gbnf_excluding_grammar in the fork, Aho-Corasick over prefixes)."""
    prefixes = sorted({d[:i] for d in delims for i in range(len(d))}, key=lambda s: (len(s), s))
    ids = {s: i for i, s in enumerate(prefixes)}
    alphabet = sorted(set("".join(delims)))
    esc = lambda c: {"\n": "\\n", "\t": "\\t", "\r": "\\r", "-": "\\-", "]": "\\]", "[": "\\[", "\\": "\\\\"}.get(c, c)  # noqa: E731
    st = lambda i: name if i == 0 else f"{name}-{i:02d}"   # noqa: E731

    def nxt(s, c):
        x = s + c
        if any(x.endswith(d) for d in delims):
            return None                          # delimiter complete: forbidden
        while x and x not in ids:
            x = x[1:]
        return ids[x]
    for s, i in ids.items():
        parts, specific = ["|"], []
        for c in alphabet:
            d = nxt(s, c)
            specific.append(c)
            if d is not None and d != 0:
                parts.append(f"[{esc(c)}] {st(d)} |")
            elif d == 0:
                parts.append(f"[{esc(c)}] {st(0)} |")
        parts.append("[^" + "".join(esc(c) for c in specific) + "] " + st(0))
        R.rules[st(i)] = " ".join(parts)
    return name


class SchemaError(ValueError):
    """Schema cannot be converted (in the fork: "JSON schema conversion failed")."""


def format_gbnf(schema: dict | None = None, grammar: str | None = None) -> str:
    """Structured output like the fork (common_chat_params_init_qwen3_coder with json_schema, not lazy): generation
    prompt, optional thinking block (text without </think> and <tool_call>), then the JSON per schema. grammar: free GBNF
    (llama.cpp field `grammar`) unchanged."""
    if grammar is not None:
        return grammar
    R = _Rules()
    try:
        rf = schema_rule(R, resolve_refs(R, schema, "input"), "response-format")
    except (AssertionError, AttributeError, ValueError, KeyError, TypeError, RuntimeError, RecursionError) as e:
        raise SchemaError(str(e) or type(e).__name__) from e
    until = _excluding(R, "until-think", ["</think>", "<tool_call>"])
    R.rules["root"] = (lit("<|im_start|>assistant\n") + " (" + lit("<think>") + " space " + until + " ("
                       + lit("</think>") + " | ))? space " + rf)
    return R.text()


def fork_gbnf(tools: list[dict], parallel: bool = True, required: bool = False, escape: bool = False) -> str:
    """Grammar like common_chat_params_init_qwen3_coder (tools in OpenAI form). escape (TOB-59): escape hatch for a
    name in no tool list (_escape_rules) – without it the language stays exactly the fork's."""
    try:
        return _fork_gbnf(tools, parallel, required, escape)
    except (AssertionError, AttributeError, ValueError, KeyError, TypeError, RuntimeError, RecursionError) as e:
        raise SchemaError(str(e) or type(e).__name__) from e


def _cls(chars) -> str:
    """Characters for a GBNF class (each as a \\U escape, so that ]^-\\ mean nothing)."""
    return "".join(f"\\U{ord(c):08X}" for c in sorted(chars))


def _escape_rules(R: _Rules, names: list[str]) -> str:
    """TOB-59: escape hatch for a tool name in no list (previously the grammar bent it onto a listed one). Trie over the listed names: along a listed prefix the next character may
    follow the trie or leave it; off it (or at a prefix that is not a listed name) the name may
    end. After that only ">" or ">\\n" and end of output – the server reports "unknown tool" (Splitter.malformed).
    Listed names keep their argument grammar. Returns the rule from the name on (after "<function=")."""
    stop = {">", "\n"}
    end = R.add("tool-esc-end", lit(">") + " " + lit("\n") + "?")
    rest = R.add("tool-esc-rest", "[^" + _cls(stop) + "]* " + end)
    trie: list[dict] = [{"kids": {}, "listed": False}]
    for n in names:
        i = 0
        for c in n:
            if c not in trie[i]["kids"]:
                trie.append({"kids": {}, "listed": False})
                trie[i]["kids"][c] = len(trie) - 1
            i = trie[i]["kids"][c]
        trie[i]["listed"] = True
    rid = [f"tool-esc-{i}" for i in range(len(trie))]
    for i, node in enumerate(trie):
        kids = {c: j for c, j in node["kids"].items() if c not in stop}
        alts = [lit(c) + " " + rid[j] for c, j in kids.items()]
        alts.append("[^" + _cls(stop | set(kids)) + "] " + rest)
        if i and not node["listed"]:
            alts.append(end)
        R.rules[rid[i]] = " | ".join(alts)
    return rid[0]


def _fork_gbnf(tools: list[dict], parallel: bool, required: bool, escape: bool = False) -> str:
    R = _Rules()
    R.rules["root"] = "tool-call-root"
    arg_string = R.add("xml-arg-string", _including(R, "ac-str", PARAM_END))
    choices = []
    for t in tools:
        fn = t.get("function", t)
        name = fn["name"]
        sch = resolve_refs(R, fn.get("parameters") or fn.get("input_schema") or {}, f"tool-{name}")
        props = sch.get("properties") or {}
        need = set(sch.get("required") or [])
        req, opt = [], []
        for p, ps in props.items():
            if resolves_to_string(ps, R.conv._refs):
                val = arg_string
            else:
                val = schema_rule(R, ps, f"tool-{name}-arg-{p}-schema") + " " + lit(PARAM_END)
            r = R.add(f"tool-{name}-arg-{p}", "(" + lit("<parameter=") + " " + lit(p) + " " + lit(">\n") + ") " + val)
            (req if p in need else opt).append(r)
        # required parameters in any order (p.permute), then optional ones any number of times
        if len(req) > 1:
            perm = R.add(f"tool-{name}-args", " | ".join(" ".join(x) for x in permutations(req)))
        else:
            perm = req[0] if req else ""
        args = perm
        if opt:
            args = "(" + (perm + " " if perm else "") + "(" + " | ".join(opt) + ")*)"
        body = "(" + lit("<function=") + " " + lit(name) + " " + lit(">\n") + ") " + (args + " " if args else "") \
            + lit("</function>\n")
        choices.append(R.add(f"tool-{name}", body))
    ch = "(" + " | ".join(choices) + ")"
    R.rules["tool-call-first"] = lit("<tool_call>\n") + "? " + ch + ' "</tool_call>" space'
    R.rules["tool-call"] = lit("<tool_call>\n") + " " + ch + ' "</tool_call>" space'
    body = "tool-call-first tool-call*" if parallel else "tool-call-first"
    if escape:          # TOB-59: unknown name ends the output after the head (also after valid calls)
        esc = lit("<function=") + " " + _escape_rules(R, [(t.get("function", t))["name"] for t in tools])
        R.rules["tool-call-esc-first"] = lit("<tool_call>\n") + "? " + esc
        R.rules["tool-call-esc"] = lit("<tool_call>\n") + " " + esc
        body = ("tool-call-first tool-call* tool-call-esc? | tool-call-esc-first" if parallel
                else "tool-call-first | tool-call-esc-first")
    R.rules["tool-call-root"] = f"({body})" if required else f"({body})?"      # tool_choice required: at least 1 call
    return R.text()


# ------------------------------------------------------------------------------------------------ GBNF parser
class Grammar:
    """GBNF -> sequences. Element: ("c", neg, ((lo, hi), ...)) or ("r", rule). Repetitions are resolved into helper rules as in
    llama.cpp."""

    def __init__(self, text: str, root: str = "root"):
        self.rules: dict[str, list[tuple]] = {}
        self._aux = 0
        self.src = text
        self.i = 0
        self._parse()
        missing = {e[1] for alts in self.rules.values() for a in alts for e in a if e[0] == "r"} - set(self.rules)
        if missing:
            raise ValueError(f"GBNF: undefined rules {sorted(missing)}")
        self._left_recursion()
        self.root = root

    def _left_recursion(self):
        """Like llama_grammar_detect_left_recursion: a rule that (via empty prefixes) starts with itself again
        is not loaded by llama.cpp ("failed to parse grammar"), e.g. "a"** from a pattern."""
        empty: set[str] = set()
        changed = True
        while changed:
            changed = False
            for name, alts in self.rules.items():
                if name not in empty and any(all(e[0] == "r" and e[1] in empty for e in a) for a in alts):
                    empty.add(name)
                    changed = True
        state: dict[str, int] = {}

        def visit(name):
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise ValueError(f"GBNF: left recursion in rule {name}")
            state[name] = 1
            for a in self.rules[name]:
                for e in a:
                    if e[0] != "r":
                        break
                    visit(e[1])
                    if e[1] not in empty:
                        break
            state[name] = 2
        for name in list(self.rules):
            visit(name)

    # --- Lexer
    def _ws(self, nl: bool):
        s = self.src
        while self.i < len(s):
            c = s[self.i]
            if c in " \t" or (nl and c in "\r\n"):
                self.i += 1
            elif c == "#":
                while self.i < len(s) and s[self.i] not in "\r\n":
                    self.i += 1
            else:
                break

    def _name(self) -> str:
        m = re.compile(r"[a-zA-Z0-9-]+").match(self.src, self.i)
        if not m:
            raise ValueError(f"GBNF: expected a name at {self.src[self.i:self.i + 30]!r}")
        self.i = m.end()
        return m.group(0)

    def _char(self) -> int:
        s = self.src
        c = s[self.i]
        if c != "\\":
            self.i += 1
            return ord(c)
        e = s[self.i + 1]
        n = {"x": 2, "u": 4, "U": 8}.get(e)
        if n:
            v = int(s[self.i + 2: self.i + 2 + n], 16)
            self.i += 2 + n
            return v
        self.i += 2
        return ord({"n": "\n", "r": "\r", "t": "\t", "0": "\0"}.get(e, e))

    def _parse(self):
        self._ws(True)
        while self.i < len(self.src):
            name = self._name()
            self._ws(False)
            if not self.src.startswith("::=", self.i):
                raise ValueError(f"GBNF: expected ::= after {name}")
            self.i += 3
            self._ws(False)
            self.rules[name] = self._alts(nested=False)
            self._ws(True)

    def _alts(self, nested: bool) -> list[tuple]:
        alts = [self._seq(nested)]
        while self.i < len(self.src) and self.src[self.i] == "|":
            self.i += 1
            self._ws(True)
            alts.append(self._seq(nested))
        return alts

    def _new_rule(self, alts) -> str:
        self._aux += 1
        name = f"_aux{self._aux}"
        self.rules[name] = alts
        return name

    def _seq(self, nested: bool) -> tuple:
        out: list = []
        last_start = 0
        s = self.src
        while self.i < len(s):
            c = s[self.i]
            if c == '"':
                self.i += 1
                last_start = len(out)
                while s[self.i] != '"':
                    v = self._char()
                    out.append(("c", False, ((v, v),)))
                self.i += 1
            elif c == "[":
                self.i += 1
                neg = s[self.i] == "^"
                if neg:
                    self.i += 1
                rng = []
                while s[self.i] != "]":
                    lo = self._char()
                    hi = lo
                    if s[self.i] == "-" and s[self.i + 1] != "]":
                        self.i += 1
                        hi = self._char()
                    rng.append((lo, hi))
                self.i += 1
                last_start = len(out)
                out.append(("c", neg, tuple(rng)))
            elif c == ".":
                self.i += 1
                last_start = len(out)
                out.append(("c", False, ((0, 0x10FFFF),)))
            elif c == "(":
                self.i += 1
                self._ws(True)
                alts = self._alts(nested=True)
                if s[self.i] != ")":
                    raise ValueError("GBNF: expected )")
                self.i += 1
                last_start = len(out)
                out.append(("r", self._new_rule(alts)))
            elif re.match(r"[a-zA-Z0-9-]", c):
                last_start = len(out)
                out.append(("r", self._name()))
            elif c in "*+?{":
                lo, hi = {"*": (0, None), "+": (1, None), "?": (0, 1)}.get(c, (None, None))
                self.i += 1
                if c == "{":
                    m = re.compile(r"(\d+)(,(\d*))?\}").match(s, self.i)
                    lo = int(m.group(1))
                    hi = lo if m.group(2) is None else (int(m.group(3)) if m.group(3) else None)
                    self.i = m.end()
                item = tuple(out[last_start:])
                del out[last_start:]
                out.extend(self._repeat(item, lo, hi))
            else:
                break
            self._ws(nested)
        return tuple(out)

    def _repeat(self, item: tuple, lo: int, hi: int | None) -> list:
        res = list(item) * lo
        if hi is None:                                     # x* -> r ::= x r |
            name = f"_aux{self._aux + 1}"
            self._aux += 1
            self.rules[name] = [item + (("r", name),), ()]
            res.append(("r", name))
        elif hi > lo:                                      # x? x? ... nested like llama.cpp
            tail: tuple = ()
            for _ in range(hi - lo):
                tail = (("r", self._new_rule([item + tail, ()])),)
            res.extend(tail)
        return res


# ------------------------------------------------------------------------------------------------ stack automaton
class Matcher:
    """Set of stacks like llama-grammar.cpp. Stack = tuple of (rule, alternative, position), top = last
    element; empty stack = grammar complete."""

    def __init__(self, g: Grammar):
        self.g = g
        self.alts = {k: tuple(v) for k, v in g.rules.items()}

    @lru_cache(maxsize=200000)
    def expand(self, stack: tuple) -> frozenset:
        if not stack:
            return frozenset([()])
        rule, alt, i = stack[-1]
        seq = self.alts[rule][alt]
        if i >= len(seq):
            return self.expand(stack[:-1])
        el = seq[i]
        if el[0] == "c":
            return frozenset([stack])
        rest = stack[:-1] + ((rule, alt, i + 1),) if i + 1 < len(seq) else stack[:-1]
        out: set = set()
        for a in range(len(self.alts[el[1]])):
            out |= self.expand(rest + ((el[1], a, 0),))
        return frozenset(out)

    def start(self, rule: str | None = None) -> frozenset:
        r = rule or self.g.root
        out: set = set()
        for a in range(len(self.alts[r])):
            out |= self.expand(((r, a, 0),))
        return frozenset(out)

    def _el(self, stack):
        rule, alt, i = stack[-1]
        return self.alts[rule][alt][i]

    @staticmethod
    def _match(el, cp: int) -> bool:
        hit = any(lo <= cp <= hi for lo, hi in el[2])
        return hit != el[1]

    @lru_cache(maxsize=400000)
    def step(self, stacks: frozenset, cp: int) -> frozenset:
        out: set = set()
        for st in stacks:
            if not st:
                continue
            el = self._el(st)
            if self._match(el, cp):
                rule, alt, i = st[-1]
                nxt = st[:-1] + ((rule, alt, i + 1),)
                out |= self.expand(nxt)
        return frozenset(out)

    def partial_ok(self, stacks: frozenset, lo: int) -> bool:
        """Incomplete UTF-8 at the token end: does any code point >= lo fit (roughly like match_partial_char)."""
        for st in stacks:
            if st:
                el = self._el(st)
                if el[1] or any(hi >= lo for _, hi in el[2]):
                    return True
        return False


def _utf8(buf: bytes):
    """-> (code points, incomplete rest)."""
    cps, i = [], 0
    while i < len(buf):
        b = buf[i]
        n = 1 if b < 0x80 else 2 if b >> 5 == 6 else 3 if b >> 4 == 14 else 4 if b >> 3 == 30 else 1
        if i + n > len(buf):
            return cps, buf[i:]
        try:
            cps.append(ord(buf[i:i + n].decode("utf-8")))
        except UnicodeDecodeError:
            cps.append(0xFFFD)
            n = 1
        i += n
    return cps, b""


# ------------------------------------------------------------------------------------------------ token level
def _byte_decoder() -> dict:
    """GPT-2 bytes_to_unicode, reversed (Qwen's byte-BPE alphabet)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenBytes:
    """Bytes per token like llama.cpp's token_to_piece(special=true); first byte per token as array (preselection)."""

    def __init__(self, hf_tok):
        t = hf_tok.t if hasattr(hf_tok, "t") else hf_tok
        n = t.get_vocab_size(with_added_tokens=True)
        dec = _byte_decoder()
        added = {}
        try:
            added = {i: a.content for i, a in t.get_added_tokens_decoder().items()}
        except AttributeError:
            pass
        self.b: list[bytes] = []
        for i in range(n):
            if i in added:
                self.b.append(added[i].encode("utf-8"))
                continue
            s = t.id_to_token(i)
            if s is None:
                self.b.append(b"")
            elif all(ch in dec for ch in s):
                self.b.append(bytes(dec[ch] for ch in s))
            else:
                self.b.append(s.encode("utf-8"))
        self.first = np.array([x[0] if x else 0 for x in self.b], dtype=np.uint8)
        self.n = n
        self._ix = None

    def index(self):
        """(bytes -> IDs, sorted bytes, IDs in this order) for ToolFormat.single; empty bytes count as allowed
        and are therefore never part of a point mass (the Qwen vocabulary has none)."""
        if self._ix is None:
            by: dict = {}
            for i, x in enumerate(self.b):
                by.setdefault(x, []).append(i)
            order = sorted(range(len(self.b)), key=lambda i: self.b[i])
            self._ix = (by, [self.b[i] for i in order], order)
            if b"" in by:
                self._ix = ({}, [], [])          # an empty token fits everywhere: never exactly one allowed
                self._empty = True
        return self._ix


_TB: dict = {}


def token_bytes(hf_tok) -> TokenBytes:
    k = id(hf_tok)
    if k not in _TB:
        _TB[k] = TokenBytes(hf_tok)
    return _TB[k]


@lru_cache(maxsize=64)
def _matcher(text: str) -> Matcher:
    return Matcher(Grammar(text))


STRICT_REQUIRED = True          # L49: tool_choice required/named strict (server --tool-choice-lax: as before)


class ToolFormat:
    """Lazy grammar per request (llama_grammar_accept_impl): free before the trigger; trigger = token ID
    (<tool_call>) or word (<function=NAME>) in the text generated so far; after that only allowed tokens, EOG only with
    a complete grammar."""

    def __init__(self, tools: list[dict], tb: TokenBytes, trigger_ids: set[int], eog: set[int], parallel: bool = True,
                 force: list | None = None, think: tuple | None = None, thinking: bool = False):
        """force (tool_choice): None = auto; [] = some call required; [name] = only this tool, required.
        Required means as in the fork: text before it is free, but no end (EOG) before the first call.
        think = (token <think>, {tokens that end the thinking: </think>, <tool_call>}), thinking = the prompt ends in the
        thinking block. As in the fork (reasoning budget, lazy grammar): while thinking, the grammar sees nothing and
        constrains nothing; the token that ends the thinking is fed to it afterwards (<tool_call> triggers)."""
        self.think_start, self.think_end = (think[0], set(think[1])) if think else (None, set())
        self.reasoning = bool(thinking and think)
        if force:
            tools = [t for t in tools if (t.get("function", t)).get("name") in force] or tools
        self.required = force is not None
        # L49: required now means as in the OpenAI API: after the thinking only whitespace, then the call (previously:
        # text before it free, just no end -> small models wrote text without a call up to max_tokens)
        self.strict = self.required and STRICT_REQUIRED
        # TOB-59: escape hatch for names in no list, except with a named tool_choice (there only that tool counts)
        self.text = fork_gbnf(tools, parallel, self.required, escape=not force)
        self.m = _matcher(self.text)
        self.tb, self.eog = tb, set(eog)
        self.trigger_ids = set(trigger_ids)
        self.words = [("<function=" + (t.get("function", t))["name"] + ">").encode() for t in tools]
        self.active = False
        self.buf = b""                  # text since the start, as long as the trigger is missing
        self.stacks = frozenset()
        self.partial = b""
        self.n_fixed = 0                # how often the unconstrained argmax was replaced
        self.fixes: list = []           # (unconstrained, allowed) per replacement (evaluation)
        self.acc = b""                  # accepted output (bytes), for drafts from the grammar

    @classmethod
    def eager(cls, text: str, tb: TokenBytes, eog: set[int], prefill: str = ""):
        """Non-lazy grammar (structured output, free GBNF): applies from the first token on; prefill = generation
        prompt that the fork gives the grammar up front (common_grammar_needs_prefill), empty for free GBNF."""
        self = cls.__new__(cls)
        self.think_start, self.think_end, self.reasoning = None, set(), False
        self.required, self.text = False, text
        self.m = _matcher(text)
        self.tb, self.eog, self.trigger_ids, self.words = tb, set(eog), set(), []
        self.active, self.buf, self.partial = True, b"", b""
        self.stacks = self.m.start()
        self.n_fixed, self.fixes, self.acc = 0, [], b""
        if prefill:
            r = self._feed((self.stacks, b""), prefill.encode("utf-8"))
            if r is None:
                raise ValueError("grammar does not match the generation prompt")
            self.stacks, self.partial = r
        return self

    def forced_text(self, state=None, limit: int = 64) -> str:
        """Text that the grammar fixes unambiguously from the state (all stacks allow exactly one character, none
        is finished): grammar jump, verified as one draft."""
        st = self.state() if state is None else state
        if not st[0] or st[4] or st[3]:
            return ""
        stacks, out = st[2], []
        while len(out) < limit and stacks and () not in stacks:
            cps = set()
            for s in stacks:
                el = self.m._el(s)
                if el[1] or len(el[2]) != 1 or el[2][0][0] != el[2][0][1]:
                    return "".join(out)
                cps.add(el[2][0][0])
            if len(cps) != 1:
                break
            cp = cps.pop()
            out.append(chr(cp))
            stacks = self.m.step(stacks, cp)
        return "".join(out)

    # --- state (immutable: (active, buf, stacks, partial, reasoning))
    def state(self):
        return (self.active, self.buf, self.stacks, self.partial, self.reasoning)

    def _feed(self, st, data: bytes):
        stacks, part = st
        cps, rest = _utf8(part + data)
        for cp in cps:
            stacks = self.m.step(stacks, cp)
            if not stacks:
                return None
        if rest and not self.m.partial_ok(stacks, 0x80):
            return None
        return stacks, rest

    def _next(self, state, tok: int):
        """State after tok or None (not allowed)."""
        if state[4]:                                   # while thinking: grammar is idle
            if tok in self.think_end:
                if getattr(self, "strict", False) and tok not in self.trigger_ids:
                    return state[:4] + (False,)        # L49 strict: </think> only ends the thinking
                return self._next(state[:4] + (False,), tok)
            return state
        nx = self._next4(state[:4], tok)
        if nx is None:
            return None
        return nx + (tok == self.think_start and not nx[0],)

    def _next4(self, state, tok: int):
        active, buf, stacks, part = state
        if not 0 <= tok < len(self.tb.b):
            return None if active else state
        if not active:
            if tok in self.trigger_ids:
                r = self._feed((self.m.start(), b""), self.tb.b[tok])
                return None if r is None else (True, b"", r[0], r[1])
            if getattr(self, "strict", False):          # at most 4 whitespace bytes before the call
                b = self.tb.b[tok]
                return (False, buf + b, stacks, part) if b and not b.strip() and len(buf) + len(b) <= 4 else None
            if self.required and tok in self.eog:
                return None                  # tool_choice required: do not end without a call
            nb = buf + self.tb.b[tok]
            hits = [nb.find(w) for w in self.words]
            hits = [h for h in hits if h >= 0]
            if not hits:
                return (False, nb, stacks, part)
            r = self._feed((self.m.start(), b""), nb[min(hits):])
            return None if r is None else (True, b"", r[0], r[1])
        if tok in self.eog:
            return state if () in stacks else None
        r = self._feed((stacks, part), self.tb.b[tok])
        return None if r is None else (True, b"", r[0], r[1])

    def allowed(self, tok: int, state=None) -> bool:
        st = self.state() if state is None else state
        if st[4] or (not st[0] and not getattr(self, "strict", False) and not (self.required and tok in self.eog)):
            return True                 # free before the trigger (including the trigger itself)
        return self._next(st, tok) is not None

    def accept(self, tok: int):
        nx = self._next(self.state(), tok)
        if nx is None:
            raise ValueError(f"token {tok} does not match the format grammar")
        self.active, self.buf, self.stacks, self.partial, self.reasoning = nx
        self.acc += self.tb.b[tok] if 0 <= tok < len(self.tb.b) else b""

    def single(self):
        """The only allowed token in the current state, else None (L38, 6a: then the distribution is a point mass
        and nothing is sampled). Allowed are exactly the tokens whose bytes are a prefix of the forced text,
        and the longer ones that start with it and still fit afterwards; cached per state."""
        st = self.state()
        if not st[0] or st[4] or st[3]:
            return None
        cache = self.__dict__.setdefault("_single_cache", {})
        key = st[2]
        if key not in cache:
            cache[key] = self._single(st)
        return cache[key]

    def _single(self, st):
        fb = self.forced_text(st).encode("utf-8")
        if not fb:
            return None
        ix = self.tb.index()
        found = []
        for n in range(1, len(fb) + 1):                  # prefixes of the forced text
            for t in ix[0].get(fb[:n], ()):
                if self._next(st, t) is not None:
                    found.append(t)
                    if len(found) > 1:
                        return None
        keys, ids = ix[1], ix[2]
        i = bisect.bisect_right(keys, fb)                # longer tokens that start with the forced text
        while i < len(keys) and keys[i].startswith(fb):
            if self._next(st, ids[i]) is not None:
                found.append(ids[i])
                if len(found) > 1:
                    return None
            i += 1
        return found[0] if len(found) == 1 else None

    def state_after(self, text: str):
        """State after the text accepted so far + rest of text (for drafts), None if it doesn't fit."""
        b = text.encode("utf-8")
        if not b.startswith(self.acc):
            return None
        st = self.state()
        if not st[0] or st[4]:
            return None
        r = self._feed((st[2], st[3]), b[len(self.acc):])
        return None if r is None else (True, b"", r[0], r[1], False)

    def prefix(self, toks: list[int]) -> int:
        """How many tokens of toks are allowed one after another (cut the draft)."""
        st = self.state()
        for i, t in enumerate(toks):
            nx = self._next(st, t)
            if nx is None:
                return i
            st = nx
        return len(toks)

    def best(self, logits) -> int:
        """Argmax among the allowed tokens (like the fork after masking)."""
        st = self.state()
        lg = np.asarray(logits, dtype=np.float32)[: self.tb.n]
        if not st[0] and getattr(self, "strict", False):     # L49 strict: trigger or some whitespace
            ws = self.__dict__.get("_ws")
            if ws is None:
                ws = self._ws = np.array(sorted({i for i, x in enumerate(self.tb.b) if x and not x.strip()} |
                                                set(self.trigger_ids)), dtype=np.int64)
            ws = ws[ws < len(lg)]
            for t in ws[np.argsort(-lg[ws], kind="stable")]:
                if self._next(st, int(t)) is not None:
                    return int(t)
        if not st[0]:                   # before the trigger (tool_choice required): everything except the end
            for tk in np.argsort(-lg, kind="stable")[: len(self.eog) + 1]:
                if int(tk) not in self.eog:
                    return int(tk)
        ok_first = np.zeros(256, dtype=bool)          # preselection by the first byte, checked exactly afterwards
        if st[3]:
            ok_first[:] = True                          # incomplete UTF-8 open: continuation bytes possible
        for stack in st[2]:
            if not stack:
                continue
            el = self.m._el(stack)
            for b in range(0x80):
                ok_first[b] |= Matcher._match(el, b)
            if el[1] or any(hi >= 0x80 for _, hi in el[2]):
                ok_first[0xC0:] = True
        cand = np.nonzero(ok_first[self.tb.first])[0]
        extra = [t for t in self.eog if t < len(lg)] if () in st[2] else []
        idx = np.concatenate([cand, np.array(extra, dtype=cand.dtype)]) if extra else cand
        order = idx[np.argsort(-lg[idx], kind="stable")]
        for t in order:
            if self._next(st, int(t)) is not None:
                return int(t)
        raise ValueError("format grammar: no allowed token")

    def ranked(self, x, want: int) -> list:
        """The first `want` allowed tokens by x descending (tie: smaller ID first) – the same as a
        loop over the whole sorted row, but active only over the preselection by the first byte (like best();
        sampling with mask, L38)."""
        st = self.state()
        x = np.asarray(x)
        if st[4] or not st[0]:          # free (thinking) or before the trigger: everything except possibly the end
            out = []
            for tk in np.lexsort((np.arange(x.size), -x))[: want + len(self.eog)]:
                if self.allowed(int(tk), st):
                    out.append(int(tk))
                    if len(out) >= want:
                        break
            return out
        lg = x[: self.tb.n]             # active: tokens without a bytes entry are never allowed
        ok_first = np.zeros(256, dtype=bool)
        if st[3]:
            ok_first[:] = True
        for stack in st[2]:
            if not stack:
                continue
            el = self.m._el(stack)
            for b in range(0x80):
                ok_first[b] |= Matcher._match(el, b)
            if el[1] or any(hi >= 0x80 for _, hi in el[2]):
                ok_first[0xC0:] = True
        idx = np.nonzero(ok_first[self.tb.first])[0]
        if () in st[2]:
            idx = np.union1d(idx, np.array([t for t in self.eog if t < lg.size], dtype=idx.dtype))
        out = []
        for t in idx[np.lexsort((idx, -lg[idx]))]:
            if self._next(st, int(t)) is not None:
                out.append(int(t))
                if len(out) >= want:
                    break
        return out


class FormatVerifier:
    """Engine with format grammar for decoder.generate: cut drafts to the allowed beginning, check the bonus token
    (unconstrained argmax) and, if it doesn't fit, replace it with the allowed argmax from the logits of the same row
    (engine.logits(row), engine.resample). Output = greedy + grammar without speculation.
    mask (L38 item 5, --draft-mask): mask per draft position. If the unconstrained argmax of a verify row is forbidden,
    the allowed argmax of that row counts; if it equals the draft, the draft token is accepted and verification
    continues (without mask the round ended there). Reference per position stays "greedy + grammar" -> bit-identical.
    Needs engine.verify_argmax(draft) (verify without committing) and engine.commit(m, tok)."""

    def __init__(self, v, fmt: ToolFormat, mask: bool = False):
        self.v, self.fmt = v, fmt
        self.mask = bool(mask) and callable(getattr(v, "verify_argmax", None))
        self.n_mask_acc = 0             # draft tokens accepted only thanks to the mask

    def __getattr__(self, k):
        return getattr(self.v, k)

    def _fix(self, tok: int, row: int | None) -> int:
        if self.fmt.allowed(tok):
            self.fmt.accept(tok)
            return tok
        lg = self.v.logits(row) if row is not None else self.v.logits()
        t2 = self.fmt.best(lg)
        self.v.resample(t2)
        self.fmt.accept(t2)
        self.fmt.n_fixed += 1
        self.fmt.fixes.append((tok, t2))
        return t2

    def start(self, prompt_ids):
        first = self.v.start(prompt_ids)
        return self._fix(first, None)

    def verify(self, draft):
        draft = list(draft)[: self.fmt.prefix(list(draft))]
        e = next((i for i, t in enumerate(draft) if t in self.fmt.eog), None)
        if e is not None:                 # check nothing more after the end
            draft = draft[: e + 1]
        if self.mask:
            return self._verify_masked(draft)
        m, nxt = self.v.verify(draft)
        for t in draft[:m]:
            self.fmt.accept(t)
        if m and draft[m - 1] in self.fmt.eog:
            return m, nxt                 # output ends at the stop token, the following token doesn't count
        return m, self._fix(nxt, m)

    def _verify_masked(self, draft):
        f = self.fmt
        vout = self.v.verify_argmax(draft)
        draft = draft[: len(vout) - 1]
        m = 0
        while True:
            t = vout[m]
            masked = not f.allowed(t)
            if masked:                    # allowed argmax of this row (like _fix, only before committing)
                t = f.best(self.v.logits(m))
                f.n_fixed += 1
                f.fixes.append((vout[m], t))
            if m < len(draft) and t == draft[m]:
                f.accept(t)
                self.n_mask_acc += masked
                m += 1
                if t in f.eog:            # end in the draft: following token doesn't count
                    self.v.commit(m, vout[m])
                    return m, vout[m]
                continue
            break
        f.accept(t)
        self.v.commit(m, t)
        return m, t


class GrammarDrafts:
    """Grammar jump from the format grammar itself (structured output): interface of a draft proposer
    (forced/field/field_kind) for the proposer. What the grammar fixes unambiguously (keys, brackets, prefixes
    of enum values) comes as a draft and is verified in the same exact verify pass as any draft."""

    def __init__(self, fmt: ToolFormat):
        self.fmt = fmt
        self.paths = []

    def forced(self, text: str) -> str:
        st = self.fmt.state_after(text)
        return self.fmt.forced_text(st) if st is not None else ""

    def field(self, text: str) -> str:
        return "json"

    def field_kind(self, text: str) -> str:
        return "scaffold"

    def region(self, text: str) -> str:
        return "scaffold"


class _NoDrafts:
    """Measurement (X-Tobu-Jump: off): no drafts from the grammar (lookup/MTP stay)."""
    paths: list = []

    def forced(self, text):
        return ""

    def field(self, text):
        return "free"

    def field_kind(self, text):
        return "text"

    def region(self, text):
        return "free"


NO_DRAFTS = _NoDrafts()
