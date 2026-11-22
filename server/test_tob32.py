"""TOB-32: thinking budget (thinking_budget) - no GPU, no model, no network.

    python test_tob32.py          (venv with numpy, tokenizers)

Test engine "Thinker": thinks without limit (running text without </think>, no repetition) or in the variant
"short" ends with </think> by itself after fixed tokens; after </think> the answer follows (text or JSON). Byte tokenizer with the
special tokens of the Qwen template (test_tob2.make_tokenizer).

  1. Wrapper alone (decoder.generate): budget applies exactly (thinking tokens == budget, then "\\n</think>\\n\\n" + answer);
     with random engine drafts the same tokens as without draft; if thinking ends with a newline, only "</think>\\n\\n".
  2. HTTP OpenAI: thinking_budget top-level and in chat_template_kwargs -> reasoning_content ends, content = answer,
     usage.completion_tokens_details.reasoning_tokens == budget, tobu.thinking_budget.hit, log line.
  3. No budget = previous behaviour: thinking endlessly -> finish length, no answer; a model that ends by itself,
     with a large budget -> the same tokens as without budget (hit no).
  4. --default-thinking-budget applies when none is given; thinking_budget 0 turns it off for the request.
  5. json_schema (response_format): after the forced end valid JSON per schema, grammar active.
  6. Anthropic thinking.budget_tokens takes effect (no longer in tobu.ignored), stream == non-stream.
  7. Sampling (temperature > 0): budget applies as well; invalid thinking_budget -> 400.
"""

from __future__ import annotations

import contextlib
import io
import json
import urllib.request
import random
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_tob2 import ANSWERS, OA_TOOLS, make_tokenizer, post, sse  # noqa: E402

FAILS: list[str] = []
THINK = "".join(f"Schritt {i}: pruefe Annahme {i * 7 % 13}. " for i in range(1, 2000))   # without </think>
ANSWER = "Die Antwort ist 42."
JSON_ANSWER = '{"answer": 42, "ok": true}'
SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}, "ok": {"type": "boolean"}},
          "required": ["answer", "ok"], "additionalProperties": False}


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}{(' – ' + str(info)[:600]) if info and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------- test engine
class Thinker:
    """'Model' as a function of the output so far: before </think> the thinking text (or with natural=k after k chars
    "\\n</think>\\n\\n" by itself), then "\\n\\n" + answer + <|im_end|>. Checks drafts like an oracle; resample/logits
    like the real engines (replaced token becomes the input of the next step)."""

    def __init__(self, tok, answer: str = ANSWER, natural: int | None = None):
        self.tok, self.name, self.max_len = tok, "denker", 1 << 20
        self.te = tok.t.token_to_id("</think>")
        self.eos = tok.t.token_to_id("<|im_end|>")
        think = THINK if natural is None else THINK[:natural].rstrip() + "\n</think>"
        self.think = tok.encode(think)
        self.answer = tok.encode("\n\n" + answer) + [self.eos]
        self.answer_key = answer

    def f(self, out: list[int]) -> int:
        if self.te in out:
            j = len(out) - out.index(self.te) - 1
            return self.answer[j] if j < len(self.answer) else self.eos
        return self.think[len(out)] if len(out) < len(self.think) else self.eos

    def make(self, prompt_ids):
        return _ThinkerV(self)


class _ThinkerV:
    def __init__(self, m: Thinker):
        self.m, self.out, self.reuse = m, [], 0
        self.n_resample = 0

    def start(self, prompt_ids):
        self.out = [self.m.f([])]
        return self.out[-1]

    def verify(self, draft):
        n = 0
        while n < len(draft) and draft[n] == self.m.f(self.out):
            self.out.append(draft[n])
            n += 1
        self.out.append(self.m.f(self.out))
        return n, self.out[-1]

    def resample(self, tok):
        self.out[-1] = tok
        self.n_resample += 1

    def logits(self, row=None):
        import numpy as np

        x = np.zeros(self.m.tok.t.get_vocab_size(), np.float32)
        x[self.out[-1]] = 50.0          # prediction of the last step (practically a point mass, also when drawing)
        return x


# ------------------------------------------------------------------------------------------------- 1: wrapper alone
def test_wrapper(tok):
    from decoder import generate
    import think_budget

    te, tc = tok.t.token_to_id("</think>"), tok.t.token_to_id("<tool_call>")
    close, close_nl = tok.encode("\n</think>\n\n"), tok.encode("</think>\n\n")
    nl = lambda t: tok.decode([t]).endswith("\n")   # noqa: E731
    m = Thinker(tok)

    def rand_drafts(inner, seed):
        """Engine drafts (self_draft) on the inner verifier: randomly correct/wrong, partly the closing chunk."""
        r = random.Random(seed)

        def self_draft(field="free", kind="text"):
            x = r.random()
            if x < 0.4:                # correct draft (oracle), also beyond the budget
                o, ids = list(inner.out), []
                for _ in range(r.randint(1, 12)):
                    ids.append(m.f(o))
                    o.append(ids[-1])
            elif x < 0.6:
                ids = list(close) + tok.encode("Die")
            elif x < 0.8:
                ids = [r.randint(300, 6000) for _ in range(r.randint(1, 6))]
            else:
                ids = []
            return ids
        inner.self_draft = self_draft
        return inner

    for budget in (1, 7, 50, 300):
        ref = None
        for seed in (None, 1, 2, 3):
            inner = m.make([]) if seed is None else rand_drafts(m.make([]), seed)
            v = think_budget.ThinkBudget(inner, budget, {te, tc}, close, close_nl, nl)
            res = generate(v, [1], n=2000, stop={m.eos}, use_self=seed is not None)
            toks = res.tokens
            if ref is None:
                ref = toks
                e = toks.index(te) if te in toks else -1
                txt = tok.decode(toks)
                check(f"1 Budget {budget}: Denken endet nach genau {budget} Tokens + \\n</think>\\n\\n, dann Antwort",
                      e == budget + 1 and v.hit and txt.endswith("</think>\n\n" + ANSWER + "<|im_end|>")
                      and toks[budget] == tok.encode("\n")[0], (e, txt[-80:]))
            else:
                check(f"1 Budget {budget}, Entwürfe (Seed {seed}): dieselben Tokens wie ohne Entwurf", toks == ref,
                      tok.decode(toks)[-120:])
    # thinking ends at the budget with a newline -> only "</think>\n\n"
    i = THINK.index(". ") + 2           # text up to after ". " -> we build a variant with "\n" at position k
    mm = Thinker(tok)
    mm.think = tok.encode(THINK[:i - 1] + "\n" + THINK[i:])
    k = len(tok.encode(THINK[:i - 1] + "\n"))
    v = think_budget.ThinkBudget(mm.make([]), k, {te, tc}, close, close_nl, nl)
    toks = generate(v, [1], n=2000, stop={mm.eos}, use_self=False).tokens
    check("1 Denken endet beim Budget mit Zeilenumbruch: nur </think>\\n\\n eingefügt (kein doppeltes \\n)",
          toks[k] == te and tok.decode(toks[:k]).endswith("\n") and not tok.decode(toks[:k]).endswith("\n\n"),
          tok.decode(toks[:k + 3]))
    # model ends by itself before the budget: wrapper changes nothing
    mn = Thinker(tok, natural=120)
    ref = generate(mn.make([]), [1], n=2000, stop={mn.eos}, use_self=False).tokens
    v = think_budget.ThinkBudget(mn.make([]), 500, {te, tc}, close, close_nl, nl)
    toks = generate(v, [1], n=2000, stop={mn.eos}, use_self=False).tokens
    check("1 Modell endet selbst vor dem Budget: Ausgabe unverändert, hit nein", toks == ref and not v.hit
          and v.n_replaced == 0)


# ------------------------------------------------------------------------------------------------- HTTP
def serve(tokfile: Path, answers: str, engine, *extra: str):
    import server

    a = server.build_parser().parse_args(["--engine", "keyed:" + answers, "--tokenizer", str(tokfile), "--port", "0",
                                          "--no-warmup", "--max-len", "32768", *extra])
    B = server.Backend(a)
    B.eng = engine(B.tok)
    B.warm.set()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(B))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    httpd.B = B                                  # tests: server state (pending etc.)
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def oa(extra=None, **kw):
    return {"model": "tobu", "temperature": 0, "max_tokens": 1500,
            "messages": [{"role": "user", "content": "Wie viel ist sechs mal sieben?"}], **(extra or {}), **kw}


def chat(base, body):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        s, j = post(base + "/v1/chat/completions", body)
    return s, j, out.getvalue()


def test_http(tokfile: Path, answers: str):
    import api

    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok))
    try:
        for label, body in (("top-level", oa(thinking_budget=64)),
                            ("chat_template_kwargs", oa(chat_template_kwargs={"thinking_budget": 64})),
                            ("max_thinking_tokens", oa(max_thinking_tokens=64))):
            s, j, log = chat(base, body)
            msg = j["choices"][0]["message"] if s == 200 else {}
            u = j.get("usage", {}) if s == 200 else {}
            check(f"2 OpenAI {label}: Denken beendet, Antwort da, finish stop",
                  s == 200 and msg.get("content") == ANSWER and (msg.get("reasoning_content") or "") and
                  j["choices"][0]["finish_reason"] == "stop", (s, j))
            check(f"2 OpenAI {label}: usage.completion_tokens_details.reasoning_tokens == 64 (+1 Zeilenumbruch)",
                  u.get("completion_tokens_details", {}).get("reasoning_tokens") == 65, u)
            tb = (j.get("tobu") or {}).get("thinking_budget") or {}
            check(f"2 OpenAI {label}: tobu.thinking_budget hit, timings, Log-Zeile 'erreicht ja'",
                  tb.get("hit") is True and tb.get("budget") == 64 and j["timings"].get("thinking_budget_hit") is True
                  and "thinking budget 64: reached yes" in log, (tb, log))
        # 3. without budget: previous behaviour (thinking endlessly -> length, no answer)
        s, j, log = chat(base, oa(max_tokens=300))
        msg = j["choices"][0]["message"]
        check("3 ohne Budget: finish length, nur Denken, keine Antwort (wie bisher)",
              s == 200 and j["choices"][0]["finish_reason"] == "length" and not msg.get("content")
              and "thinking_budget" not in j["tobu"] and "thinking budget" not in log
              and j["usage"]["completion_tokens_details"]["reasoning_tokens"] == 300, j)
        s, j, _ = chat(base, oa(max_tokens=300, thinking_budget=0))
        check("3 thinking_budget 0: aus", s == 200 and j["choices"][0]["finish_reason"] == "length", j)
        s, j, _ = chat(base, oa(chat_template_kwargs={"enable_thinking": False}, thinking_budget=10))
        check("3 Denken aus: Budget wirkungslos, reasoning_tokens 0",
              s == 200 and "thinking_budget" not in j["tobu"]
              and j["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0, j)
        # 7. errors, sampling
        for bad in (-1, "4096", 1.5, True):
            s, j, _ = chat(base, oa(thinking_budget=bad))
            check(f"7 thinking_budget {bad!r} -> 400", s == 400 and "thinking_budget" in json.dumps(j), (s, j))
        s, j, _ = chat(base, oa(temperature=0.7, seed=3, thinking_budget=40))
        check("7 Sampling: Budget greift, Antwort da",
              s == 200 and j["choices"][0]["message"].get("content") == ANSWER
              and j["tobu"]["thinking_budget"]["hit"] and j["tobu"]["sampling"] != "greedy", j)
        # 6. Anthropic budget_tokens
        an = {"model": "tobu", "max_tokens": 1500, "temperature": 0, "thinking": {"type": "enabled", "budget_tokens": 50},
              "messages": [{"role": "user", "content": "Wie viel ist sechs mal sieben?"}]}
        AV = {"anthropic-version": "2023-06-01"}
        with contextlib.redirect_stdout(io.StringIO()):
            s, j = post(base + "/v1/messages", an, AV)
            s2, raw = post(base + "/v1/messages", {**an, "stream": True}, AV, raw=True)
        bl = [b["type"] for b in j.get("content", [])] if s == 200 else []
        check("6 Anthropic budget_tokens: thinking-Block, dann Text = Antwort, end_turn, nicht ignoriert",
              s == 200 and bl == ["thinking", "text"] and j["content"][1]["text"] == ANSWER
              and j["stop_reason"] == "end_turn" and "thinking.budget_tokens" not in j["tobu"].get("ignored", [])
              and j["tobu"]["thinking_budget"]["budget"] == 50, j)
        txt = "".join(d["delta"].get("text", "") for e, d in sse(raw) if e == "content_block_delta")
        check("6 Anthropic Stream: dieselbe Antwort", s2 == 200 and txt == ANSWER, raw[-400:])
        # 9. metrics: usage.tobu (OpenAI, Anthropic, stream) and GET /tobu/metrics, without contents
        keys = {"request_id", "client_ip", "harness", "ttft_ms", "prefill_tps", "decode_tps", "thinking_tokens",
                "mtp_accept", "concurrency", "cache_hit_tokens"}
        s, j, _ = chat(base, oa(thinking_budget=64))
        ut = (j.get("usage") or {}).get("tobu") or {}
        check("9 usage.tobu mit allen Feldern (OpenAI)", s == 200 and keys <= set(ut) and ut["thinking_tokens"] == 65, ut)
        mu = (j.get("usage") or {})
        an_u = (json.loads(json.dumps(j_an := post(base + "/v1/messages", an, AV)[1])).get("usage") or {}).get("tobu") or {}
        check("9 usage.tobu auch bei Anthropic", keys <= set(an_u), an_u)
        with contextlib.redirect_stdout(io.StringIO()):
            s3, raw3 = post(base + "/v1/chat/completions", {**oa(thinking_budget=64), "stream": True,
                                                            "stream_options": {"include_usage": True}}, raw=True)
        txt3 = raw3.decode() if isinstance(raw3, bytes) else raw3
        evs = [json.loads(ln[5:]) for ln in txt3.splitlines() if ln.startswith("data:") and "[DONE]" not in ln]
        last_u = [d.get("usage") for d in evs if isinstance(d, dict) and d.get("usage")]
        check("9 usage.tobu im letzten Strom-Stück", s3 == 200 and last_u and keys <= set(last_u[-1].get("tobu") or {}),
              last_u[-1:] if last_u else raw3[-300:])
        r = urllib.request.urlopen(base + "/tobu/metrics", timeout=10)
        met = json.loads(r.read())
        rids = [x["request_id"] for x in met["requests"]]
        check("9 /tobu/metrics: live-Felder, letzte Anfragen mit request_id, keine Inhalte",
              {"slots_busy", "slots_total", "active_requests", "tok_s_now"} <= set(met["live"]) and
              ut["request_id"] in rids and ANSWER not in json.dumps(met) and "Wie viel" not in json.dumps(met), met)
        since = met["requests"][-1]["t"]
        met2 = json.loads(urllib.request.urlopen(base + f"/tobu/metrics?since={since}", timeout=10).read())
        check("9 /tobu/metrics?since= filtert", met2["requests"] == [], met2["requests"][:1])
        # 9b. probes (monitor: "Reply with exactly: ok", max_tokens <= 16) run, but do not count in the history
        probe = {"model": "tobu", "max_tokens": 16, "stream": False,
                 "messages": [{"role": "user", "content": "Reply with exactly: ok"}]}
        sp, jp, _ = chat(base, probe)
        sh, jh, _ = chat(base, {**oa(), "max_tokens": 1500})
        mt = lambda q="": json.loads(urllib.request.urlopen(base + "/tobu/metrics" + q, timeout=10).read())  # noqa: E731
        hs = [x["harness"] for x in mt(f"?since={since}")["requests"]]
        check("9b Probe beantwortet, harness probe, nicht in der Historie",
              sp == 200 and (jp.get("usage") or {}).get("tobu", {}).get("harness") == "probe" and "probe" not in hs
              and len(hs) == 1, (hs, (jp.get("usage") or {}).get("tobu")))
        check("9b ?probes=1 und ?harness=probe zeigen sie",
              [x["harness"] for x in mt(f"?since={since}&probes=1")["requests"]].count("probe") == 1
              and len(mt(f"?since={since}&harness=probe")["requests"]) == 1)
        check("9b ?client_ip= filtert", len(mt(f"?since={since}&client_ip=127.0.0.1")["requests"]) == 1
              and mt(f"?since={since}&client_ip=203.0.113.9")["requests"] == [])
        check("9b ?limit= begrenzt, Standard bis 50", len(mt("?limit=1")["requests"]) == 1
              and len(mt()["requests"]) == min(50, len(mt("?limit=200")["requests"])))
        # a user message with a content list (images, Anthropic blocks) must not disturb probe detection
        sl, jl, _ = chat(base, {"model": "tobu", "max_tokens": 16, "stream": False,
                                "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply with exactly: ok"}]}]})
        sa, ja = post(base + "/v1/messages", {"model": "tobu", "max_tokens": 16, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Reply with exactly: ok"}]}]}, AV)
        check("9b Inhaltsliste (OpenAI und Anthropic): beantwortet, keine Probe",
              sl == 200 and sa == 200 and (jl.get("usage") or {}).get("tobu", {}).get("harness") != "probe", (sl, sa))
        check("9b pending nach allen Anfragen 0 (Proben/Bilder nicht gezählt)", httpd.B.pending == 0, httpd.B.pending)
    finally:
        httpd.shutdown()
    # 3b. model ends by itself: a large budget changes nothing
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok, natural=200))
    try:
        s, j0, _ = chat(base, oa())
        s1, j1, log = chat(base, oa(thinking_budget=4096))
        same = (j0["tobu"]["out_hash"] == j1["tobu"]["out_hash"]
                and j0["choices"][0]["message"] == j1["choices"][0]["message"])
        check("3 Modell endet selbst: mit Budget 4096 dieselben Tokens wie ohne, hit nein",
              s == s1 == 200 and same and j1["tobu"]["thinking_budget"]["hit"] is False
              and "reached no" in log, (j0.get("tobu"), j1.get("tobu")))
    finally:
        httpd.shutdown()
    # 4. --default-thinking-budget
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok), "--default-thinking-budget", "32")
    try:
        s, j, _ = chat(base, oa())
        check("4 --default-thinking-budget 32 ohne Angabe: greift",
              s == 200 and j["tobu"].get("thinking_budget", {}).get("budget") == 32
              and j["choices"][0]["message"].get("content") == ANSWER, j)
        s, j, _ = chat(base, oa(max_tokens=200, thinking_budget=0))
        check("4 Anfrage thinking_budget 0 schaltet den Standard ab", s == 200 and
              j["choices"][0]["finish_reason"] == "length" and "thinking_budget" not in j["tobu"], j)
    finally:
        httpd.shutdown()
        api.DEFAULT_THINKING_BUDGET = 0
    # 5. json_schema: grammar only after the thinking block
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok, answer=JSON_ANSWER))
    try:
        rf = {"type": "json_schema", "json_schema": {"name": "a", "schema": SCHEMA}}
        s, j, _ = chat(base, oa(response_format=rf, thinking_budget=80))
        msg = j["choices"][0]["message"] if s == 200 else {}
        try:
            val = json.loads(msg.get("content") or "")
        except json.JSONDecodeError:
            val = None
        check("5 json_schema nach erzwungenem Ende: gültiges JSON nach Schema, Grammatik aktiv, hit",
              s == 200 and val == {"answer": 42, "ok": True} and j["timings"].get("tool_grammar") is True
              and j["tobu"]["thinking_budget"]["hit"] and j["choices"][0]["finish_reason"] == "stop"
              and j["usage"]["completion_tokens_details"]["reasoning_tokens"] == 81, j)
        s, j, _ = chat(base, oa(response_format=rf, thinking_budget=80, temperature=0.6, seed=1))
        msg = j["choices"][0]["message"] if s == 200 else {}
        check("5 json_schema + Sampling nach erzwungenem Ende: gültiges JSON",
              s == 200 and msg.get("content") == JSON_ANSWER and j["tobu"]["thinking_budget"]["hit"], j)
    finally:
        httpd.shutdown()
    # 5b. tool grammar (lazy, tool_choice required strict): tool call after forced end
    call = ("<tool_call>\n<function=read>\n<parameter=path>\nsrc/a.ts\n</parameter>\n</function>\n</tool_call>")
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok, answer=call))
    try:
        for tc in ("auto", "required"):
            s, j, _ = chat(base, oa(tools=OA_TOOLS, tool_choice=tc, thinking_budget=30))
            calls = j["choices"][0]["message"].get("tool_calls") or [] if s == 200 else []
            check(f"5 Tool-Aufruf nach erzwungenem Ende (tool_choice {tc}): read(src/a.ts), Grammatik aktiv",
                  s == 200 and j["choices"][0]["finish_reason"] == "tool_calls" and len(calls) == 1
                  and json.loads(calls[0]["function"]["arguments"]) == {"path": "src/a.ts"}
                  and j["timings"].get("tool_grammar") is True and j["tobu"]["thinking_budget"]["hit"], j)
    finally:
        httpd.shutdown()


def test_effort():
    """reasoning_effort -> budget, an explicit thinking_budget/max_thinking_tokens wins."""
    import api
    m = [{"role": "user", "content": "hi"}]
    for b, want in [({}, None), ({"reasoning_effort": "low"}, 4096), ({"reasoning_effort": "medium"}, 16384),
                    ({"reasoning_effort": "high"}, 16384), ({"reasoning_effort": "xhigh"}, 24576),
                    ({"chat_template_kwargs": {"reasoning_effort": "medium"}}, 16384),
                    ({"reasoning_effort": "low", "thinking_budget": 64}, 64),
                    ({"reasoning_effort": "high", "max_thinking_tokens": 99}, 99)]:
        got = api.parse_openai({"model": "x", "messages": m, **b}).thinking_budget
        check(f"8 reasoning_effort {b} -> Budget {want}", got == want, got)
    # Flash-Next: template level (default medium, high -> medium), Bonsai: high -> xhigh, no default
    old = api.EFFORT_DEFAULT, api.EFFORT_TEMPLATE
    try:
        api.EFFORT_DEFAULT, api.EFFORT_TEMPLATE = "medium", {"high": "medium"}
        for b, want in [({}, "medium"), ({"reasoning_effort": "high"}, "medium"), ({"reasoning_effort": "xhigh"}, "xhigh"),
                        ({"reasoning_effort": "low"}, "low"), ({"chat_template_kwargs": {"enable_thinking": False}}, None)]:
            r = api.parse_openai({"model": "x", "messages": m, **b})
            check(f"8 Flash-Next Vorlage {b} -> effort {want}", r.effort == want, r.effort)
        r = api.parse_openai({"model": "x", "messages": m, "reasoning_effort": "none"})
        check("8 Flash-Next reasoning_effort none -> Denken aus", r.thinking is False, r.thinking)
    finally:
        api.EFFORT_DEFAULT, api.EFFORT_TEMPLATE = old
    r = api.parse_openai({"model": "x", "messages": m, "reasoning_effort": "high"})
    check("8 Bonsai: high -> xhigh", r.effort == "xhigh", r.effort)


if __name__ == "__main__":
    from hftok import HFTok

    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        make_tokenizer(tokfile)
        ans = Path(td) / "answers.json"
        ans.write_text(json.dumps(ANSWERS, ensure_ascii=False), encoding="utf-8")
        test_wrapper(HFTok.from_file(str(tokfile)))
        test_http(tokfile, str(ans))
    test_effort()
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
