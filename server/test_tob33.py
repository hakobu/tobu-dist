"""TOB-33: guard the thinking budget against max_tokens (reserve for the answer) - no GPU, no model, no network.

    python test_tob33.py          (venv with numpy, tokenizers)

Test engine "Thinker" from test_tob32 (thinks without limit, after </think> the answer).

  1. effective_budget: capped at max_tokens - reserve, reserve at least 25 % of max_tokens, without capping
     unchanged, max_tokens < reserve -> 0 (no error).
  2. Wrapper with budget 0: thinking ends immediately ("\\n</think>\\n\\n" + answer), also with drafts.
  3. HTTP OpenAI: thinking_budget >= max_tokens -> capped, answer present, tobu.thinking_budget.capped, log line;
     without capping no new fields and no log; max_tokens < reserve -> budget 0, answer present (also json_schema,
     sampling); --thinking-answer-reserve takes effect.
  4. Anthropic thinking.budget_tokens >= max_tokens: capped, answer present, stream == non-stream; budget 0 without
     thinking block.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_tob2 import ANSWERS, make_tokenizer, post, sse  # noqa: E402
from test_tob32 import ANSWER, JSON_ANSWER, SCHEMA, Thinker, chat, oa, serve  # noqa: E402

FAILS: list[str] = []


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}{(' – ' + str(info)[:600]) if info and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def test_effective():
    from think_budget import answer_reserve, effective_budget

    check("1 Reserve: Standard 1024, mindestens 25 % von max_tokens",
          answer_reserve(1500) == 1024 and answer_reserve(16384) == 4096 and answer_reserve(4097) == 1025
          and answer_reserve(1500, 0) == 375 and answer_reserve(100, 0) == 25)
    check("1 thinking_budget >= max_tokens wird gekappt",
          effective_budget(4096, 4096) == (3072, True, 1024) and effective_budget(100000, 16384) == (12288, True, 4096)
          and effective_budget(1500, 1500) == (476, True, 1024))
    check("1 ohne Kappung unverändert", effective_budget(64, 1500) == (64, False, 1024)
          and effective_budget(4096, 16384) == (4096, False, 4096) and effective_budget(476, 1500) == (476, False, 1024))
    check("1 max_tokens kleiner als die Reserve: Budget 0, kein Fehler",
          effective_budget(100, 600) == (0, True, 1024) and effective_budget(5, 1) == (0, True, 1024)
          and effective_budget(5, 0) == (0, True, 1024))


def test_wrapper_zero(tok):
    import random

    from decoder import generate
    import think_budget

    te, tc = tok.t.token_to_id("</think>"), tok.t.token_to_id("<tool_call>")
    close, close_nl = tok.encode("\n</think>\n\n"), tok.encode("</think>\n\n")
    nl = lambda t: tok.decode([t]).endswith("\n")   # noqa: E731
    m = Thinker(tok)

    def rand_drafts(inner, seed):
        r = random.Random(seed)

        def self_draft(field="free", kind="text"):
            x = r.random()
            return (list(close) + tok.encode("Die") if x < 0.4 else
                    [r.randint(300, 6000) for _ in range(r.randint(1, 6))] if x < 0.8 else [])
        inner.self_draft = self_draft
        return inner

    ref = None
    for seed in (None, 1, 2):
        inner = m.make([]) if seed is None else rand_drafts(m.make([]), seed)
        v = think_budget.ThinkBudget(inner, 0, {te, tc}, close, close_nl, nl)
        toks = generate(v, [1], n=500, stop={m.eos}, use_self=seed is not None).tokens
        txt = tok.decode(toks)
        if ref is None:
            ref = toks
            check("2 Budget 0: Denken sofort beendet, Ausgabe = \\n</think>\\n\\n + Antwort",
                  txt == "\n</think>\n\n" + ANSWER + "<|im_end|>" and v.hit and v.n_replaced >= 1, txt[:120])
        else:
            check(f"2 Budget 0, Entwürfe (Seed {seed}): dieselben Tokens", toks == ref, txt[:120])


def test_http(tokfile: Path, answers: str):
    import api

    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok))
    try:
        # capping: budget 2000 >= max_tokens 1500 -> 476 (reserve 1024)
        s, j, log = chat(base, oa(thinking_budget=2000))
        msg = j["choices"][0]["message"] if s == 200 else {}
        tb = (j.get("tobu") or {}).get("thinking_budget") or {}
        check("3 thinking_budget >= max_tokens: gekappt, Antwort da, finish stop",
              s == 200 and msg.get("content") == ANSWER and j["choices"][0]["finish_reason"] == "stop", j)
        check("3 tobu.thinking_budget: capped true, wirksames Budget 476, requested 2000, Reserve 1024",
              tb.get("capped") is True and tb.get("budget") == 476 and tb.get("requested") == 2000
              and tb.get("reserve") == 1024 and tb.get("hit") is True
              and j["usage"]["completion_tokens_details"]["reasoning_tokens"] == 477, tb)
        check("3 Log-Zeile zur Kappung", "thinking budget capped: 2000 -> 476" in log
              and "thinking budget 476: reached yes" in log, log)
        # without capping unchanged
        s, j, log = chat(base, oa(thinking_budget=64))
        tb = j["tobu"]["thinking_budget"]
        check("3 ohne Kappung: Budget 64 unverändert, kein capped, kein Log zur Kappung",
              s == 200 and tb == {"budget": 64, "hit": True, "reasoning_tokens": 65}
              and "capped" not in log and j["choices"][0]["message"].get("content") == ANSWER, (tb, log))
        # max_tokens < reserve: budget 0
        s, j, log = chat(base, oa(max_tokens=600, thinking_budget=100))
        msg = j["choices"][0]["message"] if s == 200 else {}
        tb = (j.get("tobu") or {}).get("thinking_budget") or {}
        check("3 max_tokens 600 < Reserve: Budget 0, kein Fehler, Denken sofort beendet, Antwort da",
              s == 200 and msg.get("content") == ANSWER and not (msg.get("reasoning_content") or "").strip()
              and tb.get("budget") == 0 and tb.get("capped") is True and j["choices"][0]["finish_reason"] == "stop"
              and "thinking budget capped: 100 -> 0" in log, (j, log))
        s, j, _ = chat(base, oa(max_tokens=600, thinking_budget=100, temperature=0.7, seed=5))
        check("3 Budget 0 mit Sampling: Antwort da",
              s == 200 and j["choices"][0]["message"].get("content") == ANSWER
              and j["tobu"]["thinking_budget"]["budget"] == 0, j)
        # without budget: reserve has no effect (previous behaviour)
        s, j, log = chat(base, oa(max_tokens=300))
        check("3 ohne Budget: keine Kappung, bisheriges Verhalten (finish length)",
              s == 200 and j["choices"][0]["finish_reason"] == "length" and "thinking_budget" not in j["tobu"]
              and "thinking budget" not in log, j)
        # Anthropic
        an = {"model": "tobu", "max_tokens": 1200, "temperature": 0,
              "thinking": {"type": "enabled", "budget_tokens": 1200},
              "messages": [{"role": "user", "content": "Wie viel ist sechs mal sieben?"}]}
        AV = {"anthropic-version": "2023-06-01"}
        with contextlib.redirect_stdout(io.StringIO()):
            s, j = post(base + "/v1/messages", an, AV)
            s2, raw = post(base + "/v1/messages", {**an, "stream": True}, AV, raw=True)
        bl = [b["type"] for b in j.get("content", [])] if s == 200 else []
        tb = (j.get("tobu") or {}).get("thinking_budget") or {}
        check("4 Anthropic budget_tokens == max_tokens: gekappt auf 176, Text = Antwort, end_turn",
              s == 200 and bl == ["thinking", "text"] and j["content"][1]["text"] == ANSWER
              and j["stop_reason"] == "end_turn" and tb.get("capped") is True and tb.get("budget") == 176, j)
        txt = "".join(d["delta"].get("text", "") for e, d in sse(raw) if e == "content_block_delta")
        check("4 Anthropic Stream: dieselbe Antwort", s2 == 200 and txt == ANSWER, raw[-400:])
        with contextlib.redirect_stdout(io.StringIO()):
            s, j = post(base + "/v1/messages", {**an, "max_tokens": 500, "thinking": {"type": "enabled",
                                                                                     "budget_tokens": 100}}, AV)
        check("4 Anthropic max_tokens < Reserve: Budget 0, kein thinking-Block, nur Text = Antwort",
              s == 200 and j["content"] == [{"type": "text", "text": ANSWER}]
              and j["tobu"]["thinking_budget"]["budget"] == 0, j)
    finally:
        httpd.shutdown()
    # --thinking-answer-reserve
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok), "--thinking-answer-reserve", "200")
    try:
        s, j, _ = chat(base, oa(thinking_budget=2000))
        tb = j["tobu"]["thinking_budget"]
        check("3 --thinking-answer-reserve 200: Reserve max(200, 25 % von 1500) = 375 -> Budget 1125",
              s == 200 and tb.get("budget") == 1125 and tb.get("reserve") == 375
              and j["choices"][0]["message"].get("content") == ANSWER, tb)
    finally:
        httpd.shutdown()
        api.THINKING_ANSWER_RESERVE = 1024
    # json_schema with budget 0
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok, answer=JSON_ANSWER))
    try:
        rf = {"type": "json_schema", "json_schema": {"name": "a", "schema": SCHEMA}}
        s, j, _ = chat(base, oa(max_tokens=500, response_format=rf, thinking_budget=50))
        msg = j["choices"][0]["message"] if s == 200 else {}
        try:
            val = json.loads(msg.get("content") or "")
        except json.JSONDecodeError:
            val = None
        check("3 json_schema mit Budget 0: gültiges JSON, Grammatik aktiv",
              s == 200 and val == {"answer": 42, "ok": True} and j["timings"].get("tool_grammar") is True
              and j["tobu"]["thinking_budget"]["budget"] == 0, j)
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    from hftok import HFTok

    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        make_tokenizer(tokfile)
        ans = Path(td) / "answers.json"
        ans.write_text(json.dumps(ANSWERS, ensure_ascii=False), encoding="utf-8")
        test_effective()
        test_wrapper_zero(HFTok.from_file(str(tokfile)))
        test_http(tokfile, str(ans))
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
