"""Script ban "no-CJK" (ban_scripts, --ban-scripts) – API/server wiring, without GPU, without model, without network.

    python test_ban_scripts.py          (venv with numpy, tokenizers)

The ban itself is computed by tobu-fnx in the sampler (engine/tobu/test_sampling_ban.cpp, engine/tobu/test_scripts.cpp); here
it is about the path there. Test engine "Thinker" from test_tob32, once with native_constraints (like flashnext: the
server passes the settings per request as a CONFIG dict to the verifier) and once without.

  1. api: validate ban_scripts (list of known scripts, [] allowed, otherwise ApiError); CJK_RE covers exactly the blocks
     of scripts.hpp (kCjk); ban_scripts_for: request before server default, default not with CJK in system,
     user or tool messages, but yes with CJK only in assistant messages.
  2. HTTP, engine with its own ban: without a value no field in the CONFIG (as before); ban_scripts ["cjk"] -> CONFIG
     and tobu.ban_scripts; unknown script / wrong type -> 400 (OpenAI and Anthropic).
  3. --ban-scripts cjk: default without a value; [] turns it off; CJK in the user message turns the default off,
     explicit ["cjk"] still applies; CJK only in the assistant history does not turn it off; Anthropic likewise.
  4. Engine without its own ban: ban_scripts (request or default) is listed in tobu.ignored, [] is not.
  5. --ban-scripts with an unknown script: startup aborts.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_tob2 import ANSWERS, make_tokenizer, post  # noqa: E402
from test_tob32 import Thinker, chat, serve  # noqa: E402

FAILS: list[str] = []
SCRIPTS_HPP = HERE.parent / "engine/tobu/scripts.hpp"
AV = {"anthropic-version": "2023-06-01"}


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FEHLER'} {name}{(' – ' + str(info)[:600]) if info and not ok else ''}", flush=True)
    if not ok:
        FAILS.append(name)


class Native(Thinker):
    """Thinker that enforces grammar/budget/sampling/ban itself like flashnext: remembers the verifiers whose
    `config` the server sets (_native_config)."""
    native_constraints = True

    def __init__(self, tok):
        super().__init__(tok, natural=40)
        self.made = []

    def make(self, prompt_ids):
        v = super().make(prompt_ids)
        v.config = None
        self.made.append(v)
        return v


def oa(content="Wie viel ist sechs mal sieben?", history=None, **kw):
    msgs = list(history or []) + [{"role": "user", "content": content}]
    return {"model": "tobu", "temperature": 0, "max_tokens": 300, "messages": msgs, **kw}


# ------------------------------------------------------------------------------------------------- 1: api
def test_api():
    import api

    def err(body):
        try:
            api._ban_scripts(body)
        except api.ApiError as e:
            return e.status == 400 and "ban_scripts" in str(e)
        return False

    check("1 ban_scripts: None/[]/[cjk]/doppelt", api._ban_scripts({}) is None
          and api._ban_scripts({"ban_scripts": None}) is None and api._ban_scripts({"ban_scripts": []}) == []
          and api._ban_scripts({"ban_scripts": ["cjk"]}) == ["cjk"]
          and api._ban_scripts({"ban_scripts": ["cjk", "cjk"]}) == ["cjk"])
    check("1 ban_scripts: unbekannt, String statt Liste, Zahl -> ApiError 400",
          err({"ban_scripts": ["latin"]}) and err({"ban_scripts": "cjk"}) and err({"ban_scripts": [1]})
          and err({"ban_scripts": ["cjk", "han"]}))

    # CJK_RE = kCjk from scripts.hpp (the same blocks in C++ and Python)
    src = SCRIPTS_HPP.read_text(encoding="utf-8")
    table = src[src.index("kCjk[] = {"):src.index("};", src.index("kCjk[] = {"))]
    ranges = [(int(a, 16), int(b, 16)) for a, b in re.findall(r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}", table)]
    diff = [cp for cp in range(0, 0x40001) if not 0xD800 <= cp <= 0xDFFF
            and bool(api.CJK_RE.match(chr(cp))) != any(a <= cp <= b for a, b in ranges)]
    check("1 CJK_RE deckt genau die Blöcke von scripts.hpp (kCjk) ab", len(ranges) > 20 and not diff,
          [hex(c) for c in diff[:10]])
    check("1 CJK_RE: 中 。 ， カ 한 ㄱ ja; ä — 😀 ䷀ (Yijing) nein",
          all(api.CJK_RE.search(c) for c in "中。，カ한ㄱ") and not any(api.CJK_RE.search(c) for c in "ä—😀䷀"))

    def req(msgs, bs=None):
        return api.Req("openai", msgs, None, False, ban_scripts=bs)

    user = [{"role": "user", "content": "Hallo"}]
    old = api.DEFAULT_BAN_SCRIPTS
    try:
        api.DEFAULT_BAN_SCRIPTS = []
        check("1 ohne Standard: nichts, ausdrücklich [cjk] gilt",
              api.ban_scripts_for(req(user)) == [] and api.ban_scripts_for(req(user, ["cjk"])) == ["cjk"])
        api.DEFAULT_BAN_SCRIPTS = ["cjk"]
        check("1 Standard [cjk] ohne Angabe; [] schaltet ab",
              api.ban_scripts_for(req(user)) == ["cjk"] and api.ban_scripts_for(req(user, [])) == [])
        cases = {
            "Nutzer": [{"role": "user", "content": "请用中文回答"}],
            "Nutzer (Teile)": [{"role": "user", "content": [{"type": "text", "text": "x"},
                                                             {"type": "text", "text": "日本語で"}]}],
            "System": [{"role": "system", "content": "Antworte auf 한국어."}, *user],
            "Werkzeug": [*user, {"role": "assistant", "content": "", "tool_calls": []},
                         {"role": "tool", "tool_call_id": "1", "content": "// 注释"}],
        }
        for k, msgs in cases.items():
            check(f"1 CJK in {k}-Nachricht: kein Standard, ausdrücklich [cjk] gilt",
                  api.ban_scripts_for(req(msgs)) == [] and api.ban_scripts_for(req(msgs, ["cjk"])) == ["cjk"])
        drift = [*user, {"role": "assistant", "content": "好的，我来看看。", "reasoning_content": "嗯"}, *user]
        check("1 CJK nur im Assistant-Verlauf (Abdrift): Standard gilt weiter", api.ban_scripts_for(req(drift)) == ["cjk"])
    finally:
        api.DEFAULT_BAN_SCRIPTS = old
    r = api.parse_openai(oa(ban_scripts=["cjk"]))
    a = api.parse_anthropic({"model": "tobu", "max_tokens": 10, "ban_scripts": [],
                             "messages": [{"role": "user", "content": "x"}]})
    check("1 parse_openai/parse_anthropic übernehmen ban_scripts", r.ban_scripts == ["cjk"] and a.ban_scripts == []
          and api.parse_openai(oa()).ban_scripts is None)


# ------------------------------------------------------------------------------------------------- 2–5: HTTP
def last_config(eng):
    return eng.made[-1].config if eng.made else "kein Verifier"


def an(content="Wie viel ist sechs mal sieben?", **kw):
    return {"model": "tobu", "max_tokens": 300, "temperature": 0, "messages": [{"role": "user", "content": content}],
            **kw}


def test_http(tokfile: Path, answers: str):
    import api

    eng = {}

    def native(tok):
        eng["n"] = Native(tok)
        return eng["n"]

    httpd, base = serve(tokfile, answers, native)
    try:
        n = eng["n"]
        s, j, _ = chat(base, oa())
        cfg = last_config(n)
        check("2 ohne Angabe: CONFIG ohne ban_scripts (wie bisher), kein tobu.ban_scripts",
              s == 200 and isinstance(cfg, dict) and "ban_scripts" not in cfg and "ban_scripts" not in j["tobu"]
              and "ban_scripts" not in j["tobu"].get("ignored", []), (s, cfg, j.get("tobu")))
        s, j, _ = chat(base, oa(ban_scripts=["cjk"]))
        cfg = last_config(n)
        check("2 ban_scripts [cjk]: CONFIG und tobu.ban_scripts, nicht ignoriert",
              s == 200 and cfg.get("ban_scripts") == ["cjk"] and j["tobu"].get("ban_scripts") == ["cjk"]
              and "ban_scripts" not in j["tobu"].get("ignored", []), (s, cfg, j.get("tobu")))
        # with sampling (directly: the HTTP path with sampling still fails for engines with their own ban at the
        # server.py diagnostic v.n_forced, independent of ban_scripts)
        import types

        import server

        rq = api.parse_openai(oa(ban_scripts=["cjk"], temperature=0.7, seed=3))
        cfg = server.Backend._native_config(types.SimpleNamespace(tool_grammar=True), rq, {})
        check("2 mit Sampling: ban_scripts neben den Sampling-Feldern in der CONFIG",
              cfg.get("ban_scripts") == ["cjk"] and cfg.get("temperature") == 0.7 and cfg.get("seed") == 3, cfg)
        for bad in (["latin"], "cjk", [1]):
            s, j, _ = chat(base, oa(ban_scripts=bad))
            check(f"2 OpenAI ban_scripts {bad!r} -> 400",
                  s == 400 and "ban_scripts" in json.dumps(j), (s, j))
        with contextlib.redirect_stdout(io.StringIO()):
            s, j = post(base + "/v1/messages", an(ban_scripts=["kana"]), AV)
        check("2 Anthropic unbekannte Schrift -> 400 invalid_request_error",
              s == 400 and j.get("error", {}).get("type") == "invalid_request_error", (s, j))
        with contextlib.redirect_stdout(io.StringIO()):
            s, j = post(base + "/v1/messages", an(ban_scripts=["cjk"]), AV)
        check("2 Anthropic ban_scripts [cjk] -> CONFIG", s == 200 and last_config(n).get("ban_scripts") == ["cjk"]
              and j["tobu"].get("ban_scripts") == ["cjk"], (s, j))
    finally:
        httpd.shutdown()

    httpd, base = serve(tokfile, answers, native, "--ban-scripts", "cjk")
    try:
        n = eng["n"]
        s, j, _ = chat(base, oa())
        check("3 --ban-scripts cjk: Standard ohne Angabe", s == 200 and last_config(n).get("ban_scripts") == ["cjk"]
              and j["tobu"].get("ban_scripts") == ["cjk"], (s, last_config(n)))
        s, j, _ = chat(base, oa(ban_scripts=[]))
        check("3 ban_scripts [] schaltet den Standard ab", s == 200 and "ban_scripts" not in last_config(n)
              and "ban_scripts" not in j["tobu"], last_config(n))
        s, j, _ = chat(base, oa("请问六乘以七是多少？"))
        check("3 CJK in der Nutzernachricht: kein Standard", s == 200 and "ban_scripts" not in last_config(n),
              last_config(n))
        s, j, _ = chat(base, oa("请问六乘以七是多少？", ban_scripts=["cjk"]))
        check("3 CJK in der Nutzernachricht, ausdrücklich [cjk]: gilt",
              s == 200 and last_config(n).get("ban_scripts") == ["cjk"], last_config(n))
        hist = [{"role": "user", "content": "Fasse zusammen."}, {"role": "assistant", "content": "好的，总结如下。"}]
        s, j, _ = chat(base, oa("Weiter auf Deutsch.", history=hist))
        check("3 CJK nur im Assistant-Verlauf: Standard gilt", s == 200
              and last_config(n).get("ban_scripts") == ["cjk"], last_config(n))
        with contextlib.redirect_stdout(io.StringIO()):
            s, j = post(base + "/v1/messages", an(), AV)
            s2, _ = post(base + "/v1/messages", an("日本語で答えて"), AV)
        check("3 Anthropic: Standard gilt; mit CJK in der Nachricht nicht",
              s == 200 and s2 == 200 and "ban_scripts" not in last_config(n), (s, s2, last_config(n)))
    finally:
        httpd.shutdown()
        api.DEFAULT_BAN_SCRIPTS = []

    # 4: engine without its own ban
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok, natural=40))
    try:
        s, j, _ = chat(base, oa(ban_scripts=["cjk"]))
        check("4 Engine ohne Sperre: ban_scripts in tobu.ignored", s == 200
              and "ban_scripts" in j["tobu"].get("ignored", []) and "ban_scripts" not in j["tobu"], j.get("tobu"))
        s, j, _ = chat(base, oa(ban_scripts=[]))
        check("4 Engine ohne Sperre, []: nicht in tobu.ignored", s == 200
              and "ban_scripts" not in j["tobu"].get("ignored", []), j.get("tobu"))
    finally:
        httpd.shutdown()
    httpd, base = serve(tokfile, answers, lambda tok: Thinker(tok, natural=40), "--ban-scripts", "cjk")
    try:
        s, j, _ = chat(base, oa())
        check("4 Engine ohne Sperre, Standard --ban-scripts: in tobu.ignored", s == 200
              and "ban_scripts" in j["tobu"].get("ignored", []), j.get("tobu"))
    finally:
        httpd.shutdown()
        api.DEFAULT_BAN_SCRIPTS = []

    # 5
    import server

    try:
        server.parse_ban_scripts("cjk, latin")
        ok = False
    except SystemExit as e:
        ok = "latin" in str(e)
    check("5 --ban-scripts mit unbekannter Schrift: Abbruch; leer/doppelt ok", ok and server.parse_ban_scripts("") == []
          and server.parse_ban_scripts("cjk,cjk") == ["cjk"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tokfile = Path(td) / "tokenizer.json"
        make_tokenizer(tokfile)
        ans = Path(td) / "answers.json"
        ans.write_text(json.dumps(ANSWERS, ensure_ascii=False), encoding="utf-8")
        test_api()
        test_http(tokfile, str(ans))
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
